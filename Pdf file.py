import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

load_dotenv()

st.title("Hybrid AI PDF Assistant")
st.write("FAISS + BM25 + Reranking")

@st.cache_resource
def load_reranker():
    return Ranker()

@st.cache_resource(show_spinner="Processing PDFs...")
def build_index(files):
    docs = []
    for name, data in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(data)
            path = f.name
        pages = PyPDFLoader(path).load()
        for p in pages:
            p.metadata["source"] = name
        docs.extend(pages)
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Local embedding model - no API quota, runs offline
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    BATCH = 20
    vectorstore = None
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        for attempt in range(4):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                    time.sleep(35)
                else:
                    raise
        if i + BATCH < len(chunks):
            time.sleep(2)

    bm25 = BM25Okapi([c.page_content.lower().split() for c in chunks])
    return chunks, vectorstore, bm25

pdfs = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)

if pdfs:
    files = tuple((p.name, p.getvalue()) for p in pdfs)
    try:
        chunks, vectorstore, bm25 = build_index(files)
    except Exception as e:
        st.error(f"Index error: {e}")
        st.stop()

    reranker = load_reranker()
    st.success(f"PDF is ready! {len(chunks)} chunks created. Ask your question.")

    question = st.text_input("Enter your question:")
    if question:
        # Dense retrieval using FAISS - top 5 results
        dense_docs = vectorstore.similarity_search(question, k=5)

        # Sparse retrieval using BM25 - top 5 results
        scores = bm25.get_scores(question.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        sparse_docs = [chunks[i] for i in top_idx]

        # Remove duplicate documents
        seen, unique_docs = set(), []
        for d in dense_docs + sparse_docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)

        # Rerank results using FlashRank
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(unique_docs)]
        results = reranker.rerank(RerankRequest(query=question, passages=passages))
        ranked = [(unique_docs[r["id"]], float(r["score"])) for r in results]
        top_3 = [d for d, _ in ranked[:3]]

        # Build prompt from top 3 relevant chunks
        context = "\n\n".join(d.page_content for d in top_3)
        prompt = (
            "Answer ONLY from the context below. "
            "If the answer is not in the context, say 'Not found in the document'.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        try:
            # Gemini Flash model for fast answers
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
            result = llm.invoke(prompt)
        except Exception as e:
            st.error(f"LLM error: {e}")
            st.stop()
        st.subheader("Answer:")
# Extract only text from Gemini response-signature ignore
        answer_text = result.content
        if isinstance(answer_text, list):
            answer_text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in answer_text)
        st.write(answer_text)
        

        st.subheader("Sources:")
        for d, s in ranked[:3]:
            page = d.metadata.get("page", 0) + 1
            st.write(f"**{d.metadata.get('source')}**, page {page} (score {round(s, 2)})")
            st.caption(d.page_content[:150] + "...")