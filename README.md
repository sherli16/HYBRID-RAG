# HYBRID-RAG
Chat with your PDF using Hybrid RAG - FAISS + BM25 + Reranking

## What is this?
Upload any PDF and ask questions. App will search the PDF and answer ONLY from your document.

## How it Works? - Hybrid Search
1.  **FAISS (Dense):** Searches by Meaning
2.  **BM25 (Sparse):** Searches by Keywords
3.  **FlashRank (Reranker):** Re-ranks Top Results and picks Best 3
4.  **Gemini 3 Flash:** Generates Answer from Top 3 Chunks

## Features
- Upload Multiple PDFs
- 1500 Chunk Size with 150 Overlap
- Local Embeddings - `all-MiniLM-L6-v2` (No API Quota)
- Shows Sources with Page Number + Score
- Says "Not found in the document" if answer is not there

## Setup
# 1. Create Virtual Env
python -m venv myenv
myenv\Scripts\activate

# 2. Create .env file
GOOGLE_API_KEY=your_api_key_here

# 3. Run App
streamlit run "Pdf file.py"
