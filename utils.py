import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from io import BytesIO

# Load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Text Extraction
# -------------------------
def extract_text(content: bytes, filename: str) -> str:
    """
    Extract text from PDF or TXT files.
    """
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

    # TXT fallback
    return content.decode("utf-8", errors="ignore")


# -------------------------
# Chunking
# -------------------------
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# -------------------------
# Background Processing
# -------------------------
def process_document(doc_id, content, filename, store):
    """
    Background task:
    - extract text
    - chunk
    - embed
    - store in FAISS
    """
    text = extract_text(content, filename)

    # If no readable text
    if not text.strip():
        store[doc_id] = (None, [])
        return

    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = embedder.encode(chunks)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    store[doc_id] = (index, chunks)


# -------------------------
# Retrieval + Answering
# -------------------------
def retrieve_and_generate(query, index, chunks):
    """
    Retrieve relevant chunks and generate a simple answer.
    (No external LLM – deterministic output)
    """
    if index is None or not chunks:
        return "No readable content found in the document."

    # Embed query
    query_embedding = embedder.encode([query])

    # Search top 5 chunks
    _, indices = index.search(query_embedding, 5)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = " ".join(retrieved_chunks)

    return (
        "Answer based on retrieved document content:\n\n"
        + context[:1000]
    )
