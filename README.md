## RAG-Based Question Answering System
This project is a Retrieval-Augmented Generation (RAG) based Question Answering system built using FastAPI.
It allows users to upload documents and ask questions, and the system answers only based on the uploaded document content.

## Features
1. Supports PDF and TXT documents
2. Background document processing
3.Text chunking and embedding generation
4.Retrieval-based question answering
5.REST API with request validation and rate limiting

## Tech Stack
fastapi
uvicorn
faiss-cpu
sentence-transformers
openai
pydantic
python-multipart
PyPDF2
slowapi
python-dotenv

## Setup Instructions
Step 1: Run the command in Command Prompt:
pip install -r requirements.txt

Step 2: Start the Server
uvicorn main:app --reload
After running this command, a local URL (e.g. http://127.0.0.1:8000) will appear in the terminal.
Open this link in your browser and go to /docs to access the API interface.

## Steps to work on it 
1. Upload Document
Endpoint: POST /upload
Upload a PDF or TXT file
This  response will return a doc_id
2. Ask a Question
Endpoint: POST /ask
Provide your question along with the doc_id
The system retrieves relevant content and returns an answer

## Design Notes

Chunk Size: 512 characters for balanced context and retrieval accuracy
Failure Case: Out-of-scope questions may return weak answers
Metric Tracked: Query latency