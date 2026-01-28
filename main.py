from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from models import UploadResponse, QueryRequest, QueryResponse
from utils import process_document, retrieve_and_generate
import uuid

app = FastAPI(title="RAG QA System")
doc_stores = {}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    doc_id = str(uuid.uuid4())
    content = await file.read()
    background_tasks.add_task(process_document, doc_id, content, file.filename, doc_stores)
    return UploadResponse(doc_id=doc_id, message="Document uploaded")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if request.doc_id not in doc_stores:
        raise HTTPException(status_code=404, detail="Document not ready")
    index, chunks = doc_stores[request.doc_id]
    answer = retrieve_and_generate(request.query, index, chunks)
    return QueryResponse(answer=answer)
