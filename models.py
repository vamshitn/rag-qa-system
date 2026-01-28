from pydantic import BaseModel

class UploadResponse(BaseModel):
    doc_id: str
    message: str

class QueryRequest(BaseModel):
    query: str
    doc_id: str

class QueryResponse(BaseModel):
    answer: str
