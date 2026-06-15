from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from rag import extract_text, chunk_text, load_or_build_index, find_relevant_chunks, answer_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    pdf_name: str
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # save uploaded file to disk
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # build FAISS index for this PDF
    text = extract_text(file_path)
    chunks = chunk_text(text)
    index, chunks = load_or_build_index(chunks, file.filename)
    
    return {
        "message": f"PDF processed successfully",
        "pdf_name": file.filename,
        "chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    file_path = f"uploads/{request.pdf_name}"
    
    if not os.path.exists(file_path):
        return {"error": "PDF not found. Please upload it first."}
    
    text = extract_text(file_path)
    chunks = chunk_text(text)
    index, chunks = load_or_build_index(chunks, request.pdf_name)
    
    relevant = find_relevant_chunks(request.question, index, chunks)
    answer = answer_question(request.question, relevant)
    
    return {
        "question": request.question,
        "answer": answer,
        "pdf": request.pdf_name
    }

@app.get("/")
def root():
    return {"message": "PDF Chatbot API running"}