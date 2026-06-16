from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from rag import extract_text, chunk_text, load_or_build_index, find_relevant_chunks, answer_question, get_index_path
import json
import faiss
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



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
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    text = extract_text(file_path)
    chunks = chunk_text(text)
    index, chunks = load_or_build_index(chunks, file_path)  # ← file_path
    
    return {
        "message": "PDF processed successfully",
        "pdf_name": file.filename,
        "chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    file_path = f"uploads/{request.pdf_name}"
    index_path, chunks_path = get_index_path(file_path)  # ← file_path
    
    if not os.path.exists(file_path):
        return {"error": "PDF not found. Please upload it first."}
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
    else:
        text = extract_text(file_path)
        chunks = chunk_text(text)
        index, chunks = load_or_build_index(chunks, file_path)  # ← file_path
    
    relevant = find_relevant_chunks(request.question, index, chunks)
    answer = answer_question(request.question, relevant)
    
    return {
        "question": request.question,
        "answer": answer,
        "pdf": request.pdf_name
    }

app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

@app.get("/")
def serve_react():
    return FileResponse("frontend/build/index.html")

@app.get("/")
def root():
    return {"message": "PDF Chatbot API running"}