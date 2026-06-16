from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import json




def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# load the embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

import hashlib

def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_index_path(file_path: str) -> tuple:
    file_hash = get_file_hash(file_path)
    return f"{file_hash}_index.faiss", f"{file_hash}_chunks.json"



def build_faiss_index(chunks: list, file_path: str) -> tuple:
    embeddings = model.encode(chunks)
    embeddings_float = np.array(embeddings).astype('float32')
    
    index = faiss.IndexFlatL2(384)
    index.add(embeddings_float)
    
    index_path, chunks_path = get_index_path(file_path)
    
    faiss.write_index(index, index_path)
    
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)
    
    print(f"Index built: {index.ntotal} vectors stored")
    return index, chunks

def load_or_build_index(chunks: list, file_path: str):
    index_path, chunks_path = get_index_path(file_path)
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print("Loading existing index from disk...")
        index = faiss.read_index(index_path)
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
        return index, chunks
    else:
        print("Building new index...")
        return build_faiss_index(chunks, file_path)
    

def find_relevant_chunks(question: str, index, chunks: list, top_k: int = 2) -> list:
    question_vector = model.encode([question])
    question_float = np.array(question_vector).astype('float32')
    
    distances, indices = index.search(question_float, k=top_k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question: str, relevant_chunks: list) -> str:
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
    


   

if __name__ == "__main__":
    file_path = "test.pdf"
    text = extract_text(file_path)
    chunks = chunk_text(text)
    index, chunks = load_or_build_index(chunks, file_path)
    
    question = "What programming languages does this person know?"
    relevant = find_relevant_chunks(question, index, chunks)
    answer = answer_question(question, relevant)
    print(f"Q: {question}\nA: {answer}")