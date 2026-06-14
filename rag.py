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



def build_faiss_index(chunks: list):
    embeddings = model.encode(chunks)
    embeddings_float = np.array(embeddings).astype('float32')
    
    index = faiss.IndexFlatL2(384)
    index.add(embeddings_float)
    
    # save index to disk
    faiss.write_index(index, "index.faiss")
    
    # save chunks to disk so we can retrieve text later
    with open("chunks.json", "w") as f:
        json.dump(chunks, f)
    
    print(f"Index built: {index.ntotal} vectors stored")
    return index, chunks

def load_or_build_index(chunks: list):
    if os.path.exists("index.faiss") and os.path.exists("chunks.json"):
        print("Loading existing index from disk...")
        index = faiss.read_index("index.faiss")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)
        return index, chunks
    else:
        print("Building new index...")
        return build_faiss_index(chunks)

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
    text = extract_text("test.pdf")
    chunks = chunk_text(text)
    
    index, chunks = load_or_build_index(chunks)
    
    questions = [
        "What programming languages does this person know?",
        "What is this person's experience?",
        "Where is this person from?"
    ]
    
    for q in questions:
        relevant = find_relevant_chunks(q, index, chunks)
        answer = answer_question(q, relevant)
        print(f"Q: {q}")
        print(f"A: {answer}\n")