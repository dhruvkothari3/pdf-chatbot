from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv



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

def embed_chunks(chunks: list) -> list:
    embeddings = model.encode(chunks)
    return embeddings

from sentence_transformers import util

def find_relevant_chunks(question: str, chunks: list, chunk_embeddings, top_k: int = 2) -> list:
    question_embedding = model.encode(question)
    
    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
    
    top_results = similarities.argsort(descending=True)[:top_k]
    
    relevant_chunks = [chunks[i] for i in top_results]
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
    embeddings = embed_chunks(chunks)
    
    questions = [
    "What programming languages does this person know?",
    "What is this person's experience?",
    "Where is this person from?"
]

for q in questions:
    relevant = find_relevant_chunks(q, chunks, embeddings, top_k=2)
    answer = answer_question(q, relevant)
    print(f"Q: {q}")
    print(f"A: {answer}\n")