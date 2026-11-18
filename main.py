import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from groq import Groq

app = FastAPI()

DOCS_DIR = Path(__file__).parent / "docs"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def tokenize(text: str):
    return text.lower().split()


def chunk_text(text: str, size: int = 60):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


# Build the index once at startup
chunks = []
for f in DOCS_DIR.glob("*.txt"):
    for c in chunk_text(f.read_text(encoding="utf-8")):
        chunks.append({"source": f.name, "text": c})

bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(chunks)}


@app.post("/ask")
def ask(body: Question):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")
    if bm25 is None:
        raise HTTPException(status_code=500, detail="No documents loaded")

    # 1. Retrieve
    top = bm25.get_top_n(tokenize(body.question), chunks, n=3)
    context = "\n\n".join(c["text"] for c in top)

    # 2. Generate
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {body.question}"
    )

    client = Groq(api_key=api_key)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": [c["source"] for c in top],
    }