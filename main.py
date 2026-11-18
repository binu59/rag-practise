import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from groq import Groq

app = FastAPI()

INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-practice")
NAMESPACE = "docs"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

_index = None


def get_index():
    """Connect to Pinecone on first use, so /health works even if keys are missing."""
    global _index
    if _index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not set")
        _index = Pinecone(api_key=api_key).Index(INDEX_NAME)
    return _index


class Question(BaseModel):
    question: str


def hit_score(h):
    """Read the similarity score, whatever the SDK version calls it."""
    for key in ("_score", "score"):
        try:
            return h[key]
        except (KeyError, TypeError):
            continue
    return None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
    }


@app.post("/ask")
def ask(body: Question):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

    # 1. Retrieve: Pinecone embeds the question and finds the closest chunks
    try:
        results = get_index().search(
            namespace=NAMESPACE,
            query={"top_k": 3, "inputs": {"text": body.question}},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone search failed: {e}")

    hits = results["result"]["hits"]
    if not hits:
        raise HTTPException(status_code=404, detail="No matching chunks found")

    context = "\n\n".join(h["fields"]["chunk_text"] for h in hits)

    # 2. Generate
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {body.question}"
    )
    try:
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return {
        "answer": resp.choices[0].message.content,
        "sources": [
            {"source": h["fields"].get("source"), "score": hit_score(h)} for h in hits
        ],
    }