import io
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pinecone import Pinecone
from pypdf import PdfReader
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

BASE_DIR = Path(__file__).parent
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-practice")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = 4
MAX_BYTES = 5 * 1024 * 1024  # 5 MB upload limit

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

_index = None


def get_index():
    """Connect to Pinecone on first use."""
    global _index
    if _index is None:
        key = (os.getenv("PINECONE_API_KEY") or "").strip()
        if not key:
            raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not set")
        _index = Pinecone(api_key=key).Index(INDEX_NAME)
    return _index


def get_groq():
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")
    return Groq(api_key=key)


def hit_score(h):
    for key in ("_score", "score"):
        try:
            return h[key]
        except (KeyError, TypeError):
            continue
    return None


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=400, detail="Only .pdf, .txt and .md files are supported")


def rewrite_question(client, question: str, history: list) -> str:
    """Turn a follow-up like 'when was it founded?' into a standalone question."""
    if not history:
        return question
    convo = "\n".join(f'{m.get("role")}: {m.get("content")}' for m in history[-6:])
    prompt = (
        "Given the conversation and a follow-up question, rewrite the follow-up "
        "as a standalone question. Replace pronouns like 'it' with what they refer to. "
        "Return only the rewritten question.\n\n"
        f"Conversation:\n{convo}\n\nFollow-up: {question}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return (resp.choices[0].message.content or "").strip() or question
    except Exception:
        return question  # if rewriting fails, fall back to the original question


class Question(BaseModel):
    question: str
    namespace: str
    history: list[dict] = []


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pinecone_key_set": bool((os.getenv("PINECONE_API_KEY") or "").strip()),
        "groq_key_set": bool((os.getenv("GROQ_API_KEY") or "").strip()),
    }


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB)")

    text = extract_text(file.filename or "", data).strip()
    if not text:
        raise HTTPException(
            status_code=400, detail="No text found in the file (scanned PDF?)"
        )

    chunks = splitter.split_text(text)
    namespace = f"doc-{uuid.uuid4().hex[:8]}"
    records = [
        {"_id": f"c{i}", "chunk_text": c, "source": file.filename}
        for i, c in enumerate(chunks)
    ]

    try:
        index = get_index()
        for i in range(0, len(records), 90):
            index.upsert_records(namespace=namespace, records=records[i:i + 90])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone upload failed: {e}")

    return {"namespace": namespace, "filename": file.filename, "chunks": len(chunks)}


@app.post("/ask")
def ask(body: Question):
    client = get_groq()

    # 1. Make follow-up questions standalone
    standalone = rewrite_question(client, body.question.strip(), body.history)

    # 2. Retrieve (new uploads take a few seconds to become searchable, so retry)
    hits = []
    for _ in range(3):
        try:
            results = get_index().search(
                namespace=body.namespace,
                query={"top_k": TOP_K, "inputs": {"text": standalone}},
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Pinecone search failed: {e}")
        hits = results["result"]["hits"]
        if hits:
            break
        time.sleep(3)

    if not hits:
        raise HTTPException(status_code=404, detail="No chunks found for this document")

    context = "\n\n".join(h["fields"]["chunk_text"] for h in hits)

    # 3. Generate
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {standalone}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return {
        "answer": resp.choices[0].message.content,
        "standalone_question": standalone,
        "sources": [
            {"source": h["fields"].get("source"), "score": hit_score(h)} for h in hits
        ],
    }