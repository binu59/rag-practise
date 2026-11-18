import os
import time
from pathlib import Path

from pinecone import Pinecone

INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-practice")
NAMESPACE = "docs"
DOCS_DIR = Path(__file__).parent / "docs"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def chunk_text(text: str, size: int = 60):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


# 1. Create the index (Pinecone embeds the "chunk_text" field automatically)
if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"},
        },
    )
    print("Index created")

index = pc.Index(INDEX_NAME)

# 2. Build records from the docs folder
records = []
for f in DOCS_DIR.glob("*.txt"):
    for i, chunk in enumerate(chunk_text(f.read_text(encoding="utf-8"))):
        records.append({
            "_id": f"{f.stem}-{i}",
            "chunk_text": chunk,
            "source": f.name,
        })

# 3. Upload in batches
for i in range(0, len(records), 90):
    index.upsert_records(namespace=NAMESPACE, records=records[i:i + 90])

print(f"Uploaded {len(records)} chunks. Waiting for them to become searchable...")
time.sleep(10)
print(index.describe_index_stats())