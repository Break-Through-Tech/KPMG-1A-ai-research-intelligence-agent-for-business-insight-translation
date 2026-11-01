"""Embedding + vector retrieval over the ingested chunks (ChromaDB + sentence-transformers)."""
from typing import Dict, List

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from data_ingestion import DATA_DIR, Chunk, ingest_directory

CHROMA_DIR = DATA_DIR / "chroma"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_models: Dict[str, object] = {}


def _embedder(name: str = EMBED_MODEL) -> SentenceTransformer:
    if name not in _models:
        _models[name] = SentenceTransformer(name)
    return _models[name]


def _reranker() -> CrossEncoder:
    if RERANK_MODEL not in _models:
        _models[RERANK_MODEL] = CrossEncoder(RERANK_MODEL)
    return _models[RERANK_MODEL]


def short_id(doc_id: str) -> str:
    """'2608.20316v1_Pandora ...' -> '2608.20316v1' (used for citations)."""
    return doc_id.split("_", 1)[0]


def _collection(embed_model: str = EMBED_MODEL):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = "chunks_" + embed_model.replace("/", "_").replace("-", "_").replace(".", "_")
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def build_index(chunks: List[Chunk] = None, embed_model: str = EMBED_MODEL, rebuild: bool = False):
    """Embed all chunks into a persistent Chroma collection (no-op if already built)."""
    col = _collection(embed_model)
    if col.count() and not rebuild:
        return col
    if col.count():
        col.delete(ids=col.get()["ids"])
    chunks = chunks or ingest_directory()
    texts = [c.text for c in chunks]
    embeddings = _embedder(embed_model).encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    col.add(
        ids=[f"{c.doc_id}::{c.page}::{c.chunk_index}" for c in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"doc_id": c.doc_id, "source": c.source, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ],
    )
    return col


def retrieve(query: str, k: int = 5, rerank: bool = False, candidates: int = 20,
             embed_model: str = EMBED_MODEL) -> List[dict]:
    """Return top-k hits: {text, doc_id, page, score}. Optionally cross-encoder rerank."""
    col = build_index(embed_model=embed_model)
    n = max(k, candidates) if rerank else k
    q = _embedder(embed_model).encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=n)
    hits = [
        {"text": t, "doc_id": m["doc_id"], "page": m["page"], "score": 1 - d}
        for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    if rerank:
        scores = _reranker().predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


if __name__ == "__main__":
    build_index(rebuild=True)
    for h in retrieve("How does model routing reduce LLM cost?"):
        print(f"{h['score']:.3f}  {short_id(h['doc_id'])} p.{h['page']}  {h['text'][:80]!r}")
