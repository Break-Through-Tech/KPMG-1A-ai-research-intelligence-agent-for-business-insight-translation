"""Compare PDF extraction settings and chunking strategies with the retrieval benchmark.

All variants use the final retrieval config (MiniLM embeddings + cross-encoder rerank).
Indexes are built in memory, so the persistent index in data/chroma is untouched.

Usage: python chunking_experiment.py
"""
from typing import Dict, List

import chromadb
import pandas as pd

from data_ingestion import DATA_DIR, Chunk, extract_text
from evaluate import EVAL_DIR, load_benchmark
from retrieval import _embedder, _reranker, short_id

K = 5
CANDIDATES = 20

# name -> (x_tolerance, strategy, chunk_size, overlap)
VARIANTS = {
    "A baseline: tol=3, fixed 1000/200": (3, "fixed", 1000, 200),
    "B extraction fix: tol=1.5, fixed 1000/200": (1.5, "fixed", 1000, 200),
    "C smaller: tol=1.5, fixed 500/100": (1.5, "fixed", 500, 100),
    "D sentence-aware: tol=1.5, ~1000/200": (1.5, "sentence", 1000, 200),
}


def build_chunks(tol: float, strategy: str, size: int, overlap: int) -> List[Chunk]:
    from data_ingestion import chunk_text

    chunks = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        for page, text in extract_text(pdf, x_tolerance=tol).items():
            for i, piece in enumerate(chunk_text(text, size, overlap, strategy)):
                chunks.append(Chunk(pdf.stem, pdf.name, page, i, piece))
    return chunks


def evaluate_variant(chunks: List[Chunk], items: List[dict]) -> Dict[str, float]:
    client = chromadb.EphemeralClient()
    name = f"tmp_{abs(hash(tuple(c.text[:20] for c in chunks[:5]))) % 10**8}_{len(chunks)}"
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    emb = _embedder().encode([c.text for c in chunks], batch_size=32, normalize_embeddings=True)
    col.add(
        ids=[f"{i}" for i in range(len(chunks))],
        documents=[c.text for c in chunks],
        embeddings=emb.tolist(),
        metadatas=[{"doc_id": c.doc_id, "page": c.page} for c in chunks],
    )
    hit = {1: 0, 3: 0, K: 0}
    rr = 0.0
    ctx_chars = 0
    for b in items:
        q = _embedder().encode([b["query"]], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=q, n_results=CANDIDATES)
        docs, metas = res["documents"][0], res["metadatas"][0]
        scores = _reranker().predict([(b["query"], d) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:K]
        good = [
            short_id(metas[i]["doc_id"]) in b["gold"] and b["evidence"] in "".join(docs[i].lower().split())
            for i in order
        ]
        for k in hit:
            hit[k] += any(good[:k])
        rr += next((1 / (r + 1) for r, g in enumerate(good) if g), 0.0)
        ctx_chars += sum(len(docs[i]) for i in order[:3])
    n = len(items)
    return {
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks)),
        "Evidence@1": hit[1] / n, "Evidence@3": hit[3] / n, f"Evidence@{K}": hit[K] / n,
        "MRR": rr / n,
        "avg_context_chars_top3": round(ctx_chars / n),
    }


if __name__ == "__main__":
    items = [b for b in load_benchmark() if not b.get("off_topic")]
    rows = []
    for label, (tol, strategy, size, overlap) in VARIANTS.items():
        chunks = build_chunks(tol, strategy, size, overlap)
        rows.append({"variant": label, **evaluate_variant(chunks, items)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(df.round(3).to_string(index=False))
    df.to_csv(EVAL_DIR / "chunking_results.csv", index=False, encoding="utf-8")
