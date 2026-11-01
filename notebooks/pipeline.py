"""End-to-end: query -> retrieve (-> rerank) -> technical summary -> business insight."""
from retrieval import retrieve
from summarize import business_translate, summarize


def answer(query: str, k: int = 4, rerank: bool = True, variant: str = "v2", business: bool = True) -> dict:
    hits = retrieve(query, k=k, rerank=rerank)
    summary = summarize(query, hits)
    out = {
        "query": query,
        "sources": [(h["doc_id"], h["page"], round(h["score"], 3)) for h in hits],
        "summary": summary,
    }
    if business:
        out["business"] = business_translate(summary, variant)
    return out


if __name__ == "__main__":
    import sys

    r = answer(sys.argv[1] if len(sys.argv) > 1 else "How does model routing reduce LLM cost?")
    print("SOURCES:", *r["sources"], sep="\n  ")
    print("\nSUMMARY:\n", r["summary"], "\n\nBUSINESS:\n", r["business"], sep="")
