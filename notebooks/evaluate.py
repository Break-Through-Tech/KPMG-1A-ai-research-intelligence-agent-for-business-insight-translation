"""Evaluation: retrieval metrics (Recall@k, MRR), generation checks, LLM-judge, human-rubric export.

Usage:
    python evaluate.py retrieval     # fast, no LLM
    python evaluate.py generation    # slow: runs the local LLM; cached in data/eval/cache.json
"""
import json
import re
import sys

import pandas as pd

from data_ingestion import DATA_DIR
from llm import generate
from prompts import BUSINESS_VARIANTS
from retrieval import retrieve, short_id
from summarize import business_translate, summarize

EVAL_DIR = DATA_DIR / "eval"
BENCHMARK = EVAL_DIR / "benchmark.json"
CACHE = EVAL_DIR / "cache.json"
SECTIONS = ("What it is", "Why it matters", "Business implications", "Risks", "Suggested next step")
NOT_FOUND = "not found in the provided sources"
CITE_RE = re.compile(r"\[(\d{4}\.\d{5}v\d)\s*p\.?\s*(\d+)\]")

RETRIEVAL_CONFIGS = {
    "minilm": dict(embed_model="all-MiniLM-L6-v2", rerank=False),
    "minilm+rerank": dict(embed_model="all-MiniLM-L6-v2", rerank=True),
    "bge-small": dict(embed_model="BAAI/bge-small-en-v1.5", rerank=False),
    "bge-small+rerank": dict(embed_model="BAAI/bge-small-en-v1.5", rerank=True),
}


def load_benchmark():
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))


def _load_cache():
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def _cached(cache, key, fn):
    if key not in cache:
        cache[key] = fn()
        CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[key]


# ---------- retrieval ----------
def evaluate_retrieval(ks=(1, 3, 5)) -> pd.DataFrame:
    """Doc-level Recall@k (any gold doc in top-k chunks) and MRR (rank of first gold chunk)."""
    items = [b for b in load_benchmark() if not b.get("off_topic")]
    rows = []
    for name, cfg in RETRIEVAL_CONFIGS.items():
        recall = {k: 0 for k in ks}
        evid = {k: 0 for k in ks}
        rr = 0.0
        for b in items:
            hits = retrieve(b["query"], k=max(ks), **cfg)
            ids = [short_id(h["doc_id"]) for h in hits]
            # chunk-level: right paper AND the chunk contains the evidence keyword (whitespace-insensitive)
            good = [i in b["gold"] and b["evidence"] in "".join(h["text"].lower().split())
                    for i, h in zip(ids, hits)]
            for k in ks:
                recall[k] += any(i in b["gold"] for i in ids[:k])
                evid[k] += any(good[:k])
            rr += next((1 / (r + 1) for r, g in enumerate(good) if g), 0.0)
        rows.append({"config": name, **{f"DocRecall@{k}": recall[k] / len(items) for k in ks},
                     **{f"Evidence@{k}": evid[k] / len(items) for k in ks}, "MRR(evidence)": rr / len(items)})
    return pd.DataFrame(rows)


# ---------- generation ----------
def citation_validity(text: str, hits: list) -> float:
    """Fraction of citations in `text` that point to a retrieved (doc, page)."""
    cites = CITE_RE.findall(text)
    if not cites:
        return float("nan")
    valid = {(short_id(h["doc_id"]), str(h["page"])) for h in hits}
    return sum(c in valid for c in cites) / len(cites)


def judge_faithfulness(summary: str, hits: list) -> int:
    """LLM-as-judge (local model, noisy): 1-5 how well the summary is supported by the excerpts."""
    context = "\n\n".join(h["text"] for h in hits)
    prompt = (
        "Rate from 1 to 5 how well every claim in the SUMMARY is supported by the EXCERPTS "
        "(5 = fully supported, 1 = mostly unsupported). Answer with a single digit only.\n\n"
        f"EXCERPTS:\n{context}\n\nSUMMARY:\n{summary}"
    )
    m = re.search(r"[1-5]", generate(prompt, temperature=0.0))
    return int(m.group()) if m else 0


def evaluate_generation(k: int = 4, rerank: bool = True, variants=("v1", "v2", "v3")) -> pd.DataFrame:
    cache = _load_cache()
    rows = []
    for b in load_benchmark():
        hits = retrieve(b["query"], k=k, rerank=rerank)
        summary = _cached(cache, f"sum|{b['id']}|{k}|{rerank}", lambda: summarize(b["query"], hits))
        faith = _cached(cache, f"faith|{b['id']}|{k}|{rerank}", lambda: judge_faithfulness(summary, hits))
        for v in variants:
            biz = _cached(cache, f"biz|{b['id']}|{k}|{rerank}|{v}", lambda: business_translate(summary, v))
            rows.append({
                "id": b["id"], "query": b["query"], "off_topic": b.get("off_topic", False),
                "variant": v, "summary": summary, "business": biz,
                "top_doc_correct": (short_id(hits[0]["doc_id"]) in b["gold"]) if b["gold"] else None,
                "summary_cite_valid": citation_validity(summary, hits),
                "biz_cite_valid": citation_validity(biz, hits),
                "refused_off_topic": NOT_FOUND in summary.lower(),
                "sections_complete": sum(s.lower() in biz.lower() for s in SECTIONS) / len(SECTIONS),
                "judge_faithfulness": faith,
                # filled in by the team (1-5):
                "human_accuracy": "", "human_clarity": "", "human_business_usefulness": "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "results.csv", index=False, encoding="utf-8")
    return df


def summarize_generation(df: pd.DataFrame) -> pd.DataFrame:
    on = df[~df.off_topic]
    by_variant = on.groupby("variant").agg(
        sections_complete=("sections_complete", "mean"),
        biz_cite_valid=("biz_cite_valid", "mean"),
    )
    by_variant["summary_cite_valid"] = on.drop_duplicates("id").summary_cite_valid.mean()
    by_variant["judge_faithfulness"] = on.drop_duplicates("id").judge_faithfulness.mean()
    by_variant["off_topic_refusal_rate"] = df[df.off_topic].drop_duplicates("id").refused_off_topic.mean()
    return by_variant


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    pd.set_option("display.width", 200)
    if mode == "retrieval":
        res = evaluate_retrieval()
        print(res.round(3).to_string(index=False))
        res.to_csv(EVAL_DIR / "retrieval_results.csv", index=False)
    else:
        df = evaluate_generation()
        print(summarize_generation(df).round(3))
