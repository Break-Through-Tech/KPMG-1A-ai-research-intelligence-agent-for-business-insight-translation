"""Terminal test bench for the generation step (summary + business translation).

Retrieval is fixed (MiniLM + rerank); this script is for checking what the LLM does with it.
Ollama must be running.

Examples (run from the repo root):
    python notebooks/generation_test.py "What is Task Model Induction (TMI)?"
    python notebooks/generation_test.py "..." --show-context          # also print the retrieved chunks
    python notebooks/generation_test.py "..." --variant all           # compare business prompts v1, v2, v3
    python notebooks/generation_test.py "..." --model qwen2.5:7b -k 6 # try another model / more chunks
    python notebooks/generation_test.py --file data/eval/benchmark_personas.json --limit 3
    python notebooks/generation_test.py                               # interactive; type 'quit' to exit
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import llm  # noqa: E402
from evaluate import SECTIONS, citation_validity  # noqa: E402
from retrieval import retrieve, short_id  # noqa: E402
from summarize import business_translate, summarize  # noqa: E402

NOT_FOUND = "not found in the provided sources"


def run_query(query: str, args) -> None:
    print("=" * 80 + f"\nQUERY: {query}\n" + "=" * 80)
    hits = retrieve(query, k=args.k)
    print("\nSOURCES")
    for h in hits:
        print(f"  {short_id(h['doc_id'])} p.{h['page']}  (rerank score {h['score']:.2f})")
    if args.show_context:
        print("\nRETRIEVED CONTEXT")
        for h in hits:
            print(f"--- [{short_id(h['doc_id'])} p.{h['page']}] ---\n{h['text']}\n")

    t0 = time.time()
    summary = summarize(query, hits)
    t_sum = time.time() - t0
    print(f"\nTECHNICAL SUMMARY  ({t_sum:.1f}s)\n{summary}")
    cv = citation_validity(summary, hits)
    refused = NOT_FOUND in summary.lower()
    print(f"\n  checks: citations valid = {'n/a' if cv != cv else f'{cv:.0%}'}"
          f" | refused (not found) = {refused}")

    if args.no_business:
        print()
        return
    variants = ["v1", "v2", "v3"] if args.variant == "all" else [args.variant]
    for v in variants:
        t0 = time.time()
        biz = business_translate(summary, v)
        t_biz = time.time() - t0
        sections = sum(s.lower() in biz.lower() for s in SECTIONS)
        bcv = citation_validity(biz, hits)
        print(f"\nBUSINESS INSIGHT [{v}]  ({t_biz:.1f}s)\n{biz}")
        print(f"\n  checks: sections = {sections}/{len(SECTIONS)}"
              f" | citations valid = {'n/a' if bcv != bcv else f'{bcv:.0%}'}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="*", help="question (omit for interactive mode)")
    p.add_argument("--variant", default="v2", choices=["v1", "v2", "v3", "all"])
    p.add_argument("--model", help=f"Ollama model (default {llm.LLM_MODEL})")
    p.add_argument("-k", type=int, default=4, help="chunks passed to the LLM (default 4)")
    p.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    p.add_argument("--no-business", action="store_true", help="skip the business translation")
    p.add_argument("--file", help="JSON file with a list of {'query': ...} items to run in batch")
    p.add_argument("--limit", type=int, help="with --file: only run the first N queries")
    args = p.parse_args()
    if args.model:
        llm.LLM_MODEL = args.model

    def safe(q: str) -> None:
        try:
            run_query(q, args)
        except Exception as e:  # usually Ollama not running or the model not pulled
            print(f"\nError: {e}\nIs Ollama running, and is the model pulled (ollama pull {llm.LLM_MODEL})?\n")

    if args.file:
        items = json.loads(Path(args.file).read_text(encoding="utf-8"))
        for item in items[: args.limit]:
            safe(item["query"])
    elif args.query:
        safe(" ".join(args.query))
    else:
        print(f"Generation test bench (model: {llm.LLM_MODEL}). Type a question, or 'quit'.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                safe(q)


if __name__ == "__main__":
    main()
