"""Ask the research assistant a question from the terminal.

Usage (from the repo root or the notebooks folder; Ollama must be running):
    python notebooks/ask.py "What is Task Model Induction (TMI)?"   # one question
    python notebooks/ask.py                                         # interactive, type 'quit' to exit
Options:
    --variant v1|v2|v3   business prompt variant (default v2)
    --no-business        skip the business translation
"""
import argparse
import os
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import answer  # noqa: E402
from retrieval import short_id  # noqa: E402


def show(result: dict) -> None:
    print("\nSOURCES")
    for doc_id, page, score in result["sources"]:
        print(f"  {short_id(doc_id)} p.{page}  (score {score})")
    print("\nTECHNICAL SUMMARY\n" + result["summary"])
    if "business" in result:
        print("\nBUSINESS INSIGHT\n" + result["business"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("--variant", default="v2", choices=["v1", "v2", "v3"])
    parser.add_argument("--no-business", action="store_true")
    args = parser.parse_args()

    def run(q: str) -> None:
        try:
            show(answer(q, variant=args.variant, business=not args.no_business))
        except Exception as e:  # most likely Ollama not running or model not pulled
            print(f"\nError: {e}\nIs Ollama running, and is the model pulled (ollama pull qwen2.5:3b)?")

    if args.query:
        run(" ".join(args.query))
        return
    print("Ask a question about the indexed papers (type 'quit' to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            run(q)


if __name__ == "__main__":
    main()
