import sys

from data_ingestion import DATA_DIR, extract_text, ingest_directory


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


pdfs = sorted(DATA_DIR.glob("*.pdf"))
safe_print(f"Found {len(pdfs)} PDF(s) in {DATA_DIR}")

sample = pdfs[0]
pages = extract_text(sample)
safe_print(f"\n{sample.name}")
safe_print(f"  pages: {len(pages)}")
safe_print(f"  page 1 preview: {pages[1][:300]!r}")

chunks = ingest_directory()
safe_print(f"\nIngested {len(pdfs)} PDF(s) into {len(chunks)} total chunks")
safe_print(f"  first chunk: {chunks[0]}")
