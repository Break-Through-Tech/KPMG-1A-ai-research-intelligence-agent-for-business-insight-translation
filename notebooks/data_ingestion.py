"""PDF data ingestion and extraction pipeline for the RAG project.

Extracts text page-by-page from arXiv PDFs in `data/`, splits it into
overlapping chunks, and writes the results to `data/processed/chunks.json`
for downstream embedding/retrieval steps.
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pdfplumber

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
X_TOLERANCE = 1.5


@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    chunk_index: int
    text: str


def extract_text(pdf_path: Path, x_tolerance: float = X_TOLERANCE) -> dict[int, str]:
    """Extract text from a PDF, keyed by 1-indexed page number.

    The pdfplumber default x_tolerance (3) merges words in several of these PDFs
    ("AdamFisch?,ShubhenduTrivedi"); 1.5 restores normal word spacing.
    """
    data = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            data[i] = page.extract_text(x_tolerance=x_tolerance) or ""
    return data


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
               strategy: str = "fixed") -> List[str]:
    """Split text into overlapping chunks. strategy: 'fixed' (characters) or 'sentence'."""
    text = text.strip()
    if not text:
        return []
    if strategy == "sentence":
        return _chunk_sentences(text, chunk_size, overlap)

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _chunk_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Pack whole sentences up to ~chunk_size chars; carry trailing sentences (<= overlap chars) forward."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    chunks, current = [], []
    for s in sentences:
        if current and sum(len(x) + 1 for x in current) + len(s) > chunk_size:
            chunks.append(" ".join(current))
            carry, size = [], 0
            for x in reversed(current):
                if size + len(x) > overlap:
                    break
                carry.insert(0, x)
                size += len(x) + 1
            current = carry
        current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks


def process_pdf(pdf_path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                strategy: str = "fixed") -> List[Chunk]:
    """Extract and chunk a single PDF into `Chunk` records."""
    doc_id = pdf_path.stem
    pages = extract_text(pdf_path)

    chunks = []
    for page_num, page_text in pages.items():
        for i, piece in enumerate(chunk_text(page_text, chunk_size, overlap, strategy)):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    source=pdf_path.name,
                    page=page_num,
                    chunk_index=i,
                    text=piece,
                )
            )
    return chunks


def ingest_directory(data_dir: Path = DATA_DIR, output_dir: Path = PROCESSED_DIR) -> List[Chunk]:
    """Run extraction + chunking over every PDF in `data_dir` and persist the result."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    all_chunks: List[Chunk] = []
    for pdf_path in pdf_paths:
        all_chunks.extend(process_pdf(pdf_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_directory()
    docs = {c.doc_id for c in chunks}
    print(f"Ingested {len(docs)} PDF(s) into {len(chunks)} chunks -> {PROCESSED_DIR / 'chunks.json'}")
