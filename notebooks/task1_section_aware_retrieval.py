"""
KPMG AI Research Intelligence Agent - October Milestone #2
Task #1: Section-Aware Retrieval Finalization & Zero-Contamination Verification
Assignees: Chris, Sidney
"""

import os
import re
import sys
from pathlib import Path
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


SECTION_NAMES = [
    "abstract", "introduction", "related work", "background",
    "method", "methodology", "approach", "experiment",
    "experiments", "results", "evaluation", "discussion",
    "conclusion", "conclusions", "limitations", "references", "bibliography"
]


def get_data_dir() -> Path:
    """Finds the data folder across root or subfolder executions."""
    for candidate in [Path("data"), Path("../data"), Path("KPMG data")]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    print("Error: Could not locate 'data/' folder. Ensure your PDFs are in ./data.")
    sys.exit(1)


def extract_clean_sections(pdf_path: Path) -> list[Document]:
    """Reads PDF, parses along section boundaries, and drops references."""
    reader = PdfReader(str(pdf_path))
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

    # Extract arXiv ID
    arxiv_match = re.search(r"\b\d{4}\.\d{4,5}(v\d+)?\b", full_text[:3000])
    arxiv_id = arxiv_match.group(0) if arxiv_match else pdf_path.stem.split("v")[0]

    # Extract Title fallback
    meta = reader.metadata or {}
    title = (meta.title or "").strip()
    if not title:
        for line in full_text.splitlines()[:15]:
            clean_line = line.strip()
            if len(clean_line) > 15 and "arxiv" not in clean_line.lower() and not clean_line.isdigit():
                title = clean_line
                break
    title = title or pdf_path.stem

    # Match section headers
    pattern = re.compile(r"(?im)^\s*\d{0,2}\.?\s*(" + "|".join(SECTION_NAMES) + r")\s*$")
    matches = list(pattern.finditer(full_text))

    docs = []
    if not matches:
        docs.append(Document(
            page_content=full_text,
            metadata={"source": pdf_path.name, "arxiv_id": arxiv_id, "title": title, "section": "body"}
        ))
        return docs

    for i, match in enumerate(matches):
        section_name = match.group(1).lower()

        # Core Task #1 Filter: Discard reference and bibliography sections
        if "reference" in section_name or "bibliography" in section_name:
            continue

        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_content = full_text[start_idx:end_idx].strip()

        if len(section_content) > 80:
            docs.append(Document(
                page_content=section_content,
                metadata={
                    "source": pdf_path.name,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "section": section_name
                }
            ))

    return docs


def build_clean_vectorstore(data_dir: Path) -> Chroma:
    """Builds and indexes the section-aware ChromaDB collection."""
    pdf_files = sorted(list(data_dir.glob("*.pdf")))
    print(f"[Task 1] Parsing {len(pdf_files)} PDF(s) from: {data_dir.resolve()}")

    section_docs = []
    for pdf in pdf_files:
        section_docs.extend(extract_clean_sections(pdf))
    print(f"[Task 1] Extracted {len(section_docs)} clean section documents (excluding references).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunked_docs = splitter.split_documents(section_docs)
    print(f"[Task 1] Created {len(chunked_docs)} chunks (800 chars / 100 overlap).")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        collection_name="kpmg_clean_research"
    )
    print("[Task 1] ChromaDB vector store successfully created.")
    return vectorstore


def verify_zero_contamination(vectorstore: Chroma):
    """Audits retrieval on benchmark queries to verify zero citation contamination."""
    test_query = "What are some AI benchmarks?"
    results = vectorstore.similarity_search_with_score(test_query, k=4)
    print(f"\n--- AUDIT: Querying '{test_query}' ---")

    has_contamination = False
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        section = meta.get("section", "")
        content = doc.page_content.strip()

        # Check if references or excessive brackets leaked into content
        is_bad = "reference" in section or len(re.findall(r"\[\d+\]", content)) > 3
        status = "PASSED (Clean)" if not is_bad else "FAILED (Contaminated)"
        if is_bad:
            has_contamination = True

        print(f"Result {i} [{status} | Distance: {score:.4f}]")
        print(f"  Title:   {meta.get('title')}")
        print(f"  Section: {section}")
        print(f"  Snippet: {content[:140].replace(chr(10), ' ')}...\n")

    if not has_contamination:
        print("VERIFICATION RESULT: PASSED (Zero bibliography contamination confirmed).")
    else:
        print("VERIFICATION RESULT: FAILED (Citation noise detected).")


if __name__ == "__main__":
    data_directory = get_data_dir()
    vstore = build_clean_vectorstore(data_directory)
    verify_zero_contamination(vstore)