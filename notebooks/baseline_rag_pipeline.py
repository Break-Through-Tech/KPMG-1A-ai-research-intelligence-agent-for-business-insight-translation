"""Task #5: Baseline RAG Retrieval Pipeline for KPMG Challenge Project

Team: KPMG 1A: Build and
test baseline vector retrieval with ChromaDB and
embeddings.
"""

from pathlib import Path
import re
import sys
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# 1. Section Header Constants (from teammate extractor script)
SECTION_NAMES = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methodology",
    "approach",
    "experiment",
    "experiments",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
]


def find_data_directory() -> Path:
  """Locates the data directory whether run from root or a subfolder."""
  candidate_paths = [Path("data"), Path("../data"), Path("KPMG data")]
  for path in candidate_paths:
    if path.exists() and path.is_dir():
      return path
  print(
      "ERROR: Could not find 'data' folder. Checked: ./data, ../data, ./KPMG"
      " data"
  )
  sys.exit(1)


def parse_pdf_sections(pdf_path: Path):
  """Extracts clean section text from PDF, tagging metadata and skipping references."""
  reader = PdfReader(str(pdf_path))
  full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

  # Extract ArXiv ID from text or file name
  arxiv_match = re.search(r"\b\d{4}\.\d{4,5}(v\d+)?\b", full_text[:3000])
  arxiv_id = (
      arxiv_match.group(0)
      if arxiv_match
      else pdf_path.stem.split("v")[0].split("_")[0]
  )

  # Title extraction fallback
  meta = reader.metadata or {}
  title = (meta.title or "").strip()
  if not title:
    for line in full_text.splitlines()[:15]:
      clean_line = line.strip()
      if (
          len(clean_line) > 15
          and "arxiv" not in clean_line.lower()
          and not clean_line.isdigit()
      ):
        title = clean_line
        break
  title = title or pdf_path.stem

  # Section splitting regex
  pattern = re.compile(
      r"(?im)^\s*\d{0,2}\.?\s*(" + "|".join(SECTION_NAMES) + r")\s*$"
  )
  matches = list(pattern.finditer(full_text))

  docs = []
  if not matches:
    # If no section headings found, keep entire document
    docs.append(
        Document(
            page_content=full_text,
            metadata={
                "source": pdf_path.name,
                "arxiv_id": arxiv_id,
                "title": title,
                "section": "full_body",
            },
        )
    )
    return docs

  for i, match in enumerate(matches):
    sec_name = match.group(1).lower()

    # Skip reference lists so citation clutter does not pollute the vector store
    if "reference" in sec_name:
      continue

    start_idx = match.end()
    end_idx = (
        matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
    )
    sec_content = full_text[start_idx:end_idx].strip()

    if len(sec_content) > 80:
      docs.append(
          Document(
              page_content=sec_content,
              metadata={
                  "source": pdf_path.name,
                  "arxiv_id": arxiv_id,
                  "title": title,
                  "section": sec_name,
              },
          )
      )

  return docs


def main():
  print("=" * 60)
  print("Starting Task #5: Baseline RAG Retrieval Pipeline Setup")
  print("=" * 60)

  # Step A: Identify PDF Data
  data_dir = find_data_directory()
  pdf_files = sorted(list(data_dir.glob("*.pdf")))
  print(f"[1/4] Found {len(pdf_files)} PDF file(s) in: {data_dir.resolve()}")
  if not pdf_files:
    print("No PDFs found to ingest. Add PDFs to the data directory and rerun.")
    return

  # Step B: Parse and Chunk Documents
  raw_section_docs = []
  for pdf in pdf_files:
    print(f"      Parsing: {pdf.name}")
    raw_section_docs.extend(parse_pdf_sections(pdf))

  print(
      f"\n[2/4] Splitting {len(raw_section_docs)} section documents into"
      " chunks..."
  )
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=100,
      separators=["\n\n", "\n", " ", ""],
  )
  chunked_docs = splitter.split_documents(raw_section_docs)
  print(f"      Created {len(chunked_docs)} clean, attributed text chunks.")

  # Step C: Generate Embeddings and Stand Up ChromaDB
  print("\n[3/4] Initializing HuggingFace Embeddings & ChromaDB Vector Store...")
  print("      Model: sentence-transformers/all-mpnet-base-v2")
  embedding_model = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-mpnet-base-v2"
  )

  # Create Chroma collection in-memory / local workspace
  vectorstore = Chroma.from_documents(
      documents=chunked_docs,
      embedding=embedding_model,
      collection_name="kpmg_task5_baseline",
  )
  print("      ChromaDB vector collection successfully indexed.")

  # Step D: Test Basic Retrieval Against Benchmark Queries
  print("\n[4/4] Executing Baseline Retrieval Tests...")
  test_queries = [
      (
          "How do we balance cost, latency, and accuracy when routing queries"
          " across models?"
      ),
      (
          "What are the failure modes and security risks when agents execute"
          " computer actions?"
      ),
      "How do we verify medical factuality and prevent clinical hallucinations?",
  ]

  for q_idx, query in enumerate(test_queries, 1):
    print("\n" + "-" * 55)
    print(f"TEST QUERY #{q_idx}: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=2)

    for r_idx, (doc, score) in enumerate(results, 1):
      meta = doc.metadata
      print(f"  Result {r_idx} [Distance: {score:.4f}]")
      print(f"    Title:    {meta.get('title')}")
      print(f"    ArXiv ID: {meta.get('arxiv_id')}")
      print(f"    Section:  {meta.get('section')}")
      print(f"    Excerpt:  {doc.page_content[:180].replace(chr(10), ' ')}...")

  print("\n" + "=" * 60)
  print("Task #5 Pipeline Verification COMPLETE: Retrieval operational.")
  print("=" * 60)


if __name__ == "__main__":
  main()