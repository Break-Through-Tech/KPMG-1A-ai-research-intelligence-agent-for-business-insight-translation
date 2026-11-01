# Evaluation Findings (October, Weeks 6–8)

Pipeline under test: PDF chunks (1000 chars, 200 overlap) -> ChromaDB retrieval (optional cross-encoder rerank) -> grounded technical summary -> business translation, using a local `qwen2.5:3b` model via Ollama.
Corpus: 5 arXiv papers (485 chunks). Benchmark: 15 on-topic + 3 off-topic queries (`data/eval/benchmark.json`).
Reproduce: `python notebooks/evaluate.py retrieval` and `python notebooks/evaluate.py generation` (LLM outputs cached in `data/eval/cache.json`).

## 1. Evaluation techniques used

### 1.1 Retrieval relevance (automatic, no LLM)
Each on-topic query has a gold paper (`gold`) and one evidence keyword (`evidence`) that must appear in a relevant chunk. Keyword matching ignores whitespace and case because PDF extraction sometimes drops spaces.

| Metric | Definition | Why |
|---|---|---|
| **DocRecall@k** | Fraction of queries where at least one of the top-k chunks comes from the gold paper | Coarse check that the right paper is found |
| **Evidence@k** | Fraction of queries where at least one top-k chunk is from the gold paper **and** contains the evidence keyword | Stricter, chunk-level check. DocRecall saturates at 1.0 on a 5-paper corpus, so this is what separates configs |
| **MRR (evidence)** | Mean of 1 / rank of the first chunk that satisfies the Evidence test | Rewards ranking the useful chunk higher |

Four configurations were compared: embedding model (`all-MiniLM-L6-v2` vs `BAAI/bge-small-en-v1.5`) x reranking (none vs cross-encoder `ms-marco-MiniLM-L-6-v2` over the top 20 candidates).

### 1.2 Summary accuracy / grounding (automatic)
- **Citation validity**: share of `[paper-id p.N]` citations in the output that point to a (paper, page) actually retrieved for that query. Catches invented citations. Reported for the technical summary and for each business variant.
- **LLM-as-judge faithfulness (1–5)**: the local model rates how well the summary is supported by the retrieved excerpts. *Noisy*: a 3B judge is weak, so treat as a screening signal only.
- **Off-topic refusal rate**: share of the 3 off-topic queries where the summary is exactly "Not found in the provided sources." Tests that the system does not hallucinate when the corpus has no answer.

### 1.3 Business-translation quality (automatic structure check + human rubric)
- **Section completeness**: share of the 5 required sections (What it is / Why it matters / Business implications / Risks & limits / Suggested next step) present in the output. Compares three prompt variants: v1 plain rewrite, v2 structured advisor persona, v3 structured plus one few-shot example.
- **Human rubric (1–5), pending**: `results.csv` has blank columns `human_accuracy`, `human_clarity`, `human_business_usefulness` for the team to fill. These are the authoritative measures of summary accuracy, clarity and business usefulness; the automatic checks above do not measure whether insights are actually useful.

## 2. Results

### 2.1 Retrieval (15 on-topic queries)

| Config | DocRecall@5 | Evidence@1 | Evidence@3 | Evidence@5 | MRR |
|---|---|---|---|---|---|
| MiniLM | 1.00 | 0.67 | 0.87 | 0.93 | 0.76 |
| MiniLM + rerank | 1.00 | 0.87 | 1.00 | 1.00 | 0.93 |
| bge-small | 1.00 | 0.73 | 0.93 | 0.93 | 0.83 |
| bge-small + rerank | 1.00 | 0.87 | 1.00 | 1.00 | 0.93 |

Reranking is the main gain (+0.20 Evidence@1, +0.17 MRR for MiniLM). The embedding model matters less. The pipeline uses MiniLM + rerank.

### 2.2 Generation (MiniLM + rerank, top 4 chunks)

| Check | Result |
|---|---|
| Top-ranked chunk from correct paper | 15 / 15 |
| Technical-summary citation validity | 0.93 (14 of 15 queries fully valid; query 2 cited a page that was not retrieved) |
| Judge faithfulness (mean, 1–5) | 2.93; only 33% of summaries scored 4 or higher |
| Off-topic refusal | 3 / 3 |

| Variant | Sections complete | Business citation validity |
|---|---|---|
| v1 plain | 0.00 (no fixed sections by design) | 1.00 |
| v2 structured advisor | **1.00** | n/a: v2 outputs contained **no citations**, despite the instruction |
| v3 few-shot | 0.79 | 0.93 |

## 3. Findings and failure analysis
1. **Retrieval works; precision at rank 1 is the weak spot.** Without reranking, the best chunk is often ranked 2nd–5th. Rerank fixes this.
2. **v2 gives the most consistent structure but drops citations**, which conflicts with the project requirement of cited outputs. v1 keeps citations but is unstructured. v3 loses sections in about 1 of 5 outputs. No variant satisfies both requirements yet.
3. **Faithfulness is the main quality risk.** Mean judge score is 2.9/5. Low scores (queries 3, 4, 7, 9, 10, 11, 13, 15) need human review to tell apart judge noise from real unsupported claims.
4. **Likely contributing cause: text extraction.** pdfplumber drops spaces in parts of these PDFs (e.g. "AdamFisch?,ShubhenduTrivedi"), which degrades chunk text for embedding and for the LLM.
5. **Off-topic handling is good** (3/3), at least for these clearly unrelated queries.

## 4. Limitations
- Only 5 papers and 18 queries; results are indicative, not statistically robust.
- Evidence keywords are a proxy for relevance, written by one team member, not independent human relevance labels.
- The judge model is the same small model that writes the summaries; scores are noisy and may be biased.
- Human accuracy / clarity / business-usefulness ratings are not yet collected.
- Off-topic test uses only blatantly unrelated queries; near-domain unanswerable queries are untested.

## 5. Improvement opportunities (for Week 9)
1. Fix extraction spacing (pdfplumber `x_tolerance`, or `pypdf`) and re-run retrieval and generation to measure the effect.
2. Make business outputs cite sources: pass the cited summary's sources explicitly, add a citation requirement to the v2 template, and add a post-check that retries when no citation is present.
3. Collect human ratings on a sample (prioritise low-faithfulness queries) and calibrate the judge against them.
4. Try a larger local model (e.g. a 7B) for summarisation/judging and compare.
5. Expand the corpus beyond 5 papers and add near-domain unanswerable queries.
6. Try chunking by section/paragraph instead of fixed 1000-character windows.
