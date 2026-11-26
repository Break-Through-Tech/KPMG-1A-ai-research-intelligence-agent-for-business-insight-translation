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
6. Try chunking by section/paragraph instead of fixed 1000-character windows. *(Sentence-aware chunking tested in Section 6; not better.)*

## 6. November update: final configuration, extraction fix and chunking comparison

**Final retrieval configuration:** `all-MiniLM-L6-v2` embeddings + cross-encoder rerank (`DEFAULT_RERANK = True` in `retrieval.py`), fixed 1000-character chunks with 200 overlap.

### 6.1 Extraction fix
pdfplumber's default `x_tolerance=3` merged words in several PDFs ("AdamFisch?,ShubhenduTrivedi"). Setting `x_tolerance=1.5` restores normal spacing (average word length dropped from 6.5–8.6 to about 4.3–5.4 characters on the worst papers). The corpus is now 499 chunks (was 485).

### 6.2 Chunking and extraction comparison (retrieval, rerank on, 15 queries; `chunking_experiment.py`)

| Variant | Chunks | Evidence@1 | Evidence@5 | MRR |
|---|---|---|---|---|
| A: old extraction, fixed 1000/200 | 485 | 0.87 | 1.00 | 0.93 |
| B: fixed extraction, fixed 1000/200 | 499 | 0.80 | 1.00 | 0.90 |
| C: fixed extraction, fixed 500/100 | 985 | 0.67 | 0.87 | 0.74 |
| D: fixed extraction, sentence-aware ~1000/200 | 499 | 0.73 | 1.00 | 0.85 |

- 500/100 is clearly worse. The keyword metric favours larger chunks, so part of that gap is bias.
- A vs B differs by one query out of 15, and the keyword check ignores whitespace, so retrieval metrics cannot show the benefit of cleaner text. Sentence-aware chunking did not beat fixed windows.
- Decision: keep fixed 1000/200, keep the extraction fix.

### 6.3 Generation after the extraction fix (same 18 queries, single run each)

| Check | Before (old extraction) | After (fixed extraction) |
|---|---|---|
| Top chunk from correct paper | 15/15 | 15/15 |
| Judge faithfulness, mean (1–5) | 2.93 | 3.93 |
| Summaries scoring 4 or higher | 33% | 67% |
| Technical-summary citation validity | 0.93 | 0.88 |
| Off-topic refusal | 3/3 | 3/3 |
| v1 / v2 / v3 sections complete | 0 / 1.00 / 0.79 | 0 / 1.00 / 0.89 |
| Business citation validity v1 / v3 | 1.00 / 0.93 | 0.75 / 0.87 |
| Business citation validity v2 | no citations | no citations |

Interpretation and caveats:
- Judge faithfulness rose by about one point, and 10 of 15 summaries scored 4 or higher. That is consistent with cleaner text helping the LLM, but it comes from a noisy 3B judge and a single run. Some individual scores moved the wrong way (queries 2 and 4 dropped), so treat it as encouraging, not confirmed. Human ratings are still needed.
- Citation validity went slightly down, not up. Fixing text did not fix citation behaviour; that remains the next prompt task.
- v2 still produces no citations.
### 6.4 Citation fix for business outputs
`business_translate` now passes the summary's citations to the prompt, requires each section to end with one, retries once if none appear, and finally appends a "Sources" line. Off-topic summaries ("Not found...") skip the business step and return the not-found message.

| Business variant | Outputs with a citation, before | after | Sections complete, before | after |
|---|---|---|---|---|
| v1 plain | 40% | 93% | 0.00 | 0.00 |
| v2 structured advisor | 0% | 100% | 1.00 | 0.93 |
| v3 few-shot | 100% | 100% | 0.89 | 0.99 |

- Valid-citation rate (citation points to a retrieved page) is about 0.87–0.88 for all variants, inherited from the technical summaries (0.88). Citations show traceability, not that each claim is supported.
- The business step still invents generic risks (for example "does not introduce significant risks") even though the prompt says to write "Not stated in the sources". This needs a larger model or human review of that section.
- The off-topic business outputs in this run were produced before the not-found short-circuit was added (earlier outputs repeated "Not found" in every section or, for v1, asked for a summary); re-run `evaluate.py generation` to refresh them.
- Raw outputs: `data/eval/results.csv` (after) and `results_baseline_tol3.csv` (before); chunking table: `data/eval/chunking_results.csv`.
