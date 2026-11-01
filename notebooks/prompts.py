"""Prompt templates: technical summary (stage 1) and business translation variants (stage 2)."""

SUMMARY_SYSTEM = (
    "You summarize AI research papers using ONLY the provided excerpts. "
    "Cite every claim as [ID p.N] using the excerpt labels. "
    "If the excerpts do not answer the question, reply exactly: Not found in the provided sources."
)

SUMMARY_TEMPLATE = """Question: {query}

Excerpts:
{context}

Write a concise, factual summary (max 150 words) answering the question, with citations like [2608.20316v1 p.3]."""

# v1: plain
BUSINESS_V1 = """Rewrite the technical summary below for a non-technical business reader. Keep citations.

Technical summary:
{summary}"""

# v2: structured, advisor persona
BUSINESS_V2 = """You are a KPMG technology advisor briefing a client executive.
Using ONLY the technical summary below, write these sections (1-2 sentences each, plain language, no jargon):
- What it is
- Why it matters
- Business implications
- Risks & limits
- Suggested next step
Keep the citations from the summary. Do not add facts that are not in the summary.
If the summary says "Not found in the provided sources.", repeat that and stop.

Technical summary:
{summary}"""

# v3: structured + one few-shot example
BUSINESS_V3 = """You are a KPMG technology advisor briefing a client executive.
Using ONLY the technical summary, write the five sections shown in the example.

Example summary: A router picks the cheapest LLM that can answer each query, cutting cost 40% with <2% accuracy loss [X p.1].
Example output:
**What it is:** A system that sends each question to the cheapest AI model able to answer it [X p.1].
**Why it matters:** AI usage costs grow quickly; routing lowers them without a visible quality drop [X p.1].
**Business implications:** Teams can scale AI tools while holding spend down.
**Risks & limits:** Savings were measured on benchmarks, not on our own workloads.
**Suggested next step:** Pilot routing on one internal use case and track cost and quality.

Now the real input. Keep citations; add no facts beyond the summary. If the summary says "Not found in the provided sources.", repeat that and stop.

Technical summary:
{summary}"""

BUSINESS_VARIANTS = {"v1": BUSINESS_V1, "v2": BUSINESS_V2, "v3": BUSINESS_V3}
