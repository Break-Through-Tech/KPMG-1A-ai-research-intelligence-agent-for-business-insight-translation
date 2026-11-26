"""Stage 1 (grounded technical summary) and stage 2 (business translation)."""
import re
from typing import List

from llm import generate
from prompts import BUSINESS_VARIANTS, SUMMARY_SYSTEM, SUMMARY_TEMPLATE
from retrieval import short_id

CITE_RE = re.compile(r"\[(\d{4}\.\d{5}v\d)\s*p\.?\s*(\d+)\]")


def format_context(hits: List[dict]) -> str:
    return "\n\n".join(f"[{short_id(h['doc_id'])} p.{h['page']}]\n{h['text']}" for h in hits)


def summarize(query: str, hits: List[dict]) -> str:
    prompt = SUMMARY_TEMPLATE.format(query=query, context=format_context(hits))
    return generate(prompt, system=SUMMARY_SYSTEM)


def extract_citations(text: str) -> List[str]:
    """Unique '[id p.N]' citations in order of appearance."""
    seen = []
    for doc, page in CITE_RE.findall(text):
        c = f"[{doc} p.{page}]"
        if c not in seen:
            seen.append(c)
    return seen


def _strip_stray_rule_line(text: str) -> str:
    """The 3B model sometimes echoes a prompt rule ("Not stated in the sources.") as a trailing line."""
    parts = text.rstrip().rsplit("\n\n", 1)  # only a separate trailing paragraph, not a section's own content
    if len(parts) == 2 and parts[1].strip().lower().rstrip(".") in {
        "not stated in the sources", "not found in the provided sources"}:
        return parts[0].rstrip()
    return text


def business_translate(summary: str, variant: str = "v2") -> str:
    """Translate a cited summary into a business brief that keeps its citations.

    The prompt is given the allowed citations (taken from the summary). If the model still
    returns none of them, retry once with a stronger reminder, then fall back to appending a
    'Sources' line so every output stays traceable.
    """
    if "not found in the provided sources" in summary.lower():
        return "Not found in the provided sources."  # nothing to translate; skip the LLM
    cites = extract_citations(summary)
    prompt = BUSINESS_VARIANTS[variant].format(summary=summary, citations=", ".join(cites) or "(none)")
    out = generate(prompt)
    out = _strip_stray_rule_line(out)
    if not cites:  # e.g. "Not found in the provided sources."
        return out

    def has_cite(text: str) -> bool:
        return any(c in text for c in cites)

    if not has_cite(out):
        out = _strip_stray_rule_line(generate(
            prompt + f"\n\nIMPORTANT: your previous answer had no citations. "
                     f"Rewrite it and end every section with one of: {', '.join(cites)}."))
    if not has_cite(out):
        out = out.rstrip() + "\n\nSources (from the technical summary): " + ", ".join(cites)
    return out
