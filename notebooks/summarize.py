"""Stage 1 (grounded technical summary) and stage 2 (business translation)."""
from typing import List

from llm import generate
from prompts import BUSINESS_VARIANTS, SUMMARY_SYSTEM, SUMMARY_TEMPLATE
from retrieval import short_id


def format_context(hits: List[dict]) -> str:
    return "\n\n".join(f"[{short_id(h['doc_id'])} p.{h['page']}]\n{h['text']}" for h in hits)


def summarize(query: str, hits: List[dict]) -> str:
    prompt = SUMMARY_TEMPLATE.format(query=query, context=format_context(hits))
    return generate(prompt, system=SUMMARY_SYSTEM)


def business_translate(summary: str, variant: str = "v2") -> str:
    return generate(BUSINESS_VARIANTS[variant].format(summary=summary))
