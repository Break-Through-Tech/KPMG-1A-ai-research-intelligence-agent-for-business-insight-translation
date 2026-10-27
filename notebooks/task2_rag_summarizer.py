"""
KPMG AI Research Intelligence Agent - October Milestone #2
Task #2: LLM Summarization Integration & Guardrailed Insight Generation
Assignees: Chris, Arjun
"""

import os
from google import genai
from task1_section_aware_retrieval import get_data_dir, build_clean_vectorstore


def generate_kpmg_executive_insight(query: str, vectorstore) -> str:
    """
    Retrieves top chunks from ChromaDB and passes them to Gemini Flash
    using a strictly guardrailed prompt to translate research into business insights.
    """
    # 1. Retrieve top 3 relevant chunks
    retrieved_results = vectorstore.similarity_search_with_score(query, k=3)

    # 2. Build structured evidence context
    context_blocks = []
    for idx, (doc, score) in enumerate(retrieved_results, 1):
        meta = doc.metadata
        title = meta.get("title", "Unknown Title")
        arxiv_id = meta.get("arxiv_id", "Unknown ID")
        section = meta.get("section", "General")
        context_blocks.append(
            f"[Source {idx}]: {title} (arXiv:{arxiv_id} | Section: {section} | Match Distance: {score:.4f})\n"
            f"Excerpt: {doc.page_content.strip()}"
        )

    context_str = "\n\n".join(context_blocks)

    # 3. Guardrailed Executive Prompt
    prompt = f"""You are an AI Strategy Senior Consultant at KPMG Advisory.
Your audience is an enterprise executive evaluating cutting-edge AI research preprints.

GUARDRAIL RULES:
- Use ONLY the facts provided in the Research Context below.
- Do NOT hallucinate, speculate, or introduce external knowledge not grounded in the text.
- If the excerpts do not contain enough evidence to answer, state: "Insufficient evidence in current corpus."

Research Context:
{context_str}

Client Business Inquiry:
{query}

Format your response strictly using these four headings:
### 1. Executive Summary
Provide a 2-3 sentence non-technical synthesis explaining the core capability and why it matters to enterprise leadership.

### 2. Operational & Cost Impact
Describe practical impacts on compute resource allocation, latency, API costs, or pipeline efficiency based on the excerpts.

### 3. Strategic Risks & Governance
Identify limitations, alignment vulnerabilities, compliance considerations, or deployment bottlenecks noted in the research.

### 4. Cited Evidence & Precedents
List each cited finding with its exact paper title and arXiv ID as given in the Research Context.
"""

    # 4. Generate summary using Gemini Flash
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def main():
    # Setup data and vector store
    data_dir = get_data_dir()
    vectorstore = build_clean_vectorstore(data_dir)

    # Test Executive Queries
    test_queries = [
        "How do we balance cost, latency, and accuracy when routing queries across different AI models?",
        "What are the primary operational risks and evaluation frameworks for AI agents executing computer actions?"
    ]

    print("\n" + "=" * 65)
    print("RUNNING TASK #2: END-TO-END RETRIEVAL & SUMMARIZATION DEMO")
    print("=" * 65)

    for q_idx, q in enumerate(test_queries, 1):
        print(f"\n>>> QUERY #{q_idx}: {q}\n")
        insight_memo = generate_kpmg_executive_insight(q, vectorstore)
        print(insight_memo)
        print("\n" + "-" * 65)


if __name__ == "__main__":
    main()