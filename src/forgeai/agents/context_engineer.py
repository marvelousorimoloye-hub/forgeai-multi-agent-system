from langchain_core.messages import AIMessage
from typing import Dict, Any
import re

from src.forgeai.utils.pydantic_models import ResearchState, HierarchicalContext
from src.forgeai.config.prompts import hierarchical_context_prompt
from src.forgeai.utils.llm import get_strong_llm


async def context_engineer_node(state: ResearchState) -> Dict[str, Any]:
    """Hierarchical Context Engineer - Non-JSON (Stable & Reliable)"""
    
    if not state.get("raw_documents") or len(state["raw_documents"]) == 0:
        empty = HierarchicalContext(
            level1_summary="No relevant documents were retrieved for this query.",
            level2_findings="",
            level3_evidence=""
        )
        return {
            "hierarchical_context": empty,
            "engineered_context": "No relevant documents were retrieved.",
            "context_compression_ratio": 0.0,
            "messages": [AIMessage(content="No documents available for context engineering.")],
            "needs_reflection": False
        }

    # Prepare raw text
    raw_docs_text = "\n\n".join([
        f"--- Source {i+1} ---\n{doc.get('page_content', doc.get('content', ''))}"
        for i, doc in enumerate(state["raw_documents"])
    ])

    original_length = len(raw_docs_text)

    try:
        llm = get_strong_llm()

        prompt = hierarchical_context_prompt.format_messages(
            query=state["query"],
            raw_documents=raw_docs_text[:100000]   # Safe limit
        )

        response = await llm.ainvoke(prompt)
        raw_output = response.content.strip()

        print("Raw Context Engineer Output (preview):", repr(raw_output[:500]) + "...")

        # Parse the structured sections
        level1 = re.search(r'LEVEL 1:?\s*EXECUTIVE SUMMARY(.*?)(?=LEVEL 2:|$)', raw_output, re.DOTALL | re.IGNORECASE)
        level2 = re.search(r'LEVEL 2:?\s*KEY FINDINGS(.*?)(?=LEVEL 3:|$)', raw_output, re.DOTALL | re.IGNORECASE)
        level3 = re.search(r'LEVEL 3:?\s*SUPPORTING EVIDENCE(.*?)$', raw_output, re.DOTALL | re.IGNORECASE)

        hierarchy = HierarchicalContext(
            level1_summary=level1.group(1).strip() if level1 else raw_output[:1000],
            level2_findings=level2.group(1).strip() if level2 else "",
            level3_evidence=level3.group(1).strip() if level3 else ""
        )

        # Create final engineered context for synthesizer
        final_context = f"""EXECUTIVE SUMMARY (Level 1)
{hierarchy.level1_summary}

KEY FINDINGS (Level 2)
{hierarchy.level2_findings}

SUPPORTING EVIDENCE (Level 3)
{hierarchy.level3_evidence[:8000] if hierarchy.level3_evidence else ''}
"""

        compression_ratio = len(final_context) / original_length if original_length > 0 else 0.0

        return {
            "hierarchical_context": hierarchy,
            "engineered_context": final_context,
            "context_compression_ratio": round(compression_ratio, 4),
            "citations": state.get("citations", []),
            "messages": [AIMessage(content=f"✅ Hierarchical Context Engineered (3 Levels). Compression: {compression_ratio:.2%}")],
            "needs_reflection": False
        }

    except Exception as e:
        print(f"Context engineering error: {e}")
        fallback = raw_docs_text[:65000]
        return {
            "hierarchical_context": HierarchicalContext(
                level1_summary="Fallback: Raw content processed",
                level2_findings="",
                level3_evidence=fallback[:3000]
            ),
            "engineered_context": fallback,
            "context_compression_ratio": 1.0,
            "messages": [AIMessage(content="Context engineering used fallback mode.")],
            "needs_reflection": True
        }