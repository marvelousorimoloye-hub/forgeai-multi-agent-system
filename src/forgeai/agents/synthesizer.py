from langchain_core.messages import AIMessage
from typing import Dict, Any

from src.forgeai.utils.pydantic_models import ResearchState
from src.forgeai.config.prompts import synthesizer_prompt
from src.forgeai.utils.llm import get_creative_llm


async def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """Advanced Synthesizer with Real Citation Support"""
    
    if not state.get("engineered_context") or len(state["engineered_context"].strip()) < 100:
        report = "Insufficient context available to generate report."
    else:
        try:
            llm = get_creative_llm()

            # Prepare citations for the prompt
            citations_text = ""
            if state.get("citations"):
                citations_text = "\nAvailable Sources:\n"
                for i, cit in enumerate(state["citations"][:15], 1):
                    title = cit.get("metadata", {}).get("title", "Untitled")
                    url = cit.get("metadata", {}).get("url", "")
                    citations_text += f"[{i}] {title} - {url}\n"

            prompt = synthesizer_prompt.format_messages(
                engineered_context=state["engineered_context"],
                query=state["query"],
                citations=citations_text
            )

            response = await llm.ainvoke(prompt)
            report = response.content.strip()

        except Exception as e:
            report = f"Error generating report: {str(e)}"

    return {
        "final_report": report,
        "citations": state.get("citations", []),
        "messages": [AIMessage(content="✅ Final report synthesized with real citations.")],
    }