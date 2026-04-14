from langchain_core.messages import AIMessage
from typing import Dict, Any
import json
import re

from src.forgeai.utils.pydantic_models import ResearchState, SupervisorDecision
from src.forgeai.config.prompts import supervisor_prompt
from src.forgeai.utils.llm import get_fast_llm


async def supervisor_node(state: ResearchState) -> Dict[str, Any]:
    """Supervisor Agent - Decides retrieval strategy with robust parsing"""
    
    llm = get_fast_llm()
    
    prompt = supervisor_prompt.format_messages(
        query=state["query"],
        messages=state.get("messages", [])
    )
    
    response = await llm.ainvoke(prompt)
    raw_content = response.content.strip()

    try:
        # Robust JSON extraction
        json_match = re.search(r'\{[\s\S]*?\}', raw_content)
        content = json_match.group(0) if json_match else raw_content

        decision = json.loads(content)

        next_action = decision.get("next", "both").lower()
        
        # Normalize to valid options
        if next_action not in ["web_retriever", "knowledge_retriever", "both"]:
            next_action = "both"

        return {
            "messages": [AIMessage(content=f"Supervisor: {decision.get('analysis', 'Decided retrieval strategy')}")],
            "sub_queries": decision.get("sub_queries", [state["query"]]),
            "next": next_action,                    # Used by graph routing
            "needs_reflection": False
        }

    except Exception as e:
        print(f"Supervisor parsing error: {e}")
        print("Raw output:", raw_content[:300])
        
        # Safe fallback - prefer both for better coverage
        return {
            "messages": [AIMessage(content="Supervisor fallback: Using both retrievers.")],
            "sub_queries": [state["query"]],
            "next": "both",
            "needs_reflection": False
        }