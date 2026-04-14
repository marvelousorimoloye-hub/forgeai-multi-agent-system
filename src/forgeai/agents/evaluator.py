from langchain_core.messages import AIMessage
from typing import Dict, Any

from src.forgeai.utils.pydantic_models import ResearchState


async def evaluator_node(state: ResearchState) -> Dict[str, Any]:
    """Evaluator Agent - Placeholder"""
    return {
        "faithfulness_score": 0.75,  # Placeholder score
        "messages": [AIMessage(content="✅ Evaluation completed (placeholder).")],
    }