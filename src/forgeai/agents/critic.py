from langchain_core.messages import AIMessage
from typing import Dict, Any
import json
import re

from src.forgeai.utils.pydantic_models import ResearchState
from src.forgeai.config.prompts import critic_prompt
from src.forgeai.utils.llm import get_fast_llm


async def critic_node(state: ResearchState) -> Dict[str, Any]:
    """Real Critic Agent - Evaluates context quality and triggers reflection"""
    
    print("→ Entering Critic Node")

    if not state.get("engineered_context") or len(state["engineered_context"]) < 200:
        return {
            "needs_reflection": True,
            "critique_scores": {"overall": 0.3},
            "messages": [AIMessage(content="Critic: Context too weak. Triggering reflection.")],
        }

    try:
        llm = get_fast_llm()

        prompt = critic_prompt.format_messages(
            query=state["query"],
            engineered_context=state["engineered_context"][:15000]
        )

        response = await llm.ainvoke(prompt)
        raw_content = response.content.strip()

        # Robust JSON extraction
        json_match = re.search(r'\{[\s\S]*?\}', raw_content)
        content = json_match.group(0) if json_match else raw_content

        critique = json.loads(content)

        overall_score = float(critique.get("overall_score", 0.65))
        needs_reflection = bool(critique.get("needs_reflection", False))

        current_iter = state.get("current_iteration", 0)
        max_iter = state.get("max_iterations", 3)

        if needs_reflection and current_iter < max_iter:
            print(f"🔄 Critic triggered reflection (Score: {overall_score:.2f})")
            return {
                "needs_reflection": True,
                "critique_scores": {"overall": overall_score},
                "messages": [AIMessage(content=f"Critic: Score {overall_score:.2f} - Needs better context.")],
                "sub_queries": critique.get("suggested_sub_queries", [state["query"]]),
                "current_iteration": current_iter + 1
            }
        else:
            print(f"✅ Critic approved context (Score: {overall_score:.2f})")
            return {
                "needs_reflection": False,
                "critique_scores": {"overall": overall_score},
                "messages": [AIMessage(content=f"Critic: Context approved (Score: {overall_score:.2f})")],
            }

    except Exception as e:
        print(f"Critic error: {e}")
        return {
            "needs_reflection": False,
            "critique_scores": {"overall": 0.7},
            "messages": [AIMessage(content="Critic: Proceeding with current context (fallback).")],
        }