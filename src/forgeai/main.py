import asyncio
import traceback
from langchain_core.messages import HumanMessage

from src.forgeai.graphs.research_graph import research_graph
from src.forgeai.utils.pydantic_models import ResearchState


async def run_research(query: str, user_id: str = "debug"):
    print(f"🚀 Starting ForgeAI research for query: {query}")
    print("→ Initializing state...")

    initial_state: ResearchState = {
        "query": query,
        "user_id": user_id,
        "messages": [HumanMessage(content=query)],
        "raw_documents": [],
        "engineered_context": "",
        "final_report": "",
        "citations": [],
        "needs_reflection": False,
        "current_iteration": 0,
    }
    
    try:
        print("→ Invoking research graph...")
        result = await research_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": user_id}}
        )
        
        print("\n" + "="*80)
        print("✅ RESEARCH COMPLETED")
        print("="*80)
        print(result.get("final_report", "No report was generated."))
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the latest advancements in multimodal large language models in 2025-2026?"
    asyncio.run(run_research(query))