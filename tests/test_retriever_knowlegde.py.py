import asyncio
from src.forgeai.agents.retriever_knowledge import knowledge_retriever_node
from src.forgeai.utils.pydantic_models import ResearchState
from src.forgeai.rag.vector_store import auto_ingest_on_startup
from langchain_core.messages import HumanMessage


async def test_node():
    print("🚀 Starting Knowledge Retriever Test...")

    # Ingest documents first
    print("📥 Ingesting PDFs...")
    await asyncio.to_thread(auto_ingest_on_startup)

    # Create proper state
    state: ResearchState = {
        "query": "What are the latest advancements in multimodal large language models in 2025-2026?",
        "user_id": "test",
        "messages": [HumanMessage(content="What are the latest advancements in multimodal large language models in 2025-2026?")],
        "raw_documents": [],
        "engineered_context": "",
        "final_report": "",
        "citations": [],
        "needs_reflection": False,
        "current_iteration": 0,
    }

    print("🔍 Calling knowledge_retriever_node...")
    try:
        result = await asyncio.wait_for(
            knowledge_retriever_node(state),
            timeout=60.0   # 60 second timeout
        )

        if result.get("raw_documents"):
            print(f"✅ SUCCESS: Retrieved {len(result['raw_documents'])} documents after reranking")
            for i, doc in enumerate(result["raw_documents"][:5]):   # show first 5
                score = doc.get("relevance_score", 0.0)
                title = doc.get("metadata", {}).get("title", "No title")
                print(f"   [{i+1}] Score: {score:.4f} | Title: {title[:80]}...")
        else:
            print("⚠️ No documents returned from knowledge retriever.")

    except asyncio.TimeoutError:
        print("❌ TIMEOUT: Knowledge retriever took too long")
    except Exception as e:
        print(f"❌ ERROR in knowledge retriever: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_node())