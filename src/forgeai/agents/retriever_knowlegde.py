import asyncio
import voyageai
import cohere
from typing import Dict, Any
from langchain_core.messages import AIMessage

from src.forgeai.config.settings import settings
from src.forgeai.utils.pydantic_models import ResearchState
from src.forgeai.rag.vector_store import get_hybrid_retriever, auto_ingest_on_startup


async def knowledge_retriever_node(state: ResearchState) -> Dict[str, Any]:
    """Knowledge Retriever with Voyage Reranker + Cohere Fallback"""
    
    print("→ Entering Knowledge Retriever Node")

    try:
        # 1. Ensure documents are ingested
        await asyncio.to_thread(auto_ingest_on_startup)

        # 2. Hybrid retrieval (Vector + BM25)
        retriever = get_hybrid_retriever(k=15)
        docs = await asyncio.to_thread(retriever.invoke, state["query"])
        
        print(f"✅ Hybrid retrieval returned {len(docs)} documents")

        if not docs:
            return {
                "raw_documents": [],
                "citations": [],
                "messages": [AIMessage(content="Knowledge retriever: No documents found.")]
            }

        # 3. Reranking with Voyage → Cohere fallback
        doc_texts = [doc.page_content for doc in docs]
        reranked_docs = None

        # Try Voyage first
        try:
            vo = voyageai.Client(api_key=settings.voyage_api_key)
            reranking = vo.rerank(
                query=state["query"],
                documents=doc_texts,
                model="rerank-2-lite",
                top_k=10
            )
            reranked_docs = reranking.results
            print("✅ Used Voyage AI reranker")
        except Exception as ve:
            print(f"Voyage failed: {ve}. Trying Cohere fallback...")

            # Cohere fallback (very generous free tier)
            try:
                co = cohere.Client(api_key=settings.cohere_api_key)
                reranking = co.rerank(
                    query=state["query"],
                    documents=doc_texts,
                    model="rerank-english-v3.0",
                    top_n=10
                )
                reranked_docs = reranking.results
                print("✅ Used Cohere reranker as fallback")
            except Exception as ce:
                print(f"Cohere also failed: {ce}. Using original retrieval order as final fallback.")
                # Ultimate fallback: no reranking, take top documents as-is
                reranked_docs = [{"index": i, "relevance_score": 1.0 - (i * 0.01)} for i in range(min(10, len(docs)))]

        # 4. Build final output
        raw_documents = []
        citations = []

        for result in reranked_docs:
            idx = result.index if hasattr(result, 'index') else getattr(result, 'index', 0)
            score = float(result.relevance_score if hasattr(result, 'relevance_score') else getattr(result, 'relevance_score', 0.8))
            
            original_doc = docs[idx]
            metadata = original_doc.metadata.copy() if hasattr(original_doc, 'metadata') else {}
            metadata["source_type"] = "local"

            raw_documents.append({
                "page_content": original_doc.page_content,
                "metadata": metadata,
                "relevance_score": score
            })

            citations.append({
                "title": metadata.get("title", "Local Document"),
                "url": metadata.get("source", ""),
                "source_type": "local",
                "snippet": original_doc.page_content[:200]
            })

        return {
            "raw_documents": raw_documents,
            "citations": citations,
            "messages": [AIMessage(content=f"📚 Knowledge Base: {len(raw_documents)} documents after reranking.")]
        }

    except Exception as e:
        print(f"❌ Knowledge retriever error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "raw_documents": [],
            "citations": [],
            "messages": [AIMessage(content=f"Knowledge retriever failed: {str(e)}")]
        }