from langchain_core.messages import AIMessage, HumanMessage
from typing import Dict, Any, List
from tavily import TavilyClient

from src.forgeai.utils.pydantic_models import ResearchState
from src.forgeai.config.settings import settings
from src.forgeai.config.prompts import SUPERVISOR_SYSTEM_PROMPT 


async def web_retriever_node(state: ResearchState) -> Dict[str, Any]:
    """
    Web Retriever Agent using Tavily
    Fetches latest, high-quality web results for real-time or recent information.
    """
    
    if not settings.tavily_api_key and not getattr(settings, 'tavily_api_key', None):
        return {
            "raw_documents": [],
            "messages": [AIMessage(content="⚠️ Tavily API key not configured. Web retrieval skipped.")],
        }

    try:
        tavily_client = TavilyClient(api_key=settings.tavily_api_key)

        # Use sub_queries if available, otherwise fall back to main query
        search_queries = state.get("sub_queries", []) or [state["query"]]
        
        all_results = []

        for query in search_queries[:3]:  # Limit to 3 parallel searches
            response = tavily_client.search(
                query=query,
                search_depth="advanced",      # "basic" or "advanced"
                max_results=8,
                include_answer=True,
                include_raw_content=False
            )
            
            # Convert Tavily results to standard document format
            for result in response.get("results", []):
                doc = {
                    "page_content": result.get("content", ""),
                    "metadata": {
                        "source": "web",                    
                        "source_type": "web",
                        "title": result.get("title", "Untitled"),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0.0),
                        "query": query
                    }
                }
                all_results.append(doc)

        # Add to state
        return {
            "raw_documents": all_results,
            "messages": [AIMessage(
                content=f"✅ Web retrieval completed. Found {len(all_results)} results from Tavily."
            )],
        }

    except Exception as e:
        print(f"Web retriever error: {str(e)}")
        return {
            "raw_documents": [],
            "messages": [AIMessage(content=f"Web retrieval failed: {str(e)[:100]}")],
        }