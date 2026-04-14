from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Union

from src.forgeai.utils.pydantic_models import ResearchState

# Import nodes
from src.forgeai.agents.supervisor import supervisor_node
from src.forgeai.agents.retriever_web import web_retriever_node
from src.forgeai.agents.retriever_knowledge import knowledge_retriever_node
from src.forgeai.agents.merge_documents import merge_documents_node   # ← New node
from src.forgeai.agents.context_engineer import context_engineer_node
from src.forgeai.agents.critic import critic_node
from src.forgeai.agents.synthesizer import synthesizer_node
from src.forgeai.agents.evaluator import evaluator_node




def route_after_supervisor(state: ResearchState) -> Union[str, List[str]]:
    """
    Returns a string for single nodes, or a list of strings 
    to trigger parallel execution.
    """
    next_action = state.get("next", "both").lower()
    
    if next_action == "both":
        # Returning a list tells LangGraph to run these in parallel
        return ["web_retriever", "knowledge_retriever"]
    
    if next_action in ["web_retriever", "knowledge_retriever"]:
        return next_action
        
    return ["web_retriever", "knowledge_retriever"] # Safe default


def build_research_graph():
    graph = StateGraph(ResearchState)
    
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("web_retriever", web_retriever_node)
    graph.add_node("knowledge_retriever", knowledge_retriever_node)
    graph.add_node("merge_documents", merge_documents_node)   # ← New merge step
    graph.add_node("context_engineer", context_engineer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("evaluator", evaluator_node)
    
    graph.set_entry_point("supervisor")
    
    # Supervisor routing
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "web_retriever": "web_retriever",
            "knowledge_retriever": "knowledge_retriever",
        }
    )
    
    
    # After both retrievers → merge → context engineer
    graph.add_edge("web_retriever", "merge_documents")
    graph.add_edge("knowledge_retriever", "merge_documents")
    
    graph.add_edge("merge_documents", "context_engineer")
    graph.add_edge("context_engineer", "critic")
    
    graph.add_conditional_edges(
        "critic",
        lambda state: "context_engineer" if state.get("needs_reflection", False) else "synthesizer",
        {
            "context_engineer": "context_engineer",
            "synthesizer": "synthesizer"
        }
    )
    
    graph.add_edge("synthesizer", "evaluator")
    graph.add_edge("evaluator", END)
    
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    return compiled_graph


research_graph = build_research_graph()