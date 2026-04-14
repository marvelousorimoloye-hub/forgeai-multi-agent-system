from typing import List, Dict, Optional, TypedDict, Annotated, Union
import operator
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class HierarchicalContext(BaseModel):
    """Multi-level hierarchical context with flexible input handling"""
    level1_summary: str = ""
    level2_findings: str = ""          # Will auto-convert list to string
    level3_evidence: str = ""          # Will auto-convert list to string
    compression_stats: Dict[str, float] = Field(default_factory=dict)

    @field_validator('level2_findings', 'level3_evidence', mode='before')
    @classmethod
    def convert_list_to_string(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return str(v) if v is not None else ""

        
class ResearchState(TypedDict):
    """Main LangGraph state with full hierarchical context support"""
    
    # User input
    query: str
    user_id: Optional[str] = None
    
    # Messages for conversation history
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Retrieved content
    raw_documents: Annotated[list, operator.add]
    
    # ★ Hierarchical Context Engineering (Showcase Feature)
    hierarchical_context: HierarchicalContext = Field(default_factory=HierarchicalContext)
    
    # Legacy flat context (still used by synthesizer for simplicity)
    engineered_context: str = ""
    
    # Intermediate data
    sub_queries: List[str] = Field(default_factory=list)
    critique_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Final output
    final_report: str = ""
    citations: List[Dict] = Field(default_factory=list)
    
    # Monitoring & Evaluation
    token_usage: Dict[str, int] = Field(default_factory=dict)
    step_times: Dict[str, float] = Field(default_factory=dict)
    faithfulness_score: Optional[float] = None
    context_compression_ratio: Optional[float] = None
    
    # Control flags for agentic loop
    needs_reflection: bool = False
    max_iterations: int = 3
    current_iteration: int = 0


class AgentResponse(BaseModel):
    """Structured output every agent should return"""
    content: str
    next: str = Field(..., description="Next node: supervisor, critic, context_engineer, synthesizer, etc.")
    metadata: Dict = Field(default_factory=dict)


class Citation(BaseModel):
    """Standard citation format"""
    source: str
    title: str
    url: Optional[str] = None
    snippet: str
    relevance_score: float = 0.0


class SupervisorDecision(BaseModel):
    """Structured output for Supervisor Agent"""
    analysis: str = Field(..., description="Brief reasoning for the decision")
    next: str = Field(..., description="Next node: 'web_retriever' or 'knowledge_retriever'")
    sub_queries: List[str] = Field(default_factory=list, description="2-4 helpful sub-questions")