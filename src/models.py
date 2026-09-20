from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

MemoryType = Literal['episodic', 'semantic', 'procedural']


class Critique(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)
    decision: Literal['accept', 'revise']
    reason: str = Field(max_length=1000)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=10)
    revision_instructions: str = Field(default='', max_length=2000)


class MemoryProposal(BaseModel):
    memory_type: Literal['semantic', 'procedural']
    content: str = Field(min_length=5, max_length=1000)
    provenance: str = Field(max_length=500)


class Curation(BaseModel):
    summary: str = Field(min_length=1, max_length=2000)
    memories: list[MemoryProposal] = Field(default_factory=list, max_length=4)


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    question: str
    user_id: str
    thread_id: str
    manual_model: str
    top_k: int
    memory_thread_id: str | None
    plan: list[str]
    selected_model: str
    actual_model: str
    routing_reason: str
    routing_category: str
    rag_needed: bool
    evidence: list[dict]
    memories: list[dict]
    retrieval_ids: list[str]
    evidence_refs: list[dict]
    memory_ids: list[str]
    draft_answer: str
    final_answer: str
    critic: dict
    critic_status: str
    revision_count: int
    temporary_response: str
    summary_saved: bool
