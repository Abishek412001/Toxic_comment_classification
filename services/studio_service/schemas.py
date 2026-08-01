"""
Pydantic v2 Schemas for AI Studio, Prompt Management, Workflows, Agents, and RAG Search.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class PromptTemplateRequest(BaseSchema):
    name: str = Field(min_length=1)
    template_text: str = Field(min_length=1, description="Prompt text with {{variables}}")
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)


class PromptTemplateResponse(BaseSchema):
    prompt_id: str
    name: str
    template_text: str
    version: str
    tags: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PromptEvaluationRequest(BaseSchema):
    prompt_text: str
    input_variables: Dict[str, str] = Field(default_factory=dict)


class PromptEvaluationResponse(BaseSchema):
    rendered_prompt: str
    toxicity_score: float
    safety_score: float
    compliance_score: float
    latency_ms: float
    is_safe: bool


class WorkflowNode(BaseSchema):
    node_id: str
    node_type: str  # MODERATION_NODE, POLICY_NODE, HUMAN_REVIEW_NODE
    config: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionRequest(BaseSchema):
    workflow_name: str
    nodes: List[WorkflowNode]
    input_text: str


class WorkflowExecutionResponse(BaseSchema):
    execution_id: str
    workflow_name: str
    status: str  # COMPLETED, FAILED
    executed_nodes_count: int
    final_output: Dict[str, Any]
    total_latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentExecutionRequest(BaseSchema):
    agent_type: str = "ContentModerationAgent"
    task_description: str
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentExecutionResponse(BaseSchema):
    agent_id: str
    agent_type: str
    status: str  # SUCCESS, FAILED
    actions_taken: List[str]
    agent_memory: Dict[str, Any]
    output_summary: str


class CitationSnippet(BaseSchema):
    document_title: str
    content_snippet: str
    similarity_score: float


class RAGQueryRequest(BaseSchema):
    query: str
    top_k: int = 3


class RAGQueryResponse(BaseSchema):
    query: str
    citations: List[CitationSnippet]
    answer_summary: str
