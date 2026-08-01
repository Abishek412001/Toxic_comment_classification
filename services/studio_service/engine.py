"""
Enterprise AI Studio Orchestrator combining Prompts, Workflows, Agents & RAG.
"""

from services.studio_service.prompt_manager import prompt_manager
from services.studio_service.workflow_builder import workflow_builder
from services.studio_service.agent_framework import agent_framework
from services.studio_service.rag_engine import rag_engine
from services.studio_service.schemas import (
    PromptTemplateRequest,
    PromptTemplateResponse,
    PromptEvaluationRequest,
    PromptEvaluationResponse,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    AgentExecutionRequest,
    AgentExecutionResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)


class StudioEngine:
    """Enterprise AI Studio Orchestration Engine."""

    def save_template(self, request: PromptTemplateRequest) -> PromptTemplateResponse:
        return prompt_manager.save_template(request)

    def evaluate_prompt(self, request: PromptEvaluationRequest) -> PromptEvaluationResponse:
        return prompt_manager.evaluate_prompt(request)

    def execute_workflow(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResponse:
        return workflow_builder.execute_workflow(request)

    def run_agent(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        return agent_framework.run_agent(request)

    def query_knowledge_base(self, request: RAGQueryRequest) -> RAGQueryResponse:
        return rag_engine.query_knowledge_base(request)


studio_engine = StudioEngine()
