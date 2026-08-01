"""
AI Studio, Prompts, Workflows, Agents & RAG API Gateway Router.
"""

from fastapi import APIRouter, Depends
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
from services.studio_service.engine import studio_engine
from opentrust_core.auth.dependencies import check_api_key_rate_limit, require_role
from opentrust_core.auth.models import RoleEnum, UserRead
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/studio", tags=["Enterprise AI Studio & AI Agents"])


@router.post("/prompts/save", response_model=APIResponse[PromptTemplateResponse])
async def save_prompt_template(
    request: PromptTemplateRequest,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Saves a prompt template into repository with versioning."""
    res = studio_engine.save_template(request)
    return APIResponse[PromptTemplateResponse](
        data=res,
        message=f"Prompt template '{res.name}' saved.",
    )


@router.post("/prompts/evaluate", response_model=APIResponse[PromptEvaluationResponse])
async def evaluate_prompt(
    request: PromptEvaluationRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Evaluates prompt quality and safety score."""
    res = studio_engine.evaluate_prompt(request)
    return APIResponse[PromptEvaluationResponse](data=res)


@router.post("/workflows/execute", response_model=APIResponse[WorkflowExecutionResponse])
async def execute_workflow(
    request: WorkflowExecutionRequest,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Executes multi-node visual workflow pipeline graph."""
    res = studio_engine.execute_workflow(request)
    return APIResponse[WorkflowExecutionResponse](
        data=res,
        message=f"Workflow '{request.workflow_name}' executed successfully.",
    )


@router.post("/agents/run", response_model=APIResponse[AgentExecutionResponse])
async def run_agent(
    request: AgentExecutionRequest,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Executes autonomous safety agent actions and memory updates."""
    res = studio_engine.run_agent(request)
    return APIResponse[AgentExecutionResponse](
        data=res,
        message=res.output_summary,
    )


@router.post("/rag/query", response_model=APIResponse[RAGQueryResponse])
async def query_knowledge_base(
    request: RAGQueryRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Executes RAG semantic search over knowledge base documents."""
    res = studio_engine.query_knowledge_base(request)
    return APIResponse[RAGQueryResponse](data=res)
