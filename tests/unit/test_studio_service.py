"""
Unit Tests for AI Studio, Prompts, Workflows, Autonomous Agents & RAG Engine.
"""

import unittest
from services.studio_service.engine import StudioEngine
from services.studio_service.schemas import (
    PromptTemplateRequest,
    PromptEvaluationRequest,
    WorkflowExecutionRequest,
    WorkflowNode,
    AgentExecutionRequest,
    RAGQueryRequest,
)


class TestStudioService(unittest.TestCase):
    def setUp(self):
        self.engine = StudioEngine()

    def test_save_and_evaluate_prompt(self):
        save_req = PromptTemplateRequest(
            name="customer_support_moderation",
            template_text="Evaluate user inquiry: {{user_input}}",
            tags=["moderation", "support"],
        )
        save_res = self.engine.save_template(save_req)
        self.assertTrue(save_res.prompt_id.startswith("prmt_"))
        self.assertEqual(save_res.name, "customer_support_moderation")

        eval_req = PromptEvaluationRequest(
            prompt_text=save_res.template_text,
            input_variables={"user_input": "How do I reset my password?"},
        )
        eval_res = self.engine.evaluate_prompt(eval_req)
        self.assertEqual(eval_res.rendered_prompt, "Evaluate user inquiry: How do I reset my password?")
        self.assertTrue(eval_res.is_safe)

    def test_execute_workflow(self):
        wf_req = WorkflowExecutionRequest(
            workflow_name="enterprise_moderation_pipeline",
            nodes=[
                WorkflowNode(node_id="n1", node_type="MODERATION_NODE"),
                WorkflowNode(node_id="n2", node_type="POLICY_NODE"),
            ],
            input_text="Check this message for compliance.",
        )
        wf_res = self.engine.execute_workflow(wf_req)
        self.assertEqual(wf_res.status, "COMPLETED")
        self.assertEqual(wf_res.executed_nodes_count, 2)

    def test_run_agent(self):
        agent_req = AgentExecutionRequest(
            agent_type="ComplianceAgent",
            task_description="Verify HIPAA compliance for incoming customer chat session.",
        )
        agent_res = self.engine.run_agent(agent_req)
        self.assertEqual(agent_res.status, "SUCCESS")
        self.assertGreater(len(agent_res.actions_taken), 0)

    def test_query_knowledge_base(self):
        rag_req = RAGQueryRequest(query="What is the PII redaction policy?", top_k=2)
        rag_res = self.engine.query_knowledge_base(rag_req)
        self.assertEqual(len(rag_res.citations), 2)
        self.assertIsNotNone(rag_res.answer_summary)


if __name__ == "__main__":
    unittest.main()
