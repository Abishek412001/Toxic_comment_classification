"""
Integration Tests for AI Studio, Workflows, Agents & RAG API Gateway Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestStudioAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Login as Admin
        login_res = self.client.post(
            "/api/v1/auth/login",
            json={"email": "admin@opentrust.ai", "password": "AdminSecure2026!"},
        )
        self.token = login_res.json()["data"]["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_save_and_evaluate_prompt_endpoint(self):
        save_res = self.client.post(
            "/api/v1/studio/prompts/save",
            json={
                "name": "toxicity_filter_prompt",
                "template_text": "Classify text safety: {{text}}",
                "version": "1.0.0",
                "tags": ["safety"],
            },
            headers=self.headers,
        )
        self.assertEqual(save_res.status_code, 200)
        self.assertTrue(save_res.json()["success"])

        eval_res = self.client.post(
            "/api/v1/studio/prompts/evaluate",
            json={
                "prompt_text": "Classify text safety: {{text}}",
                "input_variables": {"text": "Hello world!"},
            },
        )
        self.assertEqual(eval_res.status_code, 200)
        self.assertTrue(eval_res.json()["data"]["is_safe"])

    def test_execute_workflow_and_agents_endpoint(self):
        wf_res = self.client.post(
            "/api/v1/studio/workflows/execute",
            json={
                "workflow_name": "live_chat_moderation",
                "nodes": [{"node_id": "n1", "node_type": "MODERATION_NODE"}],
                "input_text": "Testing workflow execution",
            },
            headers=self.headers,
        )
        self.assertEqual(wf_res.status_code, 200)
        self.assertEqual(wf_res.json()["data"]["status"], "COMPLETED")

        agent_res = self.client.post(
            "/api/v1/studio/agents/run",
            json={
                "agent_type": "RiskAnalysisAgent",
                "task_description": "Assess threat risk for flagged prompt",
            },
            headers=self.headers,
        )
        self.assertEqual(agent_res.status_code, 200)
        self.assertEqual(agent_res.json()["data"]["status"], "SUCCESS")

    def test_rag_query_endpoint(self):
        rag_res = self.client.post(
            "/api/v1/studio/rag/query",
            json={"query": "What are the human review thresholds?", "top_k": 2},
        )
        self.assertEqual(rag_res.status_code, 200)
        self.assertTrue(rag_res.json()["success"])
        self.assertEqual(len(rag_res.json()["data"]["citations"]), 2)


if __name__ == "__main__":
    unittest.main()
