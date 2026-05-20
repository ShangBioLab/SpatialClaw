import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer


MODULE_PATH = Path(__file__).resolve().parent / "remem_reason_server.py"
MODULE_NAME = "remem_reason_server_test_module"


def load_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReMemReasonServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mod = load_module()
        self.client = TestClient(TestServer(self.mod.create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        sys.modules.pop(MODULE_NAME, None)

    async def test_healthz(self):
        response = await self.client.get("/healthz")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("upstream_base_url", payload)
        self.assertIn("default_model", payload)

    async def test_remem_reason_requires_query_and_model(self):
        response = await self.client.post("/remem_reason", json={})
        self.assertEqual(response.status, 400)
        payload = await response.json()
        self.assertEqual(payload["error"], "query is required")

        response = await self.client.post("/remem_reason", json={"query": "hello"})
        self.assertEqual(response.status, 400)
        payload = await response.json()
        self.assertEqual(payload["error"], "model is required")

    async def test_remem_reason_empty_candidates(self):
        response = await self.client.post(
            "/remem_reason",
            json={"query": "continue analysis", "model": "mock-model", "candidates": []},
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["summary"], "")
        self.assertEqual(payload["used_memory_ids"], [])
        self.assertEqual(payload["callbacks"], [])
        self.assertEqual(payload["trace"]["steps"], 0)

    async def test_remem_reason_loop_and_summary(self):
        responses = iter(
            [
                "<thinking>first</thinking><update>Dataset is Visium.</update><recall>previous preprocessing run</recall>",
                "<thinking>second</thinking><update>Dataset is Visium and clustering already finished.</update>",
                "<summary>Use the prior Visium dataset and preserve the preprocessing lineage.</summary>",
            ]
        )

        async def fake_chat_once(*args, **kwargs):
            return next(responses)

        with patch.object(self.mod, "_chat_once", side_effect=fake_chat_once):
            response = await self.client.post(
                "/remem_reason",
                json={
                    "session_id": "cli:u1:c1",
                    "query": "continue the previous spatial analysis",
                    "model": "mock-model",
                    "candidates": [
                        {"id": "mem_1", "type": "dataset", "text": "platform: Visium"},
                        {"id": "mem_2", "type": "analysis", "text": "status: clustered"},
                    ],
                },
            )

        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(
            payload["summary"],
            "Use the prior Visium dataset and preserve the preprocessing lineage.",
        )
        self.assertEqual(payload["used_memory_ids"], ["mem_1", "mem_2"])
        self.assertEqual(payload["callbacks"], ["previous preprocessing run"])
        self.assertEqual(payload["trace"]["steps"], 2)
        self.assertEqual(payload["trace"]["upstream_model"], "mock-model")
        self.assertEqual(payload["trace"]["step_trace"][0]["candidate_type"], "dataset")

    async def test_remem_reason_honors_service_token(self):
        with patch.object(self.mod, "SERVICE_API_TOKEN", "secret-token"):
            response = await self.client.post(
                "/remem_reason",
                json={"query": "continue analysis", "model": "mock-model"},
            )
            self.assertEqual(response.status, 401)
            payload = await response.json()
            self.assertEqual(payload["error"], "unauthorized")


if __name__ == "__main__":
    unittest.main()
