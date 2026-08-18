from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest

import httpx

from mikucli.cancellation import StopRequested, stop_context
from mikucli.llm import BigModelClient, ModelClient


class ModelClientTests(unittest.TestCase):
    def test_chat_posts_payload_and_parses_response(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "done", "tool_calls": []}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                },
            )

        client = ModelClient(
            "secret",
            "https://example.test/chat",
            transport=httpx.MockTransport(handler),
        )

        response = client.chat(
            model="provider/model-name",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )

        self.assertEqual(response.content, "done")
        self.assertEqual(response.token_usage.total_tokens, 5)
        self.assertEqual(requests[0].headers["authorization"], "Bearer secret")
        self.assertEqual(json.loads(requests[0].content)["model"], "provider/model-name")

    def test_chat_cancels_in_flight_request_when_task_stops(self) -> None:
        stop = threading.Event()
        request_started = threading.Event()
        request_cancelled = threading.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            del request
            request_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                request_cancelled.set()
                raise
            return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})

        client = ModelClient(
            "secret",
            "https://example.test/chat",
            transport=httpx.MockTransport(handler),
        )
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        started_at = time.monotonic()
        try:
            with stop_context(stop.is_set):
                with self.assertRaises(StopRequested):
                    client.chat(model="provider/model-name", messages=[], tools=[])
        finally:
            timer.cancel()

        self.assertTrue(request_started.is_set())
        self.assertTrue(request_cancelled.is_set())
        self.assertLess(time.monotonic() - started_at, 1.0)

    def test_legacy_bigmodel_client_name_remains_an_alias(self) -> None:
        self.assertIs(BigModelClient, ModelClient)


if __name__ == "__main__":
    unittest.main()
