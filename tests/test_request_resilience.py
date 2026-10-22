from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx

from mikucli.cancellation import StopRequested, stop_context
from mikucli.codebase.embeddings import EmbeddingError, OllamaEmbeddingClient
from mikucli.http_requests import (
    ModelResponseError,
    ProviderRequestError,
    RequestPolicy,
    retry_after,
)
from mikucli.llm import ModelClient


def success() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


class RequestResilienceTests(unittest.TestCase):
    def client(self, handler, **policy):
        return ModelClient(
            "dummy",
            "https://example.test/chat",
            transport=httpx.MockTransport(handler),
            request_policy=RequestPolicy(max_retry_delay=0, **policy),
        )

    def test_transient_statuses_retry_but_authentication_errors_do_not(self):
        for status in (408, 429, 500, 502, 503, 504):
            calls = []

            def handler(request, calls=calls, status=status):
                calls.append(request)
                return (
                    httpx.Response(status, headers={"Retry-After": "0"})
                    if len(calls) == 1
                    else success()
                )

            with self.subTest(status=status):
                result = self.client(handler).chat(model="test", messages=[], tools=[])
                self.assertEqual(result.request_attempts, 2)
                self.assertEqual(result.content, "done")
        calls = []

        def denied(request):
            calls.append(request)
            return httpx.Response(401)

        with self.assertRaisesRegex(ProviderRequestError, "HTTP 401"):
            self.client(denied).chat(model="test", messages=[], tools=[])
        self.assertEqual(len(calls), 1)

    def test_connection_failure_retries_and_exhaustion_is_bounded(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ConnectError("offline")

        with self.assertRaisesRegex(ProviderRequestError, "Exhausted 3"):
            self.client(handler).chat(model="test", messages=[], tools=[])
        self.assertEqual(len(calls), 3)

    def test_retry_after_is_respected_and_large_wait_is_refused(self):
        calls = []

        def handler(request):
            calls.append(time.monotonic())
            return (
                httpx.Response(429, headers={"Retry-After": "0.08"})
                if len(calls) == 1
                else success()
            )

        client = ModelClient(
            "dummy",
            "https://example.test/chat",
            transport=httpx.MockTransport(handler),
            request_policy=RequestPolicy(max_retry_delay=0.2),
        )
        client.chat(model="test", messages=[], tools=[])
        self.assertGreaterEqual(calls[1] - calls[0], 0.07)
        with self.assertRaisesRegex(ProviderRequestError, "retry-delay limit"):
            self.client(
                lambda _: httpx.Response(429, headers={"Retry-After": "100"})
            ).chat(model="test", messages=[], tools=[])
        date = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
        self.assertGreater(retry_after(date), 27)
        self.assertIsNone(retry_after("nan"))

    def test_user_can_cancel_during_retry_backoff(self):
        stop = threading.Event()

        def handler(request):
            return httpx.Response(503, headers={"Retry-After": "10"})

        client = ModelClient(
            "dummy", "https://example.test/chat", transport=httpx.MockTransport(handler)
        )
        started = time.monotonic()
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        try:
            with stop_context(stop.is_set), self.assertRaises(StopRequested):
                client.chat(model="test", messages=[], tools=[])
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 1)

    def test_each_request_timeout_cancels_transport_before_retry(self):
        calls, cancelled = [], []

        async def handler(request):
            calls.append(request)
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return success()

        with self.assertRaises(ProviderRequestError):
            self.client(handler, timeout_seconds=0.03, max_retries=1).chat(
                model="test", messages=[], tools=[]
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(cancelled), 2)

    def test_malformed_or_incomplete_success_responses_are_not_retried(self):
        payloads = [
            {},
            {"choices": []},
            {"choices": [{"message": {"content": ""}}]},
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "partial"}}
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "read_file", "arguments": "[]"}}
                            ]
                        }
                    }
                ]
            },
        ]
        duplicate = {"id": "same", "function": {"name": "read_file", "arguments": "{}"}}
        payloads.extend([
            {"choices": [{"message": {"tool_calls": [duplicate, duplicate]}}]},
            {"choices": [{"message": {"tool_calls": [duplicate] * 129}}]},
        ])
        for payload in payloads:
            calls = []

            def handler(request, calls=calls, payload=payload):
                calls.append(request)
                return httpx.Response(200, json=payload)

            with self.subTest(payload=payload), self.assertRaises(ModelResponseError):
                self.client(handler).chat(model="test", messages=[], tools=[])
            self.assertEqual(len(calls), 1)
        with self.assertRaises(ModelResponseError):
            self.client(lambda _: httpx.Response(200, content="not json")).chat(
                model="test", messages=[], tools=[]
            )

    def test_response_body_size_is_bounded(self):
        client = ModelClient(
            "dummy",
            "https://example.test/chat",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=b"x" * 5000)
            ),
            request_policy=RequestPolicy(max_response_bytes=1024),
        )
        with self.assertRaisesRegex(ModelResponseError, "response-size limit"):
            client.chat(model="test", messages=[], tools=[])

    def test_embeddings_use_retries_and_fail_with_recovery_instructions(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return (
                httpx.Response(503)
                if len(calls) == 1
                else httpx.Response(200, json={"embeddings": [[1, 0]]})
            )

        client = OllamaEmbeddingClient(
            model="test-embed",
            base_url="http://example.test",
            request_policy=RequestPolicy(max_retry_delay=0),
            transport=httpx.MockTransport(handler),
        )
        self.assertEqual(client.embed(["query"]), [[1.0, 0.0]])
        self.assertEqual(len(calls), 2)
        client.transport = httpx.MockTransport(lambda _: httpx.Response(404))
        with self.assertRaisesRegex(EmbeddingError, "ollama pull test-embed"):
            client.embed(["query"])
        client.transport = httpx.MockTransport(
            lambda _: httpx.Response(200, content=b'{"embeddings":[[NaN]]}')
        )
        with self.assertRaisesRegex(EmbeddingError, "invalid embedding vectors"):
            client.embed(["query"])


if __name__ == "__main__":
    unittest.main()
