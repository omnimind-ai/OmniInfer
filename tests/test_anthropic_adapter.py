#!/usr/bin/env python3
"""Anthropic <-> OpenAI format conversion tests."""

from __future__ import annotations

import unittest

from service_core.anthropic_adapter import (
    anthropic_request_to_openai,
    openai_response_to_anthropic,
)


# ---------------------------------------------------------------------------
# Request conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------


class AnthropicRequestToOpenaiTests(unittest.TestCase):
    # --- basic messages ---

    def test_simple_text_message(self) -> None:
        result = anthropic_request_to_openai({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        })
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["max_tokens"], 100)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0], {"role": "user", "content": "hello"})

    def test_empty_messages(self) -> None:
        result = anthropic_request_to_openai({"messages": []})
        self.assertEqual(result["messages"], [])

    # --- system prompt ---

    def test_system_string(self) -> None:
        result = anthropic_request_to_openai({
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(result["messages"][0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(result["messages"][1], {"role": "user", "content": "hi"})

    def test_system_block_list(self) -> None:
        result = anthropic_request_to_openai({
            "system": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
            "messages": [],
        })
        self.assertEqual(result["messages"][0]["content"], "Part 1\nPart 2")

    def test_empty_system_ignored(self) -> None:
        result = anthropic_request_to_openai({
            "system": "",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(len(result["messages"]), 1)

    # --- scalar parameters ---

    def test_scalar_parameters_mapped(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
            "stream": True,
        })
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["top_p"], 0.9)
        self.assertEqual(result["top_k"], 40)
        self.assertEqual(result["stop"], ["END"])
        self.assertTrue(result["stream"])

    # --- thinking mode ---

    def test_thinking_enabled(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        })
        self.assertTrue(result["think"])
        self.assertEqual(result["thinking_budget"], 1024)

    def test_thinking_disabled(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [],
            "thinking": {"type": "disabled"},
        })
        self.assertFalse(result["think"])

    # --- image content ---

    def test_base64_image_converted(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            ]}],
        })
        parts = result["messages"][0]["content"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], {"type": "text", "text": "describe"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_url_image_converted(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
            ]}],
        })
        parts = result["messages"][0]["content"]
        self.assertEqual(parts[1]["image_url"]["url"], "https://example.com/img.png")

    def test_single_text_block_flattened(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hello"},
            ]}],
        })
        self.assertEqual(result["messages"][0]["content"], "hello")

    # --- tool use ---

    def test_tools_converted(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [],
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
        })
        self.assertEqual(len(result["tools"]), 1)
        tool = result["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "get_weather")
        self.assertEqual(tool["function"]["parameters"], {"type": "object"})

    def test_tool_use_message_converted(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Tokyo"}},
            ]}],
        })
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Let me check.")
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["id"], "call_1")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")

    def test_tool_result_message_converted(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny, 25°C"},
            ]}],
        })
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "call_1")
        self.assertEqual(msg["content"], "Sunny, 25°C")

    # --- tool_choice ---

    def test_tool_choice_auto(self) -> None:
        result = anthropic_request_to_openai({"messages": [], "tool_choice": {"type": "auto"}})
        self.assertEqual(result["tool_choice"], "auto")

    def test_tool_choice_none(self) -> None:
        result = anthropic_request_to_openai({"messages": [], "tool_choice": {"type": "none"}})
        self.assertEqual(result["tool_choice"], "none")

    def test_tool_choice_specific(self) -> None:
        result = anthropic_request_to_openai({
            "messages": [],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        })
        self.assertEqual(result["tool_choice"]["function"]["name"], "get_weather")


# ---------------------------------------------------------------------------
# Response conversion: OpenAI → Anthropic
# ---------------------------------------------------------------------------


class OpenaiResponseToAnthropicTests(unittest.TestCase):
    def test_simple_text_response(self) -> None:
        oai = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_response_to_anthropic(oai, model="test-model")

        self.assertEqual(result["type"], "message")
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0], {"type": "text", "text": "Hello!"})
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 5)

    def test_tool_call_response(self) -> None:
        oai = {
            "choices": [{"message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                }],
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        result = openai_response_to_anthropic(oai, model="test")

        self.assertEqual(result["stop_reason"], "tool_use")
        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        self.assertEqual(tool_block["name"], "get_weather")
        self.assertEqual(tool_block["input"], {"city": "Tokyo"})

    def test_length_finish_reason(self) -> None:
        oai = {
            "choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}],
            "usage": {},
        }
        result = openai_response_to_anthropic(oai, model="test")
        self.assertEqual(result["stop_reason"], "max_tokens")

    def test_empty_response(self) -> None:
        oai = {"choices": [{"message": {}, "finish_reason": "stop"}], "usage": {}}
        result = openai_response_to_anthropic(oai, model="test")
        self.assertEqual(result["content"], [])
        self.assertEqual(result["usage"]["input_tokens"], 0)
        self.assertEqual(result["usage"]["output_tokens"], 0)

    def test_malformed_tool_arguments_fallback(self) -> None:
        oai = {
            "choices": [{"message": {
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "not json"}}],
            }, "finish_reason": "tool_calls"}],
            "usage": {},
        }
        result = openai_response_to_anthropic(oai, model="test")
        tool_block = result["content"][0]
        self.assertEqual(tool_block["input"], {})


if __name__ == "__main__":
    unittest.main()
