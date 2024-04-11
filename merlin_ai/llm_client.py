"""
Merlin LLM API client
"""
import json
from typing import Optional, Any, Callable
import time

import openai
from anthropic import Anthropic
from anthropic.types.beta.tools import ToolParam, ToolsBetaMessageParam, ToolsBetaMessage
from openai.util import convert_to_openai_object

CLAUDE_ENUM_MAX_TOKENS = 1500


def _try_get(
    getter: Callable[[], Optional[Any]], default: Optional[Any] = None
) -> Optional[Any]:
    """
    Execute getter, return default value if it fails
    """
    try:
        return getter()
    except Exception:  # pylint: disable=broad-exception-caught
        return default


class MerlinChatCompletion(openai.ChatCompletion):
    """
    Extends OpenAI ChatCompletion to support Claude models
    """

    # pylint: disable=abstract-method

    anthropic_api_key: str = None

    @staticmethod
    def _create_claude_request(**openai_request) -> dict[str, Any]:
        """
        Create Claude request
        """
        is_enum = "functions" not in openai_request
        if is_enum:
            function = {
                "name": "format_response",
                "parameters": {
                    "type": "object",
                    "required": [
                        "category",
                    ],
                    "properties": {
                        "category": {"type": "integer"},
                    },
                },
                "description": "Format response wrt identified category.",
            }
        else:
            function = _try_get(lambda: openai_request["functions"][0])
        system = _try_get(
            lambda: [
                message
                for message in openai_request["messages"]
                if message["role"] == "system"
            ][0]["content"],
            "",
        )
        tools: list[ToolParam] = [
            {
                "name": _try_get(lambda: function["name"], ""),
                "description": _try_get(lambda: function["description"], ""),
                "input_schema": {
                    "type": "object",
                    "properties": _try_get(
                        lambda: function["parameters"]["properties"], {}
                    ),
                    "required": _try_get(
                        lambda: function["parameters"]["required"], []
                    ),
                },
            }
        ]
        messages: list[ToolsBetaMessageParam] = [
            message
            for message in openai_request["messages"]
            if message["role"] != "system"
        ]
        return {
            "model": openai_request["model"],
            "max_tokens": CLAUDE_ENUM_MAX_TOKENS
            if is_enum
            else openai_request["max_tokens"],
            "system": system,
            "messages": messages,
            "tools": tools,
        }

    @staticmethod
    def _create_openai_response(claude_response: ToolsBetaMessage, is_enum: bool) -> Any:
        """
        Create OpenAI response
        """
        openai_response = {
            "id": claude_response.id,
            "model": claude_response.model,
            "usage": {
                "prompt_tokens": claude_response.usage.input_tokens,
                "completion_tokens": claude_response.usage.output_tokens,
                "total_tokens": claude_response.usage.input_tokens
                + claude_response.usage.output_tokens,
            },
            "object": "chat.completion",
            "created": int(time.time()),
        }
        tool = next(
            content for content in claude_response.content if content.type == "tool_use"
        )
        openai_response["choices"] = (
            [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "format_response",
                            "arguments": json.dumps(tool.input),
                        },
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ]
            if not is_enum
            else [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": tool.input["category"],
                    },
                    "finish_reason": "length",
                    "logprobs": None,
                }
            ]
        )
        return convert_to_openai_object(openai_response)

    @classmethod
    def _call_claude(cls, **kwargs):
        with Anthropic(
            api_key=cls.anthropic_api_key,
            default_headers={"anthropic-beta": "tools-2024-04-04"},
        ) as claude_client:
            claude_request = cls._create_claude_request(**kwargs)
            claude_response = claude_client.beta.tools.messages.create(**claude_request)
            return cls._create_openai_response(
                claude_response, is_enum="functions" not in kwargs
            )

    @classmethod
    def _claude_requested(cls, **kwargs):
        return (
            "model" in kwargs
            and kwargs["model"].startswith("claude-")
            # and cls.anthropic_api_key
        )

    @classmethod
    def create(cls, *args, **kwargs):
        if cls._claude_requested(**kwargs):
            return cls._call_claude(**kwargs)
        return super().create(*args, **kwargs)

    @classmethod
    async def acreate(cls, *args, **kwargs):
        if cls._claude_requested(**kwargs):
            raise NotImplementedError(
                "Merlin does not support async calls for Claude models"
            )
        return await super().acreate(*args, **kwargs)
