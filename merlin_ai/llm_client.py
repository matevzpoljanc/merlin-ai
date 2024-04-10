import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Optional, Any, Callable
import time

import openai
from anthropic import Anthropic
from anthropic.types.beta.tools import ToolParam, ToolsBetaMessageParam
from dotenv import load_dotenv

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = "claude-3-sonnet-20240229"

claude_client = Anthropic(
    api_key=CLAUDE_API_KEY, default_headers={"anthropic-beta": "tools-2024-04-04"}
)


@dataclass
class Function:
    """
    Function
    """

    name: str
    parameters: dict[str, Any]
    description: Optional[str] = None


@dataclass
class Message:
    """
    Message
    """

    role: str
    content: str


@dataclass
class OpenAIRequest:
    """
    OpenAI request
    """

    messages: list[Message]
    functions: Optional[list[Function]] = None


def try_get(
    getter: Callable[[], Optional[Any]], default: Optional[Any] = None
) -> Optional[Any]:
    """
    Execute getter, return default value if it fails
    """
    try:
        return getter()
    except Exception:  # pylint: disable=broad-exception-caught
        return default


def openai_to_claude(request: OpenAIRequest):
    """
    Convert Merlin's OpenAI request to Claude request
    """
    is_enum = not request.functions
    if is_enum:
        function = Function(
            name="format_response",
            parameters={
                "type": "object",
                "required": [
                    "category",
                ],
                "properties": {
                    "category": {"type": "integer"},
                },
            },
            description="Format response wrt identified category.",
        )
    else:
        function = try_get(lambda: request.functions[0])
    system = try_get(
        lambda: [message for message in request.messages if message.role == "system"][
            0
        ].content,
        "",
    )
    tools: list[ToolParam] = [
        {
            "name": try_get(lambda: function.name, ""),
            "description": try_get(lambda: function.description, ""),
            "input_schema": {
                "type": "object",
                "properties": try_get(lambda: function.parameters["properties"], {}),
                "required": try_get(lambda: function.parameters["required"], []),
            },
        }
    ]
    messages: list[ToolsBetaMessageParam] = [
        dataclasses.asdict(message) for message in request.messages if message.role != "system"
    ]
    args = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    response = claude_client.beta.tools.messages.create(**args)
    openai_response = {
        "id": response.id,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens
            + response.usage.output_tokens,
        },
        "object": "chat.completion",
        "created": int(time.time()),
    }
    tool = next(c for c in response.content if c.type == "tool_use")
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
    return openai_response


class MerlinChatCompletion(openai.ChatCompletion):
    @classmethod
    def create(cls, *args, **kwargs):
        print(kwargs)
        return super().create(*args, **kwargs)

    @classmethod
    async def acreate(cls, *args, **kwargs):
        print(kwargs)
        return await super().acreate(*args, **kwargs)
