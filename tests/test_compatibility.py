"""Exercise Merlin's public behavior without calling an LLM provider."""

import datetime
import json
import unittest
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import List, Literal, Optional
from unittest.mock import patch

import tiktoken

from merlin_ai import ai_enum, ai_enum_model, ai_model, default_model_settings
from merlin_ai.ai_classes import OpenAIEnum, OpenAIEnumModel, OpenAIModel
from merlin_ai.data_classes import NativeDataClass
from merlin_ai.llm_classes import OpenAIPrompt


class Category(Enum):
    """Classify a sales conversation."""

    SALES = "sales"
    SUPPORT = "support"


@dataclass
class Contact:
    """Extract contact details."""

    name: str
    birthday: datetime.date
    category: Category
    nickname: Optional[str]
    tags: List[str]
    status: Literal["new", "existing"]


@dataclass
class Appointment:
    """Extract an appointment."""

    name: str
    day: datetime.date
    start: datetime.time
    created_at: datetime.datetime


def function_response(arguments, name="format_response"):
    """Build the function-call response consumed by Merlin."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    function_call=SimpleNamespace(
                        name=name, arguments=json.dumps(arguments)
                    )
                )
            )
        ]
    )


class CompatibilityTests(unittest.TestCase):
    """Cover annotations, native tokenization and mocked provider responses."""

    def setUp(self):
        # Model settings are shared by Merlin; restore them after each test.
        self.settings = default_model_settings.copy()
        self.addCleanup(self.restore_settings)

    def restore_settings(self):
        """Restore global settings after model construction mutates them."""
        default_model_settings.clear()
        default_model_settings.update(self.settings)

    def test_dataclass_schema(self):
        """Dataclass annotations retain their meaning across Python versions."""
        schema = NativeDataClass(Contact).generate_format_function_call_object()
        properties = schema["parameters"]["properties"]
        self.assertEqual(schema["name"], "format_response")
        self.assertEqual(properties["name"]["type"], "string")
        self.assertEqual(properties["birthday"]["format"], "date")
        self.assertEqual(properties["category"]["enum"], ["SALES", "SUPPORT"])
        self.assertEqual(properties["nickname"]["type"], ["string", "null"])
        self.assertEqual(properties["tags"]["items"], {"type": "string"})
        self.assertEqual(properties["status"]["enum"], ["new", "existing"])
        self.assertEqual(schema["parameters"]["required"], list(properties))

    def test_enum_model_dynamic_dataclass(self):
        """Generated enum wrapper dataclasses produce a usable schema."""
        prompt = OpenAIEnumModel(Category).as_prompt("A pricing question")
        properties = prompt["tools"][0]["parameters"]["properties"]
        self.assertEqual(properties["category"]["enum"], ["SALES", "SUPPORT"])
        self.assertEqual(properties["explanation"]["type"], "string")

    def test_tiktoken_and_enum_prompt(self):
        """Run the native tokenizer used to build classification logit biases."""
        encoding = tiktoken.get_encoding("cl100k_base")
        text = "Merlin: Živjo, Python 3.14!"
        self.assertEqual(encoding.decode(encoding.encode(text)), text)
        prompt = OpenAIEnum(Category).as_prompt("A pricing question")
        self.assertEqual(
            prompt["logit_bias"],
            {str(encoding.encode("1")[0]): 100, str(encoding.encode("2")[0]): 100},
        )
        self.assertEqual(prompt["max_tokens"], 1)

    def test_model_response_dates(self):
        """Function-call JSON is converted back to native datetime values."""
        result = OpenAIModel(Appointment).create_instance_from_response(
            function_response(
                {
                    "name": "Demo",
                    "day": "2026-10-26",
                    "start": "14:30:00",
                    "created_at": "2026-10-26T12:30:00",
                }
            )
        )
        self.assertEqual(result.name, "Demo")
        self.assertEqual(result.day, datetime.date(2026, 10, 26))
        self.assertEqual(result.start, datetime.time(14, 30))
        self.assertEqual(result.created_at, datetime.datetime(2026, 10, 26, 12, 30))

    def test_wrong_function_response(self):
        """Unexpected function calls remain rejected."""
        with self.assertRaisesRegex(RuntimeError, "format_response"):
            OpenAIModel(Appointment).create_instance_from_response(
                function_response({}, name="unexpected")
            )

    def test_enum_response(self):
        """Numeric classifier responses map to the original enum members."""
        model = OpenAIEnum(Category)
        for content, expected in [("1", Category.SALES), ("2", Category.SUPPORT)]:
            with self.subTest(content=content):
                response = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
                self.assertIs(model.create_instance_from_response(response), expected)
        self.assertIs(model.from_json("support"), Category.SUPPORT)

    def test_invalid_enum_response(self):
        """Invalid model choices do not silently select an enum member."""
        model = OpenAIEnum(Category)
        for content in ["0", "3", "unknown", None]:
            with self.subTest(content=content):
                response = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
                with self.assertRaises(RuntimeError):
                    model.create_instance_from_response(response)

    def test_enum_model_response(self):
        """Structured classification preserves its explanation."""
        result = OpenAIEnumModel(Category).create_instance_from_response(
            function_response({"category": "SALES", "explanation": "Pricing question"})
        )
        self.assertIs(result.category, Category.SALES)
        self.assertEqual(result.explanation, "Pricing question")

    def test_public_decorators(self):
        """The public decorators still construct their corresponding models."""
        self.assertIsInstance(ai_model(Appointment), OpenAIModel)
        self.assertIsInstance(ai_enum(Category), OpenAIEnum)
        self.assertIsInstance(ai_enum_model(Category), OpenAIEnumModel)
        self.assertIsInstance(ai_model(model="gpt-4o-mini")(Appointment), OpenAIModel)

    def test_completion_dispatch(self):
        """Provider settings and messages reach LiteLLM without a live request."""
        prompt = OpenAIPrompt({"model": "gpt-4o-mini"})
        prompt.add_message("user", "Hello")
        with patch("merlin_ai.llm_classes.completion", return_value="response") as call:
            self.assertEqual(prompt.get_llm_response(), "response")
        call.assert_called_once_with(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}]
        )

    def test_prompt_hides_api_key(self):
        """Prompt inspection does not expose the configured API key."""
        prompt = OpenAIModel(Appointment, {"api_key": "test-only"}).as_prompt("Demo")
        self.assertNotIn("api_key", prompt)


if __name__ == "__main__":
    unittest.main()
