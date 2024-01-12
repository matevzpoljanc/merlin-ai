"""
Merlin AI prompts
"""
from datetime import datetime
from typing import Optional, Type
import dataclasses

import tiktoken

from merlin_ai.data_classes import NativeDataClass
from merlin_ai.types import DocEnum


class OpenAISettings:
    @staticmethod
    def generate_function_call_object(data_class: Type) -> dict:
        """
        Generate function call object from data class
        """
        if dataclasses.is_dataclass(data_class):
            return NativeDataClass(data_class).generate_format_function_call_object()

        raise ValueError(f"Unsupported data class: {data_class}")

    @staticmethod
    def set_parser_settings(data_class: Type, model_settings: dict, function_name: Optional[str] = "format_response") \
            -> dict:
        model_settings["functions"] = [
            OpenAISettings.generate_function_call_object(data_class)
        ]
        model_settings["function_call"] = {"name": function_name}
        return model_settings

    @staticmethod
    def set_classifier_settings(enum_options: list, model_settings: dict) -> dict:
        encoding = tiktoken.get_encoding("cl100k_base")
        model_settings["logit_bias"] = {
            str(encoding.encode(str(idx))[0]): 100
            for idx in range(1, len(enum_options) + 1)
        }
        return model_settings


class OpenAIPrompts:
    @staticmethod
    def create_parser_prompt(data_class: Type, value: str, function_name: Optional[str] = "format_response",
                             instruction: Optional[str] = None) -> list[dict[str, str]]:
        if not instruction:
            instruction = data_class.__doc__.strip()
        system_instruction = ""
        if not instruction.startswith(f"{data_class.__name__}("):
            system_instruction = f"Also note that: {instruction}\n\n"
        return [
                    {
                        "role": "system",
                        "content": "The user will provide text that you need to parse into a structured form.\n"
                        f"To validate your response, you must call the `{function_name}` function.\n"
                        "Use the provided text and context to extract, deduce, or infer\n"
                        f"any parameters needed by `{function_name}`, including any missing data.\n\n{system_instruction}"
                        "You have been provided the following context to perform your task:\n"
                        f"    - The current time is {datetime.now()}.",
                    },
                    {
                        "role": "user",
                        "content": f"The text to parse:\n{value}"
                    }
                ]

    @staticmethod
    def create_classifier_prompt(data_class: Type, enum_options: list, value: str,
                                 instruction: Optional[str] = None) -> list[dict[str, str]]:
        system_prompt = "You are an expert classifier that always chooses correctly.\n\n"
        if not instruction:
            instruction = data_class.__doc__.strip()
        if instruction != "An enumeration.":
            system_prompt += f"Also note that:\n{instruction}\n\n"
        system_prompt += (
            "The user will provide text to classify, you will use your expertise "
            "to choose the best option below based on it:\n"
            + "\n".join(
                [
                    f"\t{idx + 1}. {option.name} ({idx + 1})"
                    f"{' - ' + option.__doc__ if isinstance(option, DocEnum) and option.__doc__ else ''}"
                    for idx, option in enumerate(enum_options)
                ]
            )
        )
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"The text to classify:\n{value}"
            }
        ]
