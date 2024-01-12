"""
Merlin AI prompts
"""
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from merlin_ai.types import DocEnum


class OpenAIPrompts:
    @staticmethod
    def create_parser_prompt(data_class, value: str, function_name: Optional[str] = "format_response",
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
    def create_classifier_prompt(data_class, enum_options: list, value: str,
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
