"""
Implementation of AI classes
"""
import dataclasses
import datetime
import json
import logging
from enum import Enum
from typing import Type, Optional

import tiktoken

from merlin_ai.data_classes import NativeDataClass
from merlin_ai.llm_classes import PromptBase, OpenAIPrompt
from merlin_ai.settings import default_model_settings


class BaseAIClass:
    """
    Base class for AI classes
    """

    def __init__(self, data_class: Type, model_settings: Optional[dict] = None):
        self._data_class = data_class
        self._model_settings = model_settings

    def as_prompt(
        self,
        value: str,
        model_settings: Optional[dict] = None,
        instruction: Optional[str] = None,
    ) -> dict:
        """
        Get prompt which would be used when evaluating model.
        """
        llm_settings = self._get_llm_settings(model_settings)
        prompt_dict = self._generate_prompt(
            value, model_settings=llm_settings, instruction=instruction
        ).as_dict()

        if "api_key" in prompt_dict:
            del prompt_dict["api_key"]

        return prompt_dict

    def _generate_prompt(
        self, value: str, model_settings: dict, instruction: Optional[str] = None
    ) -> PromptBase:
        """
        Generate LLM prompt from provided value
        :param value:
        :return:
        """
        raise NotImplementedError()

    def _get_llm_settings(
        self, function_call_model_settings: Optional[dict] = None
    ) -> dict:
        """
        Get LLM settings
        """
        settings = default_model_settings
        if self._model_settings:
            settings.update(self._model_settings)
        if function_call_model_settings:
            settings.update(function_call_model_settings)

        return {key: value for key, value in settings.items() if value is not None}

    def _create_instance_from_response(self, llm_response):
        """
        Create base class instance from LLM response
        """
        raise NotImplementedError()

    def __call__(
        self,
        value: str,
        model_settings: Optional[dict] = None,
        instruction: Optional[str] = None,
        return_raw_response: bool = False,
    ):
        """
        Call LLM model, parse the response and instantiate data class
        :param value: user provided string from which to instantiate data class
        :param model_settings: settings for LLM model
        :param instruction: custom instruction to use when generating LLM prompt
        :param return_raw_response: if True raw response from LLM API will be returned alongside the AIModel instance
        :return: instance of data class
        """
        llm_settings = self._get_llm_settings(model_settings)

        prompt = self._generate_prompt(
            value, model_settings=llm_settings, instruction=instruction
        )
        response = prompt.get_llm_response()

        instance = self._create_instance_from_response(response)
        if return_raw_response:
            return instance, response

        return instance


class OpenAIModel(BaseAIClass):
    """
    OpenAI-based AI Model
    """

    def __str__(self):
        return f"OpenAIModel: {self._data_class.__name__}"

    @staticmethod
    def _generate_function_call_object(data_class: Type) -> dict:
        """
        Generate function call object from data class
        """
        if dataclasses.is_dataclass(data_class):
            return NativeDataClass(data_class).generate_format_function_call_object()

        raise ValueError(f"Unsupported data class: {data_class}")

    def _convert_date_related_fields(self, arguments: dict) -> dict:
        """
        Convert fields that are date, datetime, time or timedelta to native python values
        """
        function_call_object = self._generate_function_call_object(self._data_class)

        conversion_functions = {
            "date": datetime.date.fromisoformat,
            "date-time": datetime.datetime.fromisoformat,
            "time": datetime.time.fromisoformat,
        }

        for parameter_name, parameter_details in (
            function_call_object["parameters"].get("properties", {}).items()
        ):
            parameter_value = arguments[parameter_name]
            parameter_format = parameter_details.get("format")
            if not parameter_format or not parameter_value:
                continue

            for format_name, conversion_function in conversion_functions.items():
                if parameter_format == format_name:
                    try:
                        arguments[parameter_name] = conversion_function(parameter_value)
                    except ValueError:
                        logging.error(
                            f"Found invalid {format_name} value: {parameter_value}"
                        )

        return arguments

    def _create_instance_from_response(self, llm_response):
        function_call_response = llm_response.choices[0].message.function_call
        function_name = "format_response"
        if not function_call_response or function_call_response.name != function_name:
            raise RuntimeError(f"LLM did not call function '{function_name}'")

        arguments = json.loads(function_call_response.arguments)
        arguments = self._convert_date_related_fields(arguments)

        return self._data_class(**arguments)

    def _generate_prompt(
        self,
        value: str,
        model_settings: Optional[dict] = None,
        instruction: Optional[str] = None,
    ) -> PromptBase:
        function_name = "format_response"
        model_settings["functions"] = [
            self._generate_function_call_object(self._data_class)
        ]
        model_settings["function_call"] = {"name": function_name}
        if not instruction:
            instruction = self._data_class.__doc__.strip()

        system_instruction = ""
        if not instruction.startswith(f"{self._data_class.__name__}("):
            system_instruction = f"Also note that: {instruction}\n\n"

        return OpenAIPrompt(
            model_settings,
            messages=[
                {
                    "role": "system",
                    "content": "The user will provide text that you need to parse into a structured form.\n"
                    f"To validate your response, you must call the `{function_name}` function.\n"
                    "Use the provided text and context to extract, deduce, or infer\n"
                    f"any parameters needed by `{function_name}`, including any missing data.\n\n{system_instruction}"
                    "You have been provided the following context to perform your task:\n"
                    f"    - The current time is {datetime.datetime.now()}.",
                },
                {"role": "user", "content": f"The text to parse:\n{value}"},
            ],
        )


class OpenAIEnum(BaseAIClass):
    """
    OpenAI-based AI Enum
    """

    def __str__(self):
        return f"OpenAIEnum: {self._data_class.__name__}"

    def __getattr__(self, item):
        if hasattr(self._data_class, item):
            return getattr(self._data_class, item)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{item}'"
        )

    def _create_instance_from_response(self, llm_response):
        content = llm_response.choices[0].message.content
        enum_options = self._get_enum_options()

        try:
            enum_option_index = int(content)
            if enum_option_index < 1 or enum_option_index > len(enum_options):
                raise ValueError()
        except (TypeError, ValueError) as err:
            raise RuntimeError(f"LLM returned invalid value {content}") from err

        return enum_options[int(content) - 1]

    def _get_llm_settings(
        self, function_call_model_settings: Optional[dict] = None
    ) -> dict:
        """
        Get LLM settings
        """
        settings = super()._get_llm_settings(function_call_model_settings)
        settings["max_tokens"] = 1
        return settings

    def _get_enum_options(self) -> list:
        """
        Get enum options.
        """
        if not issubclass(self._data_class, Enum):
            raise ValueError(f"{self._data_class} is not an Enum")

        return list(self._data_class)

    def _generate_prompt(
        self, value: str, model_settings: dict, instruction: Optional[str] = None
    ) -> PromptBase:
        enum_options = self._get_enum_options()
        encoding = tiktoken.get_encoding("cl100k_base")
        model_settings["logit_bias"] = {
            str(encoding.encode(str(idx))[0]): 100
            for idx in range(1, len(enum_options) + 1)
        }
        system_prompt = "You are an expert classifier that always chooses correctly\n\n"

        if not instruction:
            instruction = self._data_class.__doc__.strip()
        if instruction != "An enumeration.":
            system_prompt += f"Also note that:\n{instruction}\n\n"

        system_prompt += (
            "The user will provide text to classify, you will use your expertise "
            "to choose the best option below based on it:\n"
            + "\n".join(
                [
                    f"\t{idx+1}. {option.name} ({idx+1})"
                    for idx, option in enumerate(enum_options)
                ]
            )
        )

        return OpenAIPrompt(
            model_settings,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": f"The text to classify:\n{value}"},
            ],
        )
