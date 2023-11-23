"""
Implementation of model config/init classes
"""
import dataclasses
from typing import Type, Optional, Any
from merlin_ai.data_classes import NativeDataClass


class ModelConfigBase:
    def __init__(self, model_settings: dict):
        self.model_settings = model_settings

    def generate_function_call_object(self) -> dict:
        raise NotImplementedError

    def get_instruction(self) -> Optional[str]:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def get_enum_options(self) -> list[str]:
        raise NotImplementedError


class ModelFromParams(ModelConfigBase):
    def __init__(self, model_params: dict, model_settings: dict):
        super().__init__(model_settings)
        self.model_params = model_params

    def generate_function_call_object(self) -> dict:
        properties = self.model_params.get("properties")
        if properties:
            function_call = {
                "name": "format_response",
                "parameters": {
                    "properties": properties,
                    "required": list(properties.keys()),
                    "type": "object"
                }
            }
            desc = self.model_params.get("description")
            if desc:
                function_call["description"] = desc
            return function_call
        raise ValueError("Properties are not specified for this model.")

    def get_instruction(self) -> Optional[str]:
        return self.model_params.get("instruction")

    def get_name(self) -> str:
        name = self.model_params.get("name")
        if name:
            return name
        raise ValueError("Name was not specified for this model.")

    def get_enum_options(self) -> list[str]:
        classes = self.model_params.get("classes")
        if classes:
            return classes
        raise ValueError("Classes were not specified for this model.")


class ModelFromDataClass(ModelConfigBase):
    def __init__(self, data_class: Type, model_settings: dict):
        super().__init__(model_settings)
        self.data_class = data_class

    def generate_function_call_object(self) -> dict:
        if dataclasses.is_dataclass(self.data_class):
            return NativeDataClass(self.data_class).generate_format_function_call_object()
        raise ValueError(f"Unsupported data class: {self.data_class}")

    def get_instruction(self) -> Optional[str]:
        instruction = self.data_class.__doc__.strip()
        return instruction if instruction != "An enumeration." else None

    def get_name(self) -> str:
        return self.data_class.__name__
