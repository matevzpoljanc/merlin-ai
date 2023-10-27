"""
Supported data classes as a base of AI Models
"""
import dataclasses
import typing
from typing import Type


class BaseSupportedDataClass:
    """
    Base class
    """

    def __init__(self, data_type):
        self.data_type = data_type

    def generate_format_function_call_object(self) -> dict:
        """
        Generate function call object for OpenAI
        """
        raise NotImplementedError()


class NativeDataClass(BaseSupportedDataClass):
    """
    Support for python's native Data
    """

    def __init__(self, data_type):
        super().__init__(data_type)
        if not dataclasses.is_dataclass(data_type):
            raise ValueError(f"{data_type} is not dataclass")

    @classmethod
    def _get_type_args(cls, param_type: Type) -> dict:
        """
        Parameter type to args
        """
        if param_type == str:
            return {"type": "string"}

        if typing.get_origin(param_type) == list:
            return {
                "type": "array",
                "items": cls._get_type_args(typing.get_args(param_type)[0])
                or {"type": "string"},
            }
        if typing.get_origin(param_type) == typing.Literal:
            options = typing.get_args(param_type)
            type_args = cls._get_type_args(type(options[0]))
            type_args["enum"] = list(options)
            return type_args

        return {}

    @classmethod
    def _generate_parameters(cls, data_class) -> dict:
        """
        Generate parameters argument
        """
        properties = {}

        for field in dataclasses.fields(data_class):
            field_property = {"title": field.name.replace("_", " ").capitalize()}
            field_property.update(cls._get_type_args(field.type))

            description = field.metadata.get("description")
            if description:
                field_property["description"] = description

            properties[field.name] = field_property

        return {
            "properties": properties,
            "required": list(properties.keys()),
            "type": "object",
        }

    def generate_format_function_call_object(self):
        function_call_obj = {
            "name": "format_response",
            "parameters": self._generate_parameters(self.data_type),
        }

        function_description: str = self.data_type.__doc__
        if function_description and not function_description.startswith(
            f"{self.data_type.__name__}("
        ):
            function_call_obj["description"] = function_description.strip()

        return function_call_obj
