"""
Merlin AI types
"""
from enum import Enum


class DocEnum(Enum):
    def __new__(cls, value, doc=""):
        self = object.__new__(cls)  # calling super().__new__(value) here would fail
        self._value_ = value
        self.__doc__ = doc
        return self
