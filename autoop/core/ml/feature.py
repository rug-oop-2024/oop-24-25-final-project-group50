
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """A description"""
    name: str = Field(default="")
    type: str = Field(default="")

    def __str__(self):
        return f"Name: {self.name}, Type: {self.type}"
