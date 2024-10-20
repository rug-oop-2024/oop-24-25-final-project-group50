
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """A description"""
    name: str = Field(default="")
    type: str = Field(default="")

    def __str__(self):
        raise NotImplementedError("To be implemented.")
