
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """A description"""
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __str__(self):
        raise NotImplementedError("To be implemented.")
