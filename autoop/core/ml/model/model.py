
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model:
    pass  # your code (attribute and methods) here
    

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from pydantic import BaseModel, PrivateAttr


class Model(ABC, BaseModel):
    """An abstract base class for a generic Machine Learning model."""

    _parameters: dict = PrivateAttr(default={})

    @property
    def parameters(self) -> dict:
        """
        Get a copy of the parameters using the getter.

        Args:
            None
        Returns:
            A deepcopy of the parameters
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on observations and ground truths.

        Args:
            observations: an n x m matrix with n observations over m variables
            ground_truth: a vector with n responses
        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict responses based on given observations.

        Args:
            observations: an n x m matrix with n observations over m variables
            to base the prediction on
        Returns:
            An array containing n predictions based on the fitted data
        """
        pass
