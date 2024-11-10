
from abc import ABC, abstractmethod
from copy import deepcopy
from autoop.core.ml.artifact import Artifact

import numpy as np
from pydantic import BaseModel, PrivateAttr
import pickle


class Model(ABC, BaseModel):
    """An abstract base class for a generic Machine Learning model."""

    _parameters: dict = PrivateAttr(default={})
    _type: str = PrivateAttr(default=None)

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

    @property
    def type(self) -> str:
        """
        Get the type of the model using the getter.

        Args:
            None
        Returns:
            The model type
        """
        return self._type

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

    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model instance to an artifact instance.

        Args:
            name: the name which is to be given to the artifact instance
        Returns:
            An Artifact instance
        """
        return Artifact(name=name, type="model",
                        data=pickle.dumps(self._parameters))
