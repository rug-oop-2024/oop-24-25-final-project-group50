import numpy as np
from pydantic import PrivateAttr
from sklearn.linear_model import ElasticNet as En

from autoop.core.ml.model import Model


class ElasticNet(Model):
    """A wrapper class for sklearn's Elastic Net."""

    _en: En = PrivateAttr(default=None)

    def __init__(self) -> None:
        """
        Class Constructor that creates an instance of Lasso.

        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self._en = En()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model based on observations and truths.

        Args:
            observations: an n x m matrix with n observations over m variables
            ground_truth: a vector with n responses
        Returns:
            None
        """
        self._en.fit(observations, ground_truth)
        self._parameters["coef"] = self._en.coef_
        self._parameters["intercept"] = self._en.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict responses based on given observations.

        Args:
            observations: an n x m matrix with n observations over m variables
            to base the prediction on
        Returns:
            An array containing n predictions based on the fitted data
        """
        return self._en.predict(observations)
