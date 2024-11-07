from sklearn.neural_network import MLPClassifier as Mlp
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
import numpy as np


class MultiLayerPerceptron(Model):
    """A wrapper class for sklearn's MultiLayerPerceptonClassifier."""

    _mlp: Mlp = PrivateAttr(default=None)

    def __init__(self) -> None:
        """
        Class Constructor that creates an instance of MLP.

        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self._mlp = Mlp()
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model based on observations and truths.

        Args:
            observations: an n x m matrix with n observations over m variables
            ground_truth: a vector with n responses
        Returns:
            None
        """
        self._mlp.fit(observations, ground_truth)
        self._parameters = self._mlp.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict responses based on given observations.

        Args:
            observations: an n x m matrix with n observations over m variables
            to base the prediction on
        Returns:
            An array containing n predictions based on the fitted data
        """
        return self._mlp.predict(observations)
