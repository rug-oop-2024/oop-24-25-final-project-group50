from collections import Counter

import numpy as np
from pydantic import Field

from autoop.core.ml.model import Model


class KNearestNeighbors(Model):
    """A class that uses close neighbors to predict outcomes."""

    k: int = Field(gt=0, default=3)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on observations and ground truths.

        Args:
            observations: an n x m matrix with n observations over m variables
            ground_truth: a vector with n responses
        Returns:
            None
        """
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth,
        }

    def predict(self, observations: np.ndarray) -> list[str]:
        """
        Predict responses based on given observations.

        Args:
            observations: an n x m matrix with n observations over m variables
            to base the prediction on
        Returns:
            A list containing n predictions based on the fitted data
        """
        return [self._predict_single(x) for x in observations]

    def _predict_single(self, observation: np.ndarray) -> str:
        """
        Predict a single variable using the corresponding observations.

        Args:
            observation: an array of observations to base the prediction on
        Returns:
            The prediction as a string
        """
        distances = np.linalg.norm(
            self._parameters["observations"] - observation, axis=1
        )
        k_idx = np.argsort(distances)[: self.k]
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in k_idx]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
