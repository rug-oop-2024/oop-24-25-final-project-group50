import numpy as np

from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    """A class that uses linear regression to predict outcomes."""

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on observations and ground truths.

        Args:
            observations: an n x m matrix with n observations over m variables
            ground_truth: a vector with n responses
        Returns:
            None
        """
        self._parameters["parameters"] = self._compute_parameters(
            observations, ground_truth
        )

    def _compute_parameters(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> np.ndarray:
        """
        Perform mathematical calculations needed to compute the parameters.

        Args:
            observations: an n x m matrix with n observations over m variables
            to base the prediction on
        Returns:
            An array containing the computed parameters
        """
        x_bar = np.c_[observations, np.ones(observations.shape[0], dtype=int)]
        # Create an extra column of n ones

        x_bar_transposed = x_bar.T
        inverse = np.linalg.inv(x_bar_transposed @ x_bar)

        return (inverse @ x_bar_transposed) @ ground_truth

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict responses based on given observations.

        Args:
            observations: an n x m matrix with n observations over m variables
            to base the prediction on
        Returns:
            An array containing n predictions based on the fitted data
        """
        if observations.ndim == 1:
            return np.array([self._predict_single(observations)])

        return np.array([self._predict_single(x) for x in observations])

    def _predict_single(self, observation: np.ndarray) -> float:
        """
        Predict a single variable using the corresponding observations.

        Args:
            observation: an array of observations to base the prediction on
        Returns:
            The prediction as a float
        """
        prediction = 0
        for i in range(len(observation)):
            prediction += observation[i] * self._parameters["parameters"][i]
        return prediction + self._parameters["parameters"][-1]
