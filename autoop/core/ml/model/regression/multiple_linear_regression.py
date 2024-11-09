# import numpy as np

# from autoop.core.ml.model import Model


# class MultipleLinearRegression(Model):
#     """A class that uses linear regression to predict outcomes."""

#     def __init__(self) -> None:
#         super().__init__()
#         self._type = "regression"

#     def fit(self, observations: np.ndarray, ground_truth: np.ndarray) ->None:
#         """
#         Train the model based on observations and ground truths.

#         Args:
#             observations:an n x m matrix with n observations over m variables
#             ground_truth: a vector with n responses
#         Returns:
#             None
#         """
#         self._parameters["parameters"] = self._compute_parameters(
#             observations, ground_truth
#         )

#     def _compute_parameters(
#         self, observations: np.ndarray, ground_truth: np.ndarray
#     ) -> np.ndarray:
#         """
#         Perform mathematical calculations needed to compute the parameters.

#         Args:
#             observations:an n x m matrix with n observations over m variables
#             to base the prediction on
#         Returns:
#             An array containing the computed parameters
#         """
#         x_bar =np.c_[observations, np.ones(observations.shape[0], dtype=int)]
#         # Create an extra column of n ones

#         x_bar_transposed = x_bar.T
#         inverse = np.linalg.inv(x_bar_transposed @ x_bar)

#         return (inverse @ x_bar_transposed) @ ground_truth

#     def predict(self, observations: np.ndarray) -> np.ndarray:
#         """
#         Predict responses based on given observations.

#         Args:
#             observations:an n x m matrix with n observations over m variables
#             to base the prediction on
#         Returns:
#             An array containing n predictions based on the fitted data
#         """
#         if observations.ndim == 1:
#             return np.array([self._predict_single(observations)])

#         return np.array([self._predict_single(x) for x in observations])

#     def _predict_single(self, observation: np.ndarray) -> float:
#         """
#         Predict a single variable using the corresponding observations.

#         Args:
#             observation: an array of observations to base the prediction on
#         Returns:
#             The prediction as a float
#         """
#         return sum(np.append(observation*self._parameters["parameters"][:-1],
#                              self._parameters["parameters"][-1]))

import numpy as np
from pydantic import PrivateAttr
from sklearn.linear_model import LinearRegression

from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    """A wrapper class for sklearn's Lasso."""

    _mlr: LinearRegression = PrivateAttr(default=None)

    def __init__(self) -> None:
        """
        Class Constructor that creates an instance of Lasso.

        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self._mlr = LinearRegression()
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
        self._mlr.fit(observations, ground_truth)
        # self._parameters["coef"] = self._ls.coef_
        # self._parameters["intercept"] = self._ls.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict responses based on given observations.

        Args:
            observations: an n x m matrix with n observations over m variables
            to base the prediction on
        Returns:
            An array containing n predictions based on the fitted data
        """
        return self._mlr.predict(observations)
