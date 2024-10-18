from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> 'Metric':
    match name:
        case "mean_squared_error":
            return MeanSquaredError()
        case "accuracy":
            return Accuracy()
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.


class Metric(ABC):
    """Base class for all metrics."""
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

    def __call__(self, model, observations, ground_truths) -> float:
        """
        Sets up the data to be calculated in one of the metrics.

        Args:
            model: the model used (or the parameters of the model)
            observations: the observations given by the dataset
            ground_truths: the ground truths corresponding the observations

        Returns:
            a float with the calculation of the chosen metric.
        """
        np_observations = np.asarray(observations)
        predicted_ground_truths = np_observations @ model.parameters  # geen idee hoe ik de parameters moet callen
        return self.metric_function(predicted_ground_truths, ground_truths)

    @abstractmethod
    def metric_function(predicted_truths, actual_truths) -> float:
        pass
# add here concrete implementations of the Metric class


class Accuracy(Metric):
    """Class for the calculation of accuracy"""
    def metric_function(predicted_truth, actual_truth) -> float:
        """
        The metric function to calculate the accuracy.

        Args:
        predicted_truth: the ground truth predicted by the model
        actual_truth: the ground truth given in the database

        Returns:
            The percentage of matching predicted and actual truths
        """
        total_result = np.sum((predicted_truth == actual_truth).astype(int))
        return 1/len(total_result) * total_result


class MeanSquaredError(Metric):
    """Class for the calculation of the mean squared error"""
    def metric_function(predicted_truth, actual_truth) -> float:
        """
        The metric function to calculate the mean squared error.

        Args:
        predicted_truth: the ground truth predicted by the model
        actual_truth: the ground truth given in the database

        Returns:
            The average error of the difference between all
            predicted and actual truths
        """
        subtracted_array = np.subtract(predicted_truth - actual_truth)
        total_result = np.sum(subtracted_array**2)
        return 1/len(total_result) * total_result
