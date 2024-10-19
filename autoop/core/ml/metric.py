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
    
    def predictions_for_regression(observations, model_parameters):
        pass

    def predictions_for_classification(observations, model_parameters):
        pass

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


class LogLoss(Metric):
    """Class for the calculation of the logloss"""
    def metric_function(predicted_truths: np.ndarray, actual_truths: np.ndarray) -> float:
        """
        The code assumes that predicted_truths consists of an array with chances which are added up together 1
        E.g. [0.1, 0.2, 0.6, 0.1]
        The code assumes that actual_truths consists of an array with a 0 or 1, indicating the right classification.
        E.g. [0, 0, 1, 0]
        """
        multiplied_arrays = np.multiply(predicted_truths, actual_truths)
        filtered_array = multiplied_arrays[multiplied_arrays != 0]
        log_list = -np.log(filtered_array)
        return 1/len(log_list) * np.sum(log_list)
