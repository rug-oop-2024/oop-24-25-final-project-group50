from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "Mean Squared Error",
    "Accuracy",
    "Mean Absolute Percentage Error",
    "Cohens Kappa",
    "R-squared score",
    "Precision"
]


def get_metric(name: str) -> "Metric":
    """
    Function to get a metric instance based on the metric name.

    Args:
        name: the name of the metric

    Returns:
        The metric instance corresponding the metric name
    """
    match name:
        case "Mean Squared Error":
            return MeanSquaredError()
        case "Accuracy":
            return Accuracy()
        case "Mean Absolute Percentage Error":
            return MeanAbsolutePercentageError()
        case "Cohens Kappa":
            return CohensKappa()
        case "R-squared score":
            return RSquaredScore()
        case "Precision":
            return Precision()
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self, predicted_truths: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Sets up the data to be calculated in one of the metrics.

        Args:
            model: the model used (or the parameters of the model)
            observations: the observations given by the dataset
            ground_truths: the ground truths corresponding the observations

        Returns:
            a float with the calculation of the chosen metric.
        """
        return self.evaluate(predicted_truths, ground_truths)

    @abstractmethod
    def evaluate(self, predicted_truths: np.ndarray,
                 actual_truths: np.ndarray) -> float:
        """
        Abstract method to evaluate data using metrics.

        Args:
        predicted_truth: the ground truth predicted by the model
        actual_truth: the ground truth given in the database

        Returns:
            A float of the result
        """
        pass


class Accuracy(Metric):
    """Class for the calculation of accuracy"""
    def evaluate(self, predicted_truth: np.ndarray,
                 actual_truth: np.ndarray) -> float:
        """
        The metric function to calculate the accuracy.

        Args:
        predicted_truth: the ground truth predicted by the model
        actual_truth: the ground truth given in the database

        Returns:
            The percentage of matching predicted and actual truths
        """
        total_result = np.sum((predicted_truth == actual_truth).astype(int))
        return 1 / len(actual_truth) * total_result


class MeanSquaredError(Metric):
    """Class for the calculation of the mean squared error"""
    def evaluate(self, predicted_truth: np.ndarray,
                 actual_truth: np.ndarray) -> float:
        """
        The metric function to calculate the mean squared error.

        Args:
        predicted_truth: the ground truth predicted by the model
        actual_truth: the ground truth given in the database

        Returns:
            The average error of the difference between all
            predicted and actual truths
        """
        total_result = np.sum((predicted_truth - actual_truth) ** 2)
        return 1 / len(actual_truth) * total_result


class MeanAbsolutePercentageError(Metric):
    """Class for the calculation of the mean absolute percentage error"""
    def evaluate(self, predicted_truths: np.ndarray,
                 actual_truths: np.ndarray) -> float:
        """
        The metric function to calculate the mean absolute percentage error.

        Args:
            predicted_truth: the predicted truths made by the model
            actual_truth: the ground truth given in the database
        Returns:
            The average degree to which the predicted value differs from the
            actual value expressed in percentages
        """
        sum_part = (np.sum(abs(
            actual_truths - predicted_truths) / actual_truths))
        return sum_part / len(predicted_truths)


class CohensKappa(Metric):
    """Class that computes Cohen's Kappa"""
    def evaluate(self, predicted_truth: np.ndarray,
                 actual_truth: np.ndarray) -> float:
        """
        The metric function to calculate Cohen's Kappa.

        Args:
            predicted_truth: the predicted truths made by the model
            actual_truth: the ground truth given in the database

        Returns:
            The degree to which the model's predictions agree with the true
            values.
            The closer to 1, the better.
        """
        # Convert one-hot encoded arrays to single labels
        if predicted_truth.ndim > 1:
            predicted_truth = np.argmax(predicted_truth, axis=1)
        if actual_truth.ndim > 1:
            actual_truth = np.argmax(actual_truth, axis=1)

        unique_labels = np.unique(np.concatenate((actual_truth,
                                                  predicted_truth)))
        label_num = len(unique_labels)

        index_map = {label: index for index, label in enumerate(unique_labels)}

        confusion_matrix = np.zeros((label_num, label_num), dtype=int)

        for truth, prediction in zip(actual_truth, predicted_truth):
            confusion_matrix[index_map[truth],
                             index_map[prediction]] += 1

        num_samples = np.sum(confusion_matrix)
        obs_agreement = np.trace(confusion_matrix) / num_samples

        row_sum = np.sum(confusion_matrix, axis=1)
        column_sum = np.sum(confusion_matrix, axis=0)
        exp_agreement = np.sum((row_sum * column_sum) / num_samples**2)

        return (obs_agreement - exp_agreement) / (1 - exp_agreement)


class RSquaredScore(Metric):
    """Class which computes the R^2 score."""
    def evaluate(self, predicted_truths: np.ndarray,
                 actual_truths: np.ndarray) -> float:
        """
        Metric function that calculates the R^2 score of a model.

        Args:
            predicted_truth: the predicted truths made by the model
            actual_truth: the ground truth given in the database

        Returns:
            How well the model fits the data.
            The closer to 1, the better.
        """

        actual_truths_mean = np.mean(actual_truths)
        total_sum_squares = np.sum((actual_truths - actual_truths_mean) ** 2)

        residual_sum_squares = np.sum((actual_truths - predicted_truths) ** 2)

        return 1 - (residual_sum_squares / total_sum_squares)


class Precision(Metric):
    """Class which computes the precision"""
    def evaluate(self, predicted_truth: np.ndarray,
                 actual_truth: np.ndarray) -> float:
        """
        Metric function that calculates the macro averaged precision.

        Args:
            predicted_truth: the predicted truths made by the model
            actual_truth: the ground truth given in the database
        Returns:
            The macro averaged precision.
        """
        # Convert one-hot encoded arrays to single labels
        if predicted_truth.ndim > 1:
            predicted_truth = np.argmax(predicted_truth, axis=1)
        if actual_truth.ndim > 1:
            actual_truth = np.argmax(actual_truth, axis=1)

        precision_dict = []
        unique_labels = np.unique(actual_truth)
        for label in unique_labels:
            true_positives = np.sum(np.logical_and(predicted_truth == label,
                                                   actual_truth == label))
            true_and_false_positives = np.sum(predicted_truth == label)
            precision_dict.append(true_positives / true_and_false_positives)
        return np.mean(precision_dict)
