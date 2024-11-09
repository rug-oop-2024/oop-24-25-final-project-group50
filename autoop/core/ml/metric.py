from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "Mean Squared Error",
    "Accuracy",
    "Mean Absolute Percentage Error",
    "Cohens Kappa",
    "R-squared score",
    "Precision"
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> "Metric":
    match name:
        case "Mean Squared Error":
            return MeanSquaredError()
        case "Accuracy":
            return Accuracy()
        case "log_loss":
            return LogLoss()
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

    # def predictions_for_regression(observations, model_parameters):
    #     pass

    # def predictions_for_classification(observations, model_parameters):
    #     pass

    @abstractmethod
    def evaluate(self, predicted_truths: np.ndarray,
                 actual_truths: np.ndarray) -> float:
        pass


class Accuracy(Metric):
    """Class for the calculation of accuracy"""
    def evaluate(self, predicted_truth, actual_truth) -> float:
        """
        The metric function to calculate the accuracy.

        Args:
        predicted_truth: the ground truth predicted by the model
        actual_truth: the ground truth given in the database

        Returns:
            The percentage of matching predicted and actual truths
        """
        total_result = np.sum((predicted_truth == actual_truth).astype(int))
        return 1/len(actual_truth) * total_result


class MeanSquaredError(Metric):
    """Class for the calculation of the mean squared error"""
    def evaluate(self, predicted_truth, actual_truth) -> float:
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
        return 1/len(actual_truth) * total_result


class LogLoss(Metric):
    """Class for the calculation of the logloss"""
    def metric_function(self, predicted_truths: np.ndarray,
                        actual_truths: np.ndarray) -> float:
        """
        The code assumes that predicted_truths consists of an array with \
              chances which are added up together 1
        E.g. [0.1, 0.2, 0.6, 0.1]
        The code assumes that actual_truths consists of an array \
              with a 0 or 1, indicating the right classification.
        E.g. [0, 0, 1, 0]
        """
        multiplied_arrays = np.multiply(predicted_truths, actual_truths)
        filtered_array = multiplied_arrays[multiplied_arrays != 0]
        log_list = -np.log(filtered_array)
        return 1/len(log_list) * np.sum(log_list)


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
        sum_part = np.sum(abs(actual_truths - predicted_truths)
                          / actual_truths)
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
        print(actual_truth)
        if predicted_truth.ndim > 1:  # Check if prediction is one-hot encoded
            predicted_truth = np.argmax(predicted_truth,
                                        axis=1)  # Flatten to single labels
        if actual_truth.ndim > 1:  # Check if actual truth is one-hot encoded
            actual_truth = np.argmax(actual_truth,
                                     axis=1)  # Flatten to single labels

        unique_labels = np.unique(np.concatenate((actual_truth,
                                                  predicted_truth)))
        label_num = len(unique_labels)

        index_map = {label: index for index, label in enumerate(unique_labels)}

        confusion_matrix = np.zeros((label_num, label_num), dtype=int)

        for truth, prediction in zip(actual_truth, predicted_truth):
            # truth, prediction = int(truth), int(prediction)
            confusion_matrix[index_map[truth],
                             index_map[prediction]] += 1

        num_samples = np.sum(confusion_matrix)
        observed_agreement = np.trace(confusion_matrix) / num_samples

        row_sum = np.sum(confusion_matrix, axis=1)
        column_sum = np.sum(confusion_matrix, axis=0)
        expected_agreement = np.sum((row_sum * column_sum) / num_samples**2)
        print(observed_agreement, expected_agreement)

        return (
            (observed_agreement - expected_agreement) /
            (1 - expected_agreement)
            )


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
                 actual_truth: np.ndarray) -> dict:
        """
        Metric function that calculates the precision per category given.

        Args:
            predicted_truth: the predicted truths made by the model
            actual_truth: the ground truth given in the database
        Returns:
            A dictionary with the precision of each category.
        """
        # Convert one-hot encoded arrays to single labels
        if predicted_truth.ndim > 1:  # Check if prediction is one-hot encoded
            predicted_truth = np.argmax(predicted_truth,
                                        axis=1)  # Flatten to single labels
        if actual_truth.ndim > 1:  # Check if actual truth is one-hot encoded
            actual_truth = np.argmax(actual_truth,
                                     axis=1)  # Flatten to single labels

        precision_dict = []
        unique_labels = np.unique(actual_truth)
        for label in unique_labels:
            true_positives = np.sum(np.logical_and(predicted_truth == label,
                                                   actual_truth == label))
            true_and_false_positives = np.sum(predicted_truth == label)
            precision_dict.append(true_positives / true_and_false_positives)
        return np.mean(precision_dict)
