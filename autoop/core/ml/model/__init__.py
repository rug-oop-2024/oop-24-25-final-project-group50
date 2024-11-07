
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.elastic_net import ElasticNet
from autoop.core.ml.model.regression.lasso import LassoCV
from autoop.core.ml.model.classification.random_forest_classifier import RandomForestClassifier
from autoop.core.ml.model.classification.k_nearest_neighbours import KNearestNeighbors
from autoop.core.ml.model.classification.multi_layer_perceptron import MultiLayerPerceptron

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "ElasticNet",
    "Lasso"
]

CLASSIFICATION_MODELS = [
    "kNearestNeighbors",
    "RandomForestClassifier",
    "MultiLayerPerceptron"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    match model_name:
        case "MultipleLinearRegression":
            return MultipleLinearRegression()
        case "ElasticNet":
            return ElasticNet()
        case "Lasso":
            return Lasso()
        case "kNearestNeighbors":
            return KNearestNeighbors()
        case "RandomForestClassifier":
            return RandomForestClassifier()
        case "MultiLayerPerceptron":
            return MultiLayerPerceptron()
