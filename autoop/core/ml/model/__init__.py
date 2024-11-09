
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression \
      import MultipleLinearRegression
from autoop.core.ml.model.regression.elastic_net import ElasticNet
from autoop.core.ml.model.regression.lasso import LassoCV
from autoop.core.ml.model.classification.random_forest_classifier \
      import RandomForestClassifier
from autoop.core.ml.model.classification.k_nearest_neighbours \
      import KNearestNeighbors
from autoop.core.ml.model.classification.multi_layer_perceptron \
    import MultiLayerPerceptron

REGRESSION_MODELS = [
    "Multiple Linear Regression",
    "Elastic Net",
    "Lasso"
]

CLASSIFICATION_MODELS = [
    "K Nearest Neighbors",
    "Random Forest Classifier",
    "Multi Layer Perceptron"
]


def get_model(model_name: str) -> Model:
    print(model_name)
    """Factory function to get a model by name."""
    match model_name:
        case "Multiple Linear Regression":
            return MultipleLinearRegression()
        case "Elastic Net":
            return ElasticNet()
        case "Lasso":
            return LassoCV()
        case "K Nearest Neighbors":
            return KNearestNeighbors()
        case "Random Forest Classifier":
            return RandomForestClassifier()
        case "Multi Layer Perceptron":
            return MultiLayerPerceptron()
