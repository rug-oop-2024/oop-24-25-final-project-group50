import pandas as pd

from typing import Any
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression.lasso import LassoCV
from autoop.core.ml.model.regression.elastic_net import ElasticNet
from autoop.core.ml.model.classification.k_nearest_neighbours import KNearestNeighbors
from autoop.core.ml.model.classification.multi_layer_perceptron import MultiLayerPerceptron
from autoop.core.ml.model.classification.random_forest_classifier import RandomForestClassifier
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact

from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric


metric_names = ["R-squared score", "Mean Squared Error"]
metrics = [get_metric(name) for name in metric_names]

metric_data = pd.DataFrame(metric_names)

bytes = metric_data.to_csv(index=False).encode()
print(bytes)