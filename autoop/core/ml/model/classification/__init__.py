"""
This package contains classes of classification models.
"""

from autoop.core.ml.model.classification.k_nearest_neighbours \
 import KNearestNeighbors
from autoop.core.ml.model.classification.multi_layer_perceptron \
 import MultiLayerPerceptron
from autoop.core.ml.model.classification.random_forest_classifier \
 import RandomForestClassifier

__all__ = ["KNearestNeighbors", "MultiLayerPerceptron",
           "RandomForestClassifier"]
