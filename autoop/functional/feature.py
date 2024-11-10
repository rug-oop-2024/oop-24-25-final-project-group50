
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from pandas.api.types import is_numeric_dtype, is_object_dtype


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data = dataset.read()
    feature_list = []
    for feature in data.columns:
        if is_object_dtype(data[feature]):
            feature_type = "categorical"
        elif is_numeric_dtype(data[feature]):
            feature_type = "numerical"
        else:
            raise TypeError(f"{feature} has unsupported types")
        feature_list.append(Feature(name=feature, type=feature_type))
    return feature_list
