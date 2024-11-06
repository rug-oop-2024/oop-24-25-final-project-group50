
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
from io import StringIO

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """

    asset_path = dataset.asset_path
    with open(f"assets/objects/{dataset.asset_path}", 'rb') as file:
        df_data = pd.read_csv(file)
    # data_str = data.decode("utf-8")
    # df_data = pd.read_csv(StringIO(data_str))
    print('-' * 69)
    print(df_data)
    print('-' * 69)
    feature_list = []
    for feature in df_data.columns:
        if is_object_dtype(df_data[feature]):
            feature_type = "categorical"
        elif is_numeric_dtype(df_data[feature]):
            feature_type = "numerical"
        else:
            raise TypeError(f"{feature} has unsupported types")
        feature_list.append(Feature(name=feature, type=feature_type))
    return feature_list
