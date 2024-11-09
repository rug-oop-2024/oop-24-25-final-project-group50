from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression import MultipleLinearRegression, ElasticNet
from autoop.core.ml.metric import MeanSquaredError, RSquaredScore, MeanAbsolutePercentageError
from sklearn.linear_model import LinearRegression, Lasso


class TestPipelineForMe():

    def setUp(self) -> None:
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(),
            input_features=list(filter(lambda x: x.name != "age", self.features)),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError(), MeanAbsolutePercentageError(), RSquaredScore()],
            split=0.8
        )

new_test = TestPipelineForMe()
new_test.setUp()
print(new_test.pipeline.execute())
