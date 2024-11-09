from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import MultipleLinearRegression, LassoCV, ElasticNet, KNearestNeighbors, RandomForestClassifier, MultiLayerPerceptron
from sklearn.linear_model import LinearRegression
from autoop.core.ml.metric import get_metric
from sklearn.datasets import fetch_openml, load_iris
import pandas as pd

METRICS = [
    "mean_squared_error",
    "accuracy",
    "log_loss",
    "mean_absolute_percentage_error",
    "cohens_kappa",
    "r_squared_score",
]

regression_metrics = [get_metric("mean_squared_error"), 
                      get_metric("mean_absolute_percentage_error"),
                      get_metric("r_squared_score"), 
                      ]

classification_metrics = [get_metric("accuracy"), 
                      get_metric("cohens_kappa"),
                      get_metric("precision"), 

]

# data = pd.read_csv('advertising.csv')
# df = pd.DataFrame(
#     data,
#     columns=data.columns,
# )
# dataset = Dataset.from_dataframe(
#     name="adult",
#     asset_path="adult.csv",
#     data=data,
# )

data = fetch_openml(name="adult", version=1, parser="auto")
df = pd.DataFrame(
    data.data,
    columns=data.feature_names,
)
dataset = Dataset.from_dataframe(
    name="adult",
    asset_path="adult.csv",
    data=df,
)

features = detect_feature_types(dataset=dataset)

new_pipeline = Pipeline(
    metrics=classification_metrics,
    dataset=dataset,
    model=MultiLayerPerceptron(),
    input_features=list(filter(lambda x: x.name != "education", features)),
    target_feature=Feature(name="education", type="categorical"),
            
)

print(new_pipeline.execute())
# print("predictions:", new_pipeline._predictions)
# print("truths:", new_pipeline._test_y)
print(new_pipeline._predictions.shape, new_pipeline._test_y.shape)
