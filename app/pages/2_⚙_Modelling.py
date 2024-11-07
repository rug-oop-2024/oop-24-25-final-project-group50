import streamlit as st
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

from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric


def get_model(name: str) -> Any:
    match name:
        case "Multiple linear regression":
            model = MultipleLinearRegression()
        case "Lasso":
            model = LassoCV()
        case "Elastic Net":
            model = ElasticNet()
        case "K Nearest Neighbors":
            model = KNearestNeighbors()
        case "Multilayer Preceptron":
            model = MultiLayerPerceptron()
        case "Random Forest Classifier":
            model = RandomForestClassifier()
    return model

METRICS = [
    "mean_squared_error",
    "accuracy",
    "log_loss",
    "mean_absolute_percentage_error",
    "cohens_kappa",
    "r_squared_score",
    "precision"
]

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
if datasets != []:
    datasets_names = [dataset.name for dataset in datasets]
    name_cur_dataset = st.selectbox(label="Select the dataset you want to use", options=datasets_names)
    cur_dataset = Dataset.from_artifact(datasets[datasets_names.index(name_cur_dataset)])

    feature_list = detect_feature_types(cur_dataset)
    print('-'*69)
    print(feature_list)
    print('-'*69)
    input_features = st.multiselect(label="Choose the input features", options=feature_list)

# code to remove selected features from list here
target_options = [option for option in feature_list if option not in input_features]
target_feature = st.selectbox(label="Choose the target feature", options=target_options)


if target_feature.type == "categorical":
    task_type = "classification"
    model_options = ["K Nearest Neighbors", "Multilayer Preceptron", "Random Forest Classifier"]
    metric_options = [
        "Mean Squared Error",
        "Accuracy",
        "Log Loss",
        "Mean Absolute Percentage Error",
        "Cohens Kappa",
        "R-squared Score",
        "Precision"
    ]
    
else:
    task_type = "regression"
    model_options = ["Multiple linear regression", "Lasso", "Elastic Net"]
    metric_options = ["Mean Squared Error", "Mean Absolute Percentage Error", "R-squared score"]


model_selection = st.selectbox(label="Choose the model you want to use:", options=model_options)

model = get_model(model_selection)

data_split = st.slider(label="Select a data split:", min_value=0.0, max_value=1.0)
selected_metrics = st.multiselect(label="Select the metrics you want to use:", options=METRICS)

pipeline_metrics = [get_metric(metric) for metric in selected_metrics]

# Summary here

# Train button here
if st.button(label="Train model"):
    # new_pipeline = Pipeline(
    #     metrics=pipeline_metrics,
    #     dataset=cur_dataset,
    #     model=model,
    #     input_features=input_features,
    #     target_feature=target_feature,
    #     split=data_split
    # )
    new_pipeline = Pipeline(
        metrics=[get_metric("r_squared_score")],
        dataset=cur_dataset,
        model=MultipleLinearRegression(),
        input_features=feature_list[:-1],
        target_feature=feature_list[-1],
        split=0.8
    )
    st.write(new_pipeline.execute())

# Pipeline results here
