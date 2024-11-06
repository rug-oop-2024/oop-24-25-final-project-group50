import streamlit as st
import pandas as pd

from typing import Any
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.elastic_net import ElasticNet
from autoop.core.ml.model.classification.k_nearest_neighbours import KNearestNeighbors
from autoop.core.ml.model.classification.multi_layer_perceptron import MultiLayerPerceptron
from autoop.core.ml.model.classification.random_forest_classifier import RandomForestClassifier

from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric


def get_model(name: str) -> Any:
    match name:
        case "Multiple linear regression":
            model = MultipleLinearRegression()
        case "Lasso":
            model = Lasso()
        case "Elastic Net":
            model = ElasticNet()
        case "K Nearest Neighbors":
            model = KNearestNeighbors()
        case "Multilayer Preceptron":
            model = MultiLayerPerceptron()
        case "Random Forest Classifier":
            model = RandomForestClassifier()
    return model



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
    cur_dataset = datasets[datasets_names.index(name_cur_dataset)]
    
    feature_list = detect_feature_types(cur_dataset)
    print('-'*69)
    print(feature_list)
    print('-'*69)
    input_features = st.multiselect(label="Choose the input features", options=feature_list)

# code to remove selected features from list here

target_feature = st.selectbox(label="Choose the target feature", options=feature_list)


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
    metric_options = []


model_selection = st.selectbox(label="Choose the model you want to use:", options=model_options)

model = get_model(model_selection)

data_split = st.slider(label="Select a data split:", min_value=0.0, max_value=1.0)
selected_metrics = st.multiselect(label="Select the metrics you want to use:", options=metric_options)

# Summary here

# Train button here

# Pipeline results here
