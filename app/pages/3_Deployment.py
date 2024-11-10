import streamlit as st
import pickle
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model
from autoop.core.ml.artifact import Artifact
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric, Metric
from autoop.core.ml.feature import Feature


def get_all_metrics(metric_name_list: list) -> list['Metric']:
    """
    Function to get all the instances of the metric names into a list.

    Args:
        metric_name_list: list with metric names
    Returns:
        A list metrics instances
    """
    return [get_metric(metric) for metric in metric_name_list]


def artifact_to_pipeline(artifact: Artifact) -> "Pipeline":
    """
    Making a pipeline given an artifact pipeline.

    Args:
        artifact: an artifact with the data of a pipeline instance
    Returns:
        A pipeline instance
    """
    pipeline_dict = pickle.loads(artifact.data)
    return Pipeline(
        model=get_model(pipeline_dict.get("model")),
        input_features=pipeline_dict.get("input_features"),
        target_feature=pipeline_dict.get("target_feature"),
        split=pipeline_dict.get("split"),
        metrics=get_all_metrics(pipeline_dict.get("metrics")),
        dataset=None
    )


def features_in_feature_list(pipeline: Pipeline,
                             feature_list: list[Feature]) -> bool:
    """
    Checks if the target and input features of pipeline are also in the dataset

    Args:
        pipeline: a pipeline instance
        feature_list: list with features of a dataset
    Returns:
        True if all features are in the feature list, else False
    """
    for feature in pipeline._input_features:
        if feature not in feature_list:
            return False
    return pipeline._target_feature in feature_list


automl = AutoMLSystem.get_instance()
saved_pipelines = []

pipeline_list = automl.registry.list(type="pipeline")
names_pipeline_list = [pipeline.name for pipeline in pipeline_list]
name_selected_pipeline = st.selectbox("Choose your saved pipeline",
                                      options=names_pipeline_list)

if name_selected_pipeline:
    selected_pipeline = (pipeline_list[
        names_pipeline_list.index(name_selected_pipeline)])
    pipeline = artifact_to_pipeline(selected_pipeline)
    names_input_features = [feature.name for feature
                            in pipeline._input_features]
    metric_name_list = [metric.__class__.__name__ for
                        metric in pipeline._metrics]
    st.subheader("Pipeline Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""**Input Features**: {', '.join(names_input_features)
                    if pipeline._input_features else 'None'}"""
                    )
        st.markdown(f"**Target**: {pipeline._target_feature.name}")

    with col2:
        st.markdown(f"**Model**: {pipeline.model.__class__.__name__}")
        st.markdown(f"**Data Split**: {pipeline._split}")
        metric_text = ", ".join(metric_name_list) \
            if pipeline._metrics else "None"
        st.markdown(f"**Metrics**: {metric_text}")

    uploaded_datasets = st.file_uploader("Please upload your \
                                         datasets here (.csv)")
    if uploaded_datasets:
        save_button = st.button("Save File")
        if save_button:
            for item in uploaded_datasets:
                if item.name.endswith('.csv'):
                    in_dataframe = pd.read_csv(item)
                    as_dataset = Dataset.from_dataframe(in_dataframe,
                                                        name=item.name,
                                                        asset_path="./assets")
                    as_dataset.asset_path = f"./{as_dataset.id}"
                    automl._registry.register(as_dataset)
                    st.success("File saved")
                    st.write(in_dataframe)
                else:
                    st.warning(f"File {item.name} is not a CSV file \
                               and was not saved")

    datasets = automl.registry.list(type="dataset")
    if datasets != []:
        datasets_names = [dataset.name for dataset in datasets]
        # chosen_dataset = st.markdown(datasets_names)
        name_cur_dataset = st.selectbox('With which file do you want to make \
                                        predictions? The target feature and \
                                        input features must be in the file!',
                                        datasets_names, index=0)
        cur_dataset = Dataset.from_artifact(datasets[
            datasets_names.index(name_cur_dataset)])
        features_cur_file = detect_feature_types(cur_dataset)
        if not features_in_feature_list(pipeline, features_cur_file):
            st.warning(f"File {cur_dataset.name} does not have all the target \
                       or input features! Please choose another file.")
        else:
            if st.button(label="Train model"):
                st.session_state.model_trained = True

                if st.session_state.model_trained:
                    pipeline._dataset = cur_dataset
                    pipeline_results = pipeline.execute()
                    metric_results = pipeline_results.get('metrics')
                    for metric in metric_results:
                        st.write(f"{metric[0].__class__.__name__}: \
                                 {metric[1]:.3f}")
                    st.write(pipeline_results.get('predictions'))
                    st.session_state.save_pipeline_clicked = True
                    st.session_state.pipeline = pipeline
