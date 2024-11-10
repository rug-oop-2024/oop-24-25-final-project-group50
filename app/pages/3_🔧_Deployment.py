import streamlit as st
import pickle
import pandas as pd
import time

from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model
from autoop.core.ml.artifact import Artifact
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric, Metric
from autoop.core.ml.feature import Feature


# ###HELPER FUNCTIONS###

def _get_all_metrics(metric_name_list: list) -> list['Metric']:
    """
    Function to get all the instances of the metric names into a list.

    Args:
        metric_name_list: list with metric names
    Returns:
        A list metrics instances
    """
    return [get_metric(metric) for metric in metric_name_list]


def _artifact_to_pipeline(artifact: Artifact) -> "Pipeline":
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
        metrics=_get_all_metrics(pipeline_dict.get("metrics")),
        dataset=None
    )


def _features_in_feature_list(pipeline: Pipeline,
                              feature_list: list[Feature]) -> bool:
    """
    Checks if the target and input features of pipeline are also in the dataset

    Args:
        pipeline: a pipeline instance
        feature_list: list with features of a dataset
    Returns:
        True if all features are in the feature list, else False
    """
    feature_names = [feature.name for feature in feature_list]

    for feature in pipeline._input_features:
        if feature.name not in feature_names:
            return False
    return pipeline._target_feature.name in feature_names


# ###PAGE SETUP###

st.set_page_config(page_title="Deployment", page_icon="🔧")

automl = AutoMLSystem.get_instance()

# ###DATASET DISPLAY & DELETING###

pipeline_list = automl.registry.list(type="pipeline")
names_pipeline_list = [pipeline.name for pipeline in pipeline_list]

if pipeline_list != []:

    st.header("Saved Pipelines")
    for name in names_pipeline_list:
        st.text(f"- {name}")

    st.header("Delete datasets")
    delete_name = st.selectbox('Which pipeline do you want to delete?',
                               names_pipeline_list, index=0)
    delete_pipeline = pipeline_list[names_pipeline_list.index(delete_name)]
    delete_button = st.button("Delete")

    if delete_button and delete_pipeline is not None:
        automl._registry.delete(delete_pipeline.id)
        st.success("File deleted")
        time.sleep(1)
        st.rerun()
else:
    st.warning("Please save a pipeline before loading them")

# ###PIPELINE SELECTION###

st.header("Make predictions")
name_selected_pipeline = st.selectbox("Choose your saved pipeline",
                                      options=names_pipeline_list)

if name_selected_pipeline:
    selected_pipeline = (pipeline_list[
        names_pipeline_list.index(name_selected_pipeline)])
    pipeline = _artifact_to_pipeline(selected_pipeline)
    # This function is created here instead of in Pipeline,
    # since we weren't allowed to modify it

    names_input_features = [feature.name for feature
                            in pipeline._input_features]

    metric_name_list = [metric.__class__.__name__ for
                        metric in pipeline._metrics]

    # ###PIPELINE SUMMARY###

    st.subheader("Pipeline Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""**Input Features**: {', '.join(names_input_features)
                    if pipeline._input_features else 'None'}"""
                    )
        st.markdown(f"**Target Feature**: {pipeline._target_feature.name}")

    with col2:
        st.markdown(f"**Model**: {pipeline.model.__class__.__name__}")
        st.markdown(f"**Data Split**: {pipeline._split}")
        metric_text = ", ".join(metric_name_list) \
            if pipeline._metrics else "None"
        st.markdown(f"**Metrics**: {metric_text}")

    # ###DATASET SELECTION###

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
                    as_dataset._asset_path = f"./{as_dataset.id}"
                    automl._registry.register(as_dataset)
                    st.success("File saved")
                    st.write(in_dataframe)
                else:
                    st.warning(f"File {item.name} is not a CSV file \
                               and cannot be used")

    datasets = automl.registry.list(type="dataset")
    if datasets != []:
        datasets_names = [dataset.name for dataset in datasets]
        name_cur_dataset = st.selectbox('With which file do you want to make \
                                        predictions? The target feature and \
                                        input features must be in the file!',
                                        datasets_names, index=0)
        cur_dataset = Dataset.from_artifact(datasets[
            datasets_names.index(name_cur_dataset)])
        features_cur_file = detect_feature_types(cur_dataset)
        if not _features_in_feature_list(pipeline, features_cur_file):
            st.warning(f"File {cur_dataset.name} does not have all the target \
                       or input features! Please choose another file.")
        else:

            # ###PREDICTIONS###

            if st.button(label="Predict"):
                pipeline._dataset = cur_dataset
                pipeline_results = pipeline.execute()

                st.write("**Predictions:**")
                st.write(pipeline_results.get('predictions'))
