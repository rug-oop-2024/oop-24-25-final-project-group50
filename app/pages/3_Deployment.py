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
    return [get_metric(metric) for metric in metric_name_list]


def artifact_to_pipeline(artifact: Artifact) -> "Pipeline":
    pipeline_dict = pickle.loads(artifact.data)
    return Pipeline(
        model=get_model(pipeline_dict.get("model")),
        input_features=pipeline_dict.get("input_features"),
        target_feature=pipeline_dict.get("target_feature"),
        split=pipeline_dict.get("split"),
        metrics=get_all_metrics(pipeline_dict.get("metrics")),
        dataset=None
    )


def features_in_feature_list(pipeline: Pipeline, feature_list: list[Feature]) -> bool:
    for feature in pipeline._input_features:
        if feature not in feature_list:
            return True
    return pipeline._target_feature not in feature_list


automl = AutoMLSystem.get_instance()
saved_pipelines = []

pipeline_list = automl.registry.list(type="pipeline")
names_pipeline_list = [pipeline.name for pipeline in pipeline_list]
name_selected_pipeline = st.selectbox("Choose your saved pipeline", options=names_pipeline_list)

if name_selected_pipeline:
    selected_pipeline = (pipeline_list[names_pipeline_list.index(name_selected_pipeline)])
    pipeline = artifact_to_pipeline(selected_pipeline)
    names_input_features = [feature.name for feature in pipeline._input_features]
    metric_name_list = [metric.__class__.__name__ for metric in pipeline._metrics]
    st.subheader("Pipeline Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Input Features**: {', '.join(names_input_features)
                                           if pipeline._input_features
                                           else 'None'}"
                    )
        st.markdown(f"**Target**: {pipeline._target_feature.name}")

    with col2:
        st.markdown(f"**Model**: {pipeline.model.__class__.__name__}")
        st.markdown(f"**Data Split**: {pipeline._split}")
        metric_text = ", ".join(metric_name_list) \
            if pipeline._metrics else "None"
        st.markdown(f"**Metrics**: {metric_text}")

    uploaded_datasets = st.file_uploader("Please upload your datasets here (.csv)")
    if uploaded_datasets:
        save_button = st.button("Save File")
        if save_button:
            for item in uploaded_datasets:
                if item.name.endswith('.csv'):
                    in_dataframe = pd.read_csv(item)
                    as_dataset = Dataset.from_dataframe(in_dataframe, name=item.name, asset_path="./assets")
                    as_dataset.asset_path = f"./{as_dataset.id}"
                    automl._registry.register(as_dataset)
                    st.success("File saved")
                    st.write(in_dataframe)
                else:
                    st.warning(f"File {item.name} is not a CSV file and was not saved")

    datasets = automl.registry.list(type="dataset")
    if datasets != []:
        datasets_names = [dataset.name for dataset in datasets]
        # chosen_dataset = st.markdown(datasets_names)
        name_cur_dataset = st.selectbox('With which file do you want to make predictions? The target feature and input features must be in the file!',
                                datasets_names, index=0)
        cur_dataset = Dataset.from_artifact(datasets[datasets_names.index(name_cur_dataset)])
        features_cur_file = detect_feature_types(cur_dataset)
        if features_in_feature_list(pipeline, features_cur_file):
            # print(pipeline._input_features, 'a', pipeline._target_feature, 'a', features_cur_file)
            st.warning(f"File {cur_dataset.name} does not have all the target or input features! Please choose another file.")
        else:
            if st.button(label="Train model"):
                st.session_state.model_trained = True

                if st.session_state.model_trained:
                    pipeline._dataset = cur_dataset
                    pipeline_results = pipeline.execute()
                    metric_results = pipeline_results.get('metrics')
                    for metric in metric_results:
                        st.write(f"{metric[0].__class__.__name__}: {metric[1]:.3f}")
                    st.write(pipeline_results.get('predictions'))
                    st.session_state.save_pipeline_clicked = True
                    st.session_state.pipeline = pipeline
















# # pipelines = automl.registry.list(type="pipeline")
# # st.selectbox(label="Test", options=[pipeline.name for pipeline in pipelines])

# def read_pickle_data(path):
#     data_bytes = automl._storage.load(path)
#     with io.BytesIO(data_bytes) as file:
#         data = pickle.load(file)
#     return data


# pipeline_list = automl.registry.list(type="pipeline")
# names_pipeline_list = [pipeline.name for pipeline in pipeline_list]
# data_pipeline_list = [pipeline.metadata for pipeline in pipeline_list]
# # print(names_pipeline_list)
# # print(data_pipeline_list)
# name_cur_pipeline = st.selectbox(label="Select the pipeline you \
#                 want to use", options=names_pipeline_list)

# current_pipeline = pipeline_list[names_pipeline_list.index(name_cur_pipeline)]


# pipeline_config_id = current_pipeline.metadata.get("pipeline_config")
# pipeline_config_artifact = automl.registry.get(pipeline_config_id)

# config_data = pickle.loads(pipeline_config_artifact.data)
# input_features = config_data.get("input_features")
# target_feature = config_data.get("target_feature")
# data_split = config_data.get("split")

# # config_path = current_pipeline.metadata.get("pipeline_config")

# # radius = current_pipeline.metadata.get("yield")

# # print("-"*70)
# # print(read_pickle_data(radius))
# # print(read_pickle_data(pipeline_model))
# # print("-"*70)


# pipeline_model_id = current_pipeline.metadata.get("pipeline_model_classification", "pipeline_model_regression")
# model_artifact = automl.registry.get(pipeline_model_id)

# model_params = pickle.loads(model_artifact.data)
# st.write(model_artifact)
# loaded_model = get_model(model_artifact.metadata['model_name'])
# loaded_model._model.set_params(model_params)

# st.write(model_params)

# # pipeline_config = read_pickle_data(config_path)
# # input_features = pipeline_config.get('input_features')
# # target_feature = pipeline_config.get("target_feature")
# # split = pipeline_config.get("split")

# # loaded_pipeline = Pipeline(
# #     metrics=[],
# #     dataset=None,
# #     model=pipeline_model,
# #     input_features=input_features,
# #     target_feature=target_feature,
# #     split=split
# # )

