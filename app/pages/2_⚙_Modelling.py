import streamlit as st
import pickle
import io

from typing import Any
from autoop.core.ml.model.model import Model
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression.lasso import LassoCV
from autoop.core.ml.model.regression.elastic_net import ElasticNet
from autoop.core.ml.model.classification.k_nearest_neighbours  \
    import KNearestNeighbors
from autoop.core.ml.model.classification.multi_layer_perceptron \
      import MultiLayerPerceptron
from autoop.core.ml.model.classification.random_forest_classifier \
      import RandomForestClassifier
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric


def get_model(model_name: str) -> 'Model':
    """Factory function to get a model by name."""
    match model_name:
        case "Multiple Linear Regression":
            return MultipleLinearRegression()
        case "Elastic Net":
            return ElasticNet()
        case "Lasso":
            return LassoCV()
        case "K Nearest Neighbors":
            return KNearestNeighbors()
        case "Random ForestClassifier":
            return RandomForestClassifier()
        case "Multi Layer Perceptron":
            return MultiLayerPerceptron()

# METRICS = [
#     "mean_squared_error",
#     "accuracy",
#     "mean_absolute_percentage_error",
#     "cohens_kappa",
#     "r_squared_score",
#     "precision"
# ]


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning \
                  pipeline to train a model on a dataset.")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
if datasets != []:
    datasets_names = [dataset.name for dataset in datasets]
    name_cur_dataset = st.selectbox(label="Select the dataset you \
                                    want to use", options=datasets_names)

    cur_dataset = Dataset.from_artifact(
        datasets[datasets_names.index(name_cur_dataset)])
    feature_list = detect_feature_types(cur_dataset)
    feature_list = feature_list[1:]

    input_features = st.multiselect(label="Choose the input features",
                                    options=feature_list)

# code to remove selected features from list here
target_options = [option for option in feature_list if
                  option not in input_features]
target_feature = st.selectbox(label="Choose the target feature",
                              options=target_options)


if target_feature.type == "categorical":
    task_type = "classification"
    model_options = ["K Nearest Neighbors", "Multi Layer Perceptron",
                     "Random Forest Classifier"]
    metric_options = [
        "Accuracy",
        "Cohens Kappa",
        "Precision"
    ]

else:
    task_type = "regression"
    model_options = ["Multiple Linear Regression", "Lasso", "Elastic Net"]
    metric_options = ["Mean Squared Error", "Mean Absolute Percentage Error",
                      "R-squared score"]

if name_cur_dataset and input_features and target_feature:
    model_selection = st.selectbox(label="Choose the model you want to use:",
                                   options=model_options)

    model = get_model(model_selection)

    data_split = st.slider(label="Select a data split:", min_value=0.0,
                           max_value=1.0, value=0.80)
    selected_metrics = st.multiselect(label="Select the metrics you \
                                       want to use:", options=metric_options)
# Summary here
    if selected_metrics:
        names_input_features = [feature.name for feature in input_features]
        st.subheader("Pipeline Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Dataset**: {name_cur_dataset}")
            st.markdown(f"**Input Features**: {', '.join(names_input_features)
                                               if input_features else 'None'}"
                        )
            st.markdown(f"**Target**: {target_feature}")

        with col2:
            st.markdown(f"**Model**: {model_selection}")
            st.markdown(f"**Data Split**: {data_split}")
            metric_text = ", ".join(selected_metrics) \
                if selected_metrics else "None"
            st.markdown(f"**Metrics**: {metric_text}")

# Train button here
        if st.button(label="Train model"):
            st.session_state.model_trained = True

        if st.session_state.model_trained:
            pipeline_metrics = [get_metric(metric)
                                for metric in selected_metrics]
            cur_pipeline = Pipeline(
                metrics=pipeline_metrics,
                dataset=cur_dataset,
                model=model,
                input_features=input_features,
                target_feature=target_feature,
                split=data_split
            )
            # st.write(new_pipeline.execute())

    # new_pipeline = Pipeline(
    #     metrics=[get_metric("r_squared_score")],
    #     dataset=cur_dataset,
    #     model=MultipleLinearRegression(),
    #     input_features=feature_list[:-1],
    #     target_feature=feature_list[-1],
    #     split=0.8
    # )
# Pipeline results here
            pipeline_results = cur_pipeline.execute()
            metric_results = pipeline_results.get('metrics')
            for metric in metric_results:
                st.write(f"{metric[0].__class__.__name__}: {metric[1]:.3f}")
            st.write(pipeline_results.get('predictions'))
            st.session_state.save_pipeline_clicked = True
            st.session_state.cur_pipeline = cur_pipeline

            st.write("Save Pipeline: ")
            pipeline_name = st.text_input("Give your pipeline a name")
            pipeline_version = st.text_input("write down the \
                                             version of the pipeline")

            if pipeline_name and pipeline_version:
                if st.button(label="Save pipeline"):
                    data = pickle.dumps({
                     "dataset": name_cur_dataset,
                     "model": model_selection,
                     "input_features": input_features,
                     "target_feature": target_feature,
                     "metrics": selected_metrics,
                     "split": data_split,
                    })

                    pipeline_artifact = Artifact(
                        name=pipeline_name,
                        version=pipeline_version,
                        asset_path="./assets",
                        data=data,
                        type='pipeline'
                        )
                    pipeline_artifact.asset_path = f"./{pipeline_artifact.id}"
                    automl.registry.register(pipeline_artifact)


                    # pipeline_list = automl.registry.list(type="pipeline")
                    # names_pipeline_list = [pipeline.name for pipeline in pipeline_list]
                    # name_selected_pipeline = st.selectbox("Choose your saved pipeline", options=names_pipeline_list)


                    # def artifact_to_pipeline(artifact: Artifact) -> "Pipeline":
                    #     pipeline_dict = pickle.loads(artifact)
                    #     return Pipeline(
                    #         dataset=pipeline_dict.get("dataset"),
                    #         model=pipeline_dict.get("model"),
                    #         input_features=pipeline_dict.get("input_features"),
                    #         target_feature=pipeline_dict.get("target_feature"),
                    #         split=pipeline_dict.get("split")
                    #     )

                    # selected_pipeline = (pipeline_list[names_pipeline_list.index(name_selected_pipeline)])
                    # new_pipeline = artifact_to_pipeline(selected_pipeline)

                    # cur_pipeline = st.session_state.cur_pipeline
                    # pipeline_artifacts = cur_pipeline.artifacts

                    # saved_pipeline = Artifact(
                    #     name=pipeline_name,
                    #     version=pipeline_version,
                    #     asset_path="./assets",
                    #     type="pipeline",
                    #     data=pickle.dumps("test")
                    # )
                    # for artifact in pipeline_artifacts:
                    #     print("artifact", artifact)
                    #     saved_pipeline.metadata[artifact.name] = artifact.id
                    #     artifact.asset_path = f"./{artifact.id}"
                    #     print(artifact)
                    #     automl.registry.register(artifact)

                    # saved_pipeline.asset_path = f"./{saved_pipeline.id}"
                    # automl.registry.register(saved_pipeline)
                    # st.success("Pipeline saved")

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

                    # config_path = current_pipeline.metadata.get("pipeline_config")
                    # pipeline_model = current_pipeline.metadata.get("pipeline_model_classification", "pipeline_model_regression")
                    # radius = current_pipeline.metadata.get("yield")

                    # print("-"*70)
                    # print(read_pickle_data(radius))
                    # print(read_pickle_data(pipeline_model))
                    # print("-"*70)

                    # pipeline_config = read_pickle_data(config_path)
                    # input_features = pipeline_config.get('input_features')
                    # target_feature = pipeline_config.get("target_feature")
                    # split = pipeline_config.get("split")


        # return Pipeline(
        #     name=artifact.name,
        #     version=artifact.version,
        #     asset_path=artifact.asset_path,
        #     tags=artifact.tags,
        #     data=artifact.data
        #     metadata= "hello"
        # )
        # self._dataset = dataset
        # self._model = model
        # self._input_features = input_features
        # self._target_feature = target_feature
        # self._metrics = metrics
        # self._artifacts = {}
        # self._split = split
# if st.session_state.save_pipeline_clicked:
#     if st.button(label="Save pipeline"):
#         pipeline_name = st.text_input("Give your pipeline a name")
#         pipeline_version = st.text_input("write down the
#                                         version of the pipeline")

#         if pipeline_name and pipeline_version:
#             cur_pipeline = st.session_state.cur_pipeline
#             pipeline_artifacts = cur_pipeline.artifacts

#             saved_pipeline = Artifact(
#                 name=pipeline_name,
#                 version=pipeline_version,
#                 asset_path="./assets",
#                 type="pipeline"
#             )
#             for artifact in pipeline_artifacts:
#                 saved_pipeline.metadata[artifact.name] = artifact.id
#                 automl.registry.register(artifact)
#             automl.registry.register(saved_pipeline)
#             st.success("Pipeline saved")

