import streamlit as st
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric, REGRESSION_METRICS, \
    CLASSIFICATION_METRICS
from autoop.core.ml.model import get_model, REGRESSION_MODELS, \
    CLASSIFICATION_MODELS


# ###HELPER FUNCTIONS###

def _write_helper_text(text: str) -> None:
    """
    Function to write text in Streamlit in a formatted way.

    Args:
        text: the text to be written in Streamit
    Returns:
        None
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# ###PAGE SETUP###

st.set_page_config(page_title="Modelling", page_icon="📈")

st.write("# ⚙ Modelling")
_write_helper_text("In this section, you can train a model on a dataset.")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

automl = AutoMLSystem.get_instance()

# ###DATASET LOADING###

datasets = automl.registry.list(type="dataset")

if not datasets:
    st.warning("Please upload a dataset to start training.")
else:
    if datasets != []:

        # ###DATASET SELECTION###

        datasets_names = [dataset.name for dataset in datasets]
        name_cur_dataset = st.selectbox(label="Select the dataset you \
                                        want to use", options=datasets_names)

        cur_dataset = Dataset.from_artifact(
            datasets[datasets_names.index(name_cur_dataset)])

        # ###FEATURE SELECTION###

        feature_list = detect_feature_types(cur_dataset)

        input_features = st.multiselect(label="Choose the input features",
                                        options=feature_list)
        input_feature_names = [feature.name for feature in input_features]

        target_options = [option for option in feature_list
                          if option.name not in input_feature_names]
        target_feature = st.selectbox(label="Choose the target feature",
                                      options=target_options)

        # ###MODEL & METRIC GENERATION###

        if target_feature.type == "categorical":
            model_options = CLASSIFICATION_MODELS
            metric_options = CLASSIFICATION_METRICS
        else:
            model_options = REGRESSION_MODELS
            metric_options = REGRESSION_METRICS

    if name_cur_dataset and input_features and target_feature:

        # ###MODEL, SPLIT & METRIC SELECTION###

        model_selection = st.selectbox(
                            label="Choose the model you want to use:",
                            options=model_options)

        model = get_model(model_selection)

        data_split = st.slider(label="Select a data split:", min_value=0.01,
                               max_value=0.99, value=0.80)

        selected_metrics = st.multiselect(label="Select the metrics you \
                                        want to use:", options=metric_options)

        if selected_metrics:

            # ###PIPELINE SUMMARY GENERATION###

            name_input_features = [feature.name for feature in input_features]
            st.subheader("Pipeline Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Dataset**: {name_cur_dataset}")
                feature_text = ', '.join(name_input_features) \
                    if input_features else 'None'
                st.markdown(f"**Input Features**: {feature_text}")
                st.markdown(f"**Target**: {target_feature}")

            with col2:
                st.markdown(f"**Model**: {model_selection}")
                st.markdown(f"**Data Split**: {data_split}")
                metric_text = ", ".join(selected_metrics) \
                    if selected_metrics else "None"
                st.markdown(f"**Metrics**: {metric_text}")

            # ###MODEL TRAINING###

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

                # ###PIPELINE RESULT DISPLAY###

                pipeline_results = cur_pipeline.execute()
                metric_results = pipeline_results.get('metrics')
                train_metric_results = \
                    pipeline_results.get('metric for training')

                for metric in metric_results:
                    st.write(f"Test Data: \
                            {metric[0].__class__.__name__}: {metric[1]:.3f}")
                for metric in train_metric_results:
                    st.write(f"Training Data: \
                            {metric[0].__class__.__name__}: {metric[1]:.3f}")

                st.write("**Predictions:**")
                st.write(pipeline_results.get('predictions'))

                # ###PIPELINE SAVING###

                st.write("Save Pipeline: ")
                pipeline_name = st.text_input("Give your pipeline a name")
                pipeline_version = st.text_input("write down the "
                                                 "version of the pipeline")

                if pipeline_name and pipeline_version:
                    if st.button(label="Save pipeline"):
                        data = pickle.dumps(
                            {
                                "dataset": name_cur_dataset,
                                "model": model_selection,
                                "input_features": input_features,
                                "target_feature": target_feature,
                                "metrics": selected_metrics,
                                "split": data_split,
                            }
                        )

                        pipeline_artifact = Artifact(
                            name=pipeline_name,
                            version=pipeline_version,
                            asset_path="./assets",
                            data=data,
                            type='pipeline'
                        )
                        pipeline_artifact._asset_path = \
                            f"./{pipeline_artifact.id}"
                        automl.registry.register(pipeline_artifact)
                        st.success("Pipeline saved")
