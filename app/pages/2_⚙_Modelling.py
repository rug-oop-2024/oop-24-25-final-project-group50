import streamlit as st
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric
from autoop.core.ml.model import get_model


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    """
    Function to write text in Streamlit in a formatted way.

    Args:
        text: the text to be written in Streamit
    Returns:
        None
    """
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
