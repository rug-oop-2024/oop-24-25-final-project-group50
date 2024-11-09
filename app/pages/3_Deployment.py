import streamlit as st
import pickle
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model

automl = AutoMLSystem.get_instance()


# pipelines = automl.registry.list(type="pipeline")
# st.selectbox(label="Test", options=[pipeline.name for pipeline in pipelines])

def read_pickle_data(path):
    data_bytes = automl._storage.load(path)
    with io.BytesIO(data_bytes) as file:
        data = pickle.load(file)
    return data


pipeline_list = automl.registry.list(type="pipeline")
names_pipeline_list = [pipeline.name for pipeline in pipeline_list]
data_pipeline_list = [pipeline.metadata for pipeline in pipeline_list]
# print(names_pipeline_list)
# print(data_pipeline_list)
name_cur_pipeline = st.selectbox(label="Select the pipeline you \
                want to use", options=names_pipeline_list)

current_pipeline = pipeline_list[names_pipeline_list.index(name_cur_pipeline)]


pipeline_config_id = current_pipeline.metadata.get("pipeline_config")
pipeline_config_artifact = automl.registry.get(pipeline_config_id)

config_data = pickle.loads(pipeline_config_artifact.data)
input_features = config_data.get("input_features")
target_feature = config_data.get("target_feature")
data_split = config_data.get("split")

# config_path = current_pipeline.metadata.get("pipeline_config")

# radius = current_pipeline.metadata.get("yield")

# print("-"*70)
# print(read_pickle_data(radius))
# print(read_pickle_data(pipeline_model))
# print("-"*70)


pipeline_model_id = current_pipeline.metadata.get("pipeline_model_classification", "pipeline_model_regression")
model_artifact = automl.registry.get(pipeline_model_id)

model_params = pickle.loads(model_artifact.data)
st.write(model_artifact)
loaded_model = get_model(model_artifact.metadata['model_name'])
loaded_model._model.set_params(model_params)

st.write(model_params)

# pipeline_config = read_pickle_data(config_path)
# input_features = pipeline_config.get('input_features')
# target_feature = pipeline_config.get("target_feature")
# split = pipeline_config.get("split")

# loaded_pipeline = Pipeline(
#     metrics=[],
#     dataset=None,
#     model=pipeline_model,
#     input_features=input_features,
#     target_feature=target_feature,
#     split=split
# )

