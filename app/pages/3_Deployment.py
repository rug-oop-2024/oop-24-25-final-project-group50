import streamlit as st
from app.core.system import AutoMLSystem

automl = AutoMLSystem.get_instance()


pipelines = automl.registry.list(type="pipeline")
st.selectbox(label="Test", options=[pipeline.name for pipeline in pipelines])

