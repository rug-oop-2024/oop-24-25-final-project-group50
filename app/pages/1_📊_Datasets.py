import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Datasets", page_icon="📊")
automl = AutoMLSystem.get_instance()
uploaded_datasets = st.file_uploader("Please upload your datasets here",
                                     accept_multiple_files=True)
if uploaded_datasets:
    save_button = st.button("Save File")
    if save_button:
        for item in uploaded_datasets:
            in_dataframe = pd.read_csv(item)
            as_dataset = Dataset.from_dataframe(in_dataframe, name=item.name,
                                                asset_path="./assets")
            as_dataset.asset_path = f"./{as_dataset.id}"
            automl._registry.register(as_dataset)

datasets = automl.registry.list(type="dataset")
if datasets != []:
    datasets_names = [dataset.name for dataset in datasets]
    chosen_dataset = st.markdown(datasets_names)
    delete_file = st.selectbox('Which file do you want to delete?',
                               datasets_names, index=0)
    delete_dataset = datasets[datasets_names.index(delete_file)]
    delete_button = st.button("Delete")

    if delete_button and delete_dataset is not None:
        automl._registry.delete(delete_dataset.id)
        st.rerun()
