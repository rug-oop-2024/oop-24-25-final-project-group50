import streamlit as st
import pandas as pd
import time

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# ###PAGE SETUP###

st.set_page_config(page_title="Datasets", page_icon="📊")

automl = AutoMLSystem.get_instance()

# ###DATASET UPLOADING###

st.header("Upload & Save datasets")

uploaded_datasets = st.file_uploader("Please upload your datasets here (.csv)",
                                     accept_multiple_files=True)
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
                st.warning(f"File {item.name} is not a CSV \
                           file and was not saved")

# ###DATASET DISPLAY & DELETING###

datasets = automl.registry.list(type="dataset")
datasets_names = [dataset.name for dataset in datasets]
if datasets != []:

    st.header("Saved datasets")
    for dataset in datasets:
        st.text(f"- {dataset.name}")

    st.header("Delete datasets")
    delete_file = st.selectbox('Which file do you want to delete?',
                               datasets_names, index=0)
    delete_dataset = datasets[datasets_names.index(delete_file)]
    delete_button = st.button("Delete")

    if delete_button and delete_dataset is not None:
        automl._registry.delete(delete_dataset.id)
        st.success("File deleted")
        time.sleep(1)
        st.rerun()
