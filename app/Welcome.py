import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.sidebar.success("Select a page above.")
# st.markdown(open("README.md").read())


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 👋 Welcome")
write_helper_text("Please select a page below to start using the AutoML")
