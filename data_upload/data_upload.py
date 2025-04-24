
from data_upload.fileopener import open_file
import streamlit as st
import polars as pl

if "df" not in st.session_state:
    st.session_state.df = None
    st.subheader("Upload a file to start")
    
uploaded_file = st.file_uploader("Upload a .csv, .tsv, or .xlsx file", type=["csv", "tsv", "xlsx", "xls"])
has_header = st.checkbox("Has header", value=True)
if uploaded_file is not None:
    df: pl.DataFrame | None = open_file(uploaded_file, has_header)
    if df is None:
        st.error("There was an error loading your file!")
    else:
        st.session_state.df = df
        # st.subheader("Raw Data Preview")
        # st.dataframe(st.session_state.df.head())
