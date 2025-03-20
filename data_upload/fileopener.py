import polars as pl
import streamlit as st

@st.cache_data
def open_file(uploaded_file, has_header)-> pl.DataFrame | None:

    if uploaded_file.name.endswith('.csv'):
        # Load the CSV file using Polars
        df = pl.read_csv(uploaded_file, has_header=has_header)
        return df
    elif uploaded_file.name.endswith('.xlsx'):
        # Load the Excel file using Polars
        df = pl.read_excel(uploaded_file)
        return df
    elif uploaded_file.name.endswith('.tsv'):
        # Load the TSV file using Polars
        df = pl.read_csv(uploaded_file, separator='\t', has_header=has_header)
        return df
    else:
        st.error("Unsupported file format. Please upload a CSV, TSV, or XLSX file.")
        return None