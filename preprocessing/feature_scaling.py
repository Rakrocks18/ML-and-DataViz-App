import streamlit as st
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.title("Feature Scaling")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    if "preprocessed_df" in st.session_state:
        df = st.session_state.preprocessed_df.to_pandas()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    selected_cols = st.multiselect("Select numeric columns", numeric_cols)
    method = st.selectbox("Scaling Method", ["Standardization", "Normalization"])
    
    if st.button("Apply Scaling"):
        scaler = StandardScaler() if method == "Standardization" else MinMaxScaler()
        df[selected_cols] = scaler.fit_transform(df[selected_cols])
        st.session_state.preprocessed_df = pl.from_pandas(df)
        st.success("Scaling applied!")
        st.dataframe(df.head())