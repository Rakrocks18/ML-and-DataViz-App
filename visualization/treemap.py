import streamlit as st
import plotly.express as px

st.title("Treemap Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    path = st.multiselect("Select Hierarchy Path", categorical_cols)
    size_col = st.selectbox("Select Size Column", numeric_cols)
    color_col = st.selectbox("Color By (Optional)", [None] + numeric_cols)
    
    if st.button("Generate Treemap") and path and size_col:
        fig = px.treemap(df, path=path, values=size_col, color=color_col)
        st.plotly_chart(fig, use_container_width=True)