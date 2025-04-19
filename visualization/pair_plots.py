from typing import Literal
import streamlit as st
import seaborn as sns
# import matplotlib.pyplot as plt
# import polars as pl

st.title("Pair Relationships Explorer")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    selected_cols = st.multiselect("Select Numerical Columns", numeric_cols, default=numeric_cols[:3])
    hue = st.selectbox("Color By", [None] + categorical_cols)
    reg_line = st.checkbox("Show Regression Lines")
    diag_kind: Literal['auto']|Literal['hist']|Literal['kde'] = st.radio("Diagonal Plot Type", ['auto', 'hist', 'kde'])
    
    if st.button("Generate Pair Plot") and len(selected_cols) > 1:
        g = sns.pairplot(
            df[selected_cols + ([hue] if hue else [])],
            hue=hue,
            kind="reg" if reg_line else "scatter",
            diag_kind=diag_kind,
            plot_kws={'alpha': 0.6},
            height=2
        )
        st.pyplot(g.fig)