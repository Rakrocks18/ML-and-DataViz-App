import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

st.title("Correlation Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols)
    method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
    
    col1, col2 = st.columns(2)
    with col1:
        color_scale = st.selectbox("Color Scale", px.colors.named_colorscales(), index=9)
    with col2:
        cluster = st.checkbox("Cluster Variables", True)
    
    annotate = st.checkbox("Show Values", True)
    zmin = st.slider("Min Value", -1.0, 0.0, -1.0)
    zmax = st.slider("Max Value", 0.0, 1.0, 1.0)

    if len(selected_cols) >= 2 and st.button("Generate Correlation Plot"):
        corr_matrix = df[selected_cols].corr(method=method)
        
        if cluster:
            from scipy.cluster import hierarchy
            dist = hierarchy.distance.squareform(1 - np.abs(corr_matrix))
            linkage = hierarchy.linkage(dist, method='average')
            order = hierarchy.leaves_list(linkage)
            corr_matrix = corr_matrix.iloc[order, :].iloc[:, order]

        fig = px.imshow(
            corr_matrix,
            color_continuous_scale=color_scale,
            zmin=zmin,
            zmax=zmax,
            text_auto=True if annotate else False
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)