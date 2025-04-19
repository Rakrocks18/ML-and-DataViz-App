import streamlit as st
import plotly.figure_factory as ff
# import polars as pl

st.title("Correlation Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    method = st.radio("Correlation Method", ["pearson", "spearman", "kendall"])
    color_scale = st.selectbox("Color Scale", 
                             ["Viridis", "Plasma", "RdBu", "Portland"])
    
    if st.button("Show Correlation Matrix"):
        corr = df[numeric_cols].corr(method=method.lower())
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=color_scale.lower(),
            annotation_text=corr.round(2).values,
            showscale=True
        )
        fig.update_layout(width=800, height=800)
        st.plotly_chart(fig, use_container_width=True)