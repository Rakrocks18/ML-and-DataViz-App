import streamlit as st
import plotly.express as px
# import polars as pl

st.title("Distribution Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    col = st.selectbox("Select Column", numeric_cols)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bins = st.slider("Number of Bins", 5, 100, 5)
    with col2:
        color = st.color_picker("Histogram Color", "#4CAF50")
    with col3:
        show_kde = st.checkbox("Show KDE", True)
    
    if st.button("Generate Distribution Plot"):
        fig = px.histogram(
            df, 
            x=col, 
            nbins=bins,
            color_discrete_sequence=[color],
            marginal="box" if show_kde else None,
            opacity=0.7
        )
        if show_kde:
            fig.add_vline(x=df[col].mean(), line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)