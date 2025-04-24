import streamlit as st
import plotly.express as px

st.title("Scatter Plot Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-axis Column", numeric_cols)
    with col2:
        y_col = st.selectbox("Y-axis Column", numeric_cols)
    
    color_col = st.selectbox("Color By", [None] + df.select_dtypes(exclude='number').columns.tolist())
    size_col = st.selectbox("Size By", [None] + numeric_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        trendline = st.selectbox("Trendline", [None, "ols", "lowess"])
    with col2:
        marginal = st.selectbox("Marginal Plot", [None, "histogram", "box", "violin"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7)
    with col2:
        size_max = st.slider("Max Size", 1, 50, 20)
    with col3:
        hover_data = st.multiselect("Hover Data", all_cols)

    if st.button("Generate Scatter Plot"):
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col, size=size_col,
            trendline=trendline, marginal_x=marginal, marginal_y=marginal,
            opacity=opacity, size_max=size_max, hover_data=hover_data
        )
        st.plotly_chart(fig, use_container_width=True)