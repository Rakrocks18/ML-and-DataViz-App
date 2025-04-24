import streamlit as st
import plotly.express as px

st.title("Line Plot Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis Column", all_cols)
    with col2:
        y_col = st.selectbox("Select Y-axis Column", numeric_cols)
    
    color_col = st.selectbox("Color By", [None] + df.select_dtypes(exclude='number').columns.tolist())
    line_group = st.selectbox("Group By", [None] + df.select_dtypes(exclude='number').columns.tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        line_shape = st.selectbox("Line Shape", ['linear', 'spline'])
    with col2:
        markers = st.checkbox("Show Markers", True)
    
    if st.button("Generate Line Plot"):
        fig = px.line(df, x=x_col, y=y_col, color=color_col, line_group=line_group,
                      line_shape=line_shape, markers=markers)
        st.plotly_chart(fig, use_container_width=True)