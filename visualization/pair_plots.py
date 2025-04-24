import streamlit as st
import plotly.express as px

st.title("Pair Plot Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    selected_cols = st.multiselect("Select Numerical Columns", numeric_cols, default=numeric_cols[:3])
    color_col = st.selectbox("Color By", [None] + categorical_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        diag_kind = st.selectbox("Diagonal Plot", ["histogram", "box", "violin"])
    with col2:
        marker_size = st.slider("Marker Size", 1, 10, 3)
    
    col1, col2 = st.columns(2)
    with col1:
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7)
    with col2:
        show_upper = st.checkbox("Show Upper Triangle", True)

    if len(selected_cols) >= 2 and st.button("Generate Pair Plot"):
        fig = px.scatter_matrix(
            df, dimensions=selected_cols, color=color_col,
            opacity=opacity
        )
        fig.update_traces(
            marker=dict(size=marker_size),
            showupperhalf=show_upper,
            diagonal_visible=True
        )
        st.plotly_chart(fig, use_container_width=True)