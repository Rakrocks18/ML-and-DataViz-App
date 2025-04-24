import streamlit as st
import plotly.express as px

st.title("Pie Chart Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        category_col = st.selectbox("Select Category Column", categorical_cols)
    with col2:
        value_col = st.selectbox("Select Value Column", numeric_cols)
    
    color = st.color_picker("Color Scheme", "#4CAF50")
    donut = st.checkbox("Donut Style", False)
    
    if st.button("Generate Pie Chart"):
        fig = px.pie(df, names=category_col, values=value_col, color_discrete_sequence=[color])
        if donut:
            fig.update_traces(hole=0.4)
        st.plotly_chart(fig, use_container_width=True)