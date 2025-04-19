import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
# import polars as pl

st.title("Box Plot Analyzer")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        numerical = st.selectbox("Numerical Column", numeric_cols)
        categorical = st.selectbox("Categorical Column", categorical_cols)
    with col2:
        orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
        palette = st.selectbox("Color Palette", sns.palettes.SEABORN_PALETTES)
    
    if st.button("Generate Box Plot"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x=categorical if orientation == "Vertical" else numerical,
            y=numerical if orientation == "Vertical" else categorical,
            data=df,
            palette=palette,
            ax=ax
        )
        ax.set_title(f"{numerical} Distribution by {categorical}")
        st.pyplot(fig)