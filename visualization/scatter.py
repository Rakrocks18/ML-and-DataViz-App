import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# import polars as pl

st.title("Scatter Plot Explorer")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X Axis", numeric_cols)
        y_axis = st.selectbox("Y Axis", numeric_cols, index=1 if len(numeric_cols) >1 else 0)
        hue = st.selectbox("Color By", [None] + categorical_cols)
        
    with col2:
        size = st.slider("Point Size", 10, 200, 50)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.8)
        palette = st.selectbox("Color Palette", ["viridis", "magma", "plasma", "coolwarm"])
    
    if st.button("Generate Plot"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x=x_axis,
            y=y_axis,
            hue=hue,
            s=size,
            alpha=opacity,
            palette=palette,
            ax=ax
        )
        ax.set_title(f"{y_axis} vs {x_axis}")
        st.pyplot(fig)
        # st.download_button("Download Plot", fig.savefig(fname="scatter_plot.png"), file_name="scatter_plot.png")