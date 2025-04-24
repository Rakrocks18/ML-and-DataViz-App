import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle

st.title("Waffle Chart Analysis")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    category_col = st.selectbox("Select Category Column", categorical_cols)
    value_col = st.selectbox("Select Value Column", numeric_cols)
    rows = st.slider("Grid Rows", 5, 20, 10)
    cols = st.slider("Grid Columns", 5, 20, 10)
    
    if st.button("Generate Waffle Chart"):
        total = df[value_col].sum()
        proportions = (df[value_col] / total).values
        blocks = (proportions * (rows * cols)).astype(int)
        
        # Create a color cycler to handle more categories than default colors
        color_cycler = cycle(px.colors.qualitative.Plotly)
        
        fig = go.Figure()
        block_idx = 0
        for cat, count in zip(df[category_col], blocks):
            # Get next color from cycler
            color = next(color_cycler)
            for _ in range(count):
                fig.add_shape(type="rect",
                    x0=block_idx % cols, y0=block_idx // cols,
                    x1=(block_idx % cols) + 1, y1=(block_idx // cols) + 1,
                    fillcolor=color,
                    line=dict(width=0)
                )
                block_idx += 1
        fig.update_layout(width=800, height=800, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)