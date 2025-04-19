import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl

def get_plot(df: pl.DataFrame, plot_type: str, x_col: str, y_col: str, color_col: str = None, title: str = None):
    """
    Generate an interactive plot using Plotly Express with enhanced customization.
    
    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame
    plot_type : str
        Type of plot to generate
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    color_col : str, optional
        Column name for color encoding
    title : str, optional
        Custom title for the plot
    """
    # Convert Polars DataFrame to pandas for Plotly
    pdf = df.to_pandas()
    
    # Default plot settings
    plot_settings = {
        "template": "plotly_white",
        "title": title or f"{plot_type}: {x_col} vs {y_col}",
        "x": x_col,
        "y": y_col,
    }
    
    if color_col:
        plot_settings["color"] = color_col

    try:
        if plot_type == "Line Plot":
            fig = px.line(pdf, **plot_settings)
        elif plot_type == "Scatter Plot":
            fig = px.scatter(pdf, **plot_settings)
        elif plot_type == "Bar Chart":
            fig = px.bar(pdf, **plot_settings)
        elif plot_type == "Box Plot":
            fig = px.box(pdf, **plot_settings)
        elif plot_type == "Violin Plot":
            fig = px.violin(pdf, **plot_settings)
        elif plot_type == "Histogram":
            fig = px.histogram(pdf, x=x_col, title=f"Histogram of {x_col}")
        elif plot_type == "Heatmap":
            pivot_table = pd.pivot_table(pdf, values=y_col, index=x_col, columns=color_col if color_col else x_col)
            fig = px.imshow(pivot_table, title=f"Heatmap: {x_col} vs {y_col}")
        else:
            st.error("Selected plot type is not supported!")
            return None

        # Enhance the layout
        fig.update_layout(
            title_x=0.5,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title=x_col,
            yaxis_title=y_col,
        )
        
        # Add hover data
        fig.update_traces(
            hovertemplate="<br>".join([
                f"{x_col}: %{{x}}",
                f"{y_col}: %{{y}}",
                "<extra></extra>"
            ])
        )

        return fig
    
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return None

def plot_section(df: pl.DataFrame):
    """
    Create an interactive plotting section in the Streamlit app.
    """
    st.header("Data Visualization")
    
    # Available plot types
    plot_types = [
        "Line Plot",
        "Scatter Plot",
        "Bar Chart",
        "Box Plot",
        "Violin Plot",
        "Histogram",
        "Heatmap"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        plot_type = st.selectbox("Select Plot Type", plot_types)
        x_col = st.selectbox("Select X-axis column", df.columns)
    
    with col2:
        y_col = st.selectbox("Select Y-axis column", df.columns)
        color_col = st.selectbox("Select Color column (optional)", ["None"] + list(df.columns))
    
    color_col = None if color_col == "None" else color_col
    
    # Additional plot settings
    with st.expander("Plot Settings"):
        title = st.text_input("Custom Plot Title", value="")
        
    if st.button("Generate Plot", key="generate_plot"):
        fig = get_plot(df, plot_type, x_col, y_col, color_col, title)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for the plot
            st.download_button(
                label="Download Plot as HTML",
                data=fig.to_html(),
                file_name=f"{plot_type.lower().replace(' ', '_')}.html",
                mime="text/html"
            ) 