import streamlit as st
from visualization.plotting import plot_section

def visualization_page():
    """
    Main visualization page that handles data visualization functionality.
    """
    st.title("Data Visualization")
    
    # Check if data is available in session state
    if "df" not in st.session_state:
        st.warning("Please upload data first!")
        return
        
    df = st.session_state.df
    
    # Data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head().to_pandas())
        
    # Show plotting section
    plot_section(df) 