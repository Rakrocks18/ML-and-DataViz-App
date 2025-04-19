import streamlit as st
from sklearn.model_selection import train_test_split
import polars as pl


st.title("Data Splitting")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    if "preprocessed_df" in st.session_state:
        df = st.session_state.preprocessed_df.to_pandas()
    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
    target = st.selectbox("Select target column", df.columns)
    random_state = st.number_input("Random State", 42)
    
    if st.button("Split Data"):
        X: pl.DataFrame = df.drop(columns=target)
        y: pl.DataFrame = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        st.success(f"Split complete! Train: {X_train.shape}, Test: {X_test.shape}")