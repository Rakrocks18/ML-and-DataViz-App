import streamlit as st
import polars as pl

st.title("Handle Missing Data")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df
    if "preprocessed_df" in st.session_state:
        df = st.session_state.preprocessed_df
    st.subheader("Current Data Summary")
    st.write(f"Shape: {df.shape}")
    
    null_counts = df.null_count().to_pandas().T
    st.dataframe(null_counts.rename(columns={0: "Missing Values"}))

    method = st.selectbox(
        "Handling Method",
        ["Drop Rows", "Drop Columns", "Impute"]
    )
    
    cols = st.multiselect("Select columns", df.columns)
    
    if st.button("Apply"):
        if method == "Drop Rows":
            if cols:
                df = df.drop_nulls(subset=cols)
            else:
                df = df.drop_nulls()
        elif method == "Drop Columns":
            df = df.drop(cols)
        elif method == "Impute":
            impute_method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode"])
            for col in cols:
                if impute_method == "Mean":
                    val = df[col].mean()
                elif impute_method == "Median":
                    val = df[col].median()
                else:  # Mode
                    val = df[col].mode()[0]
                df = df.with_columns(pl.col(col).fill_null(val))
        
        st.session_state.preprocessed_df = df
        st.success("Data updated!")
        st.dataframe(df.head().to_pandas())