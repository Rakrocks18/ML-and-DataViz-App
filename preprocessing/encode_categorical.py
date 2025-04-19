import streamlit as st
import polars as pl
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

st.title("Categorical Encoding")



if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.to_pandas()
    if "preprocessed_df" in st.session_state:
        df = st.session_state.preprocessed_df.to_pandas()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.info("No categorical columns found")
    else:
        selected_cols = st.multiselect("Select categorical columns", categorical_cols)
        method = st.radio("Encoding Method", ["One-Hot Encoding", "Label Encoding"])
        
        if st.button("Apply Encoding"):
            if not selected_cols:
                st.error("Please select at least one column!")
                st.stop()
                
            try:
                if method == "One-Hot Encoding":
                    ohe = OneHotEncoder(sparse_output=False, drop='first')
                    encoded_array = ohe.fit_transform(df[selected_cols])
                    
                    # Explicitly typed column names with string conversion
                    new_columns: list[str] = []
                    for i, col in enumerate(selected_cols):
                        categories = [str(c) for c in ohe.categories_[i][1:]]
                        for category in categories:
                            new_columns.append(f"{col}")
                    
                    encoded_df = pd.DataFrame(encoded_array, columns=new_columns)
                    df = pd.concat([
                        df.drop(columns=selected_cols),
                        encoded_df
                    ], axis=1)
                    
                else:
                    le = LabelEncoder()
                    for col in selected_cols:
                        df[col] = le.fit_transform(df[col].astype(str))
                
                st.session_state.preprocessed_df = pl.from_pandas(df)
                st.success("Encoding applied successfully!")
                st.subheader("Transformed Data Preview")
                st.dataframe(df.head())
                
            except Exception as e:
                st.error(f"Error during encoding: {str(e)}")