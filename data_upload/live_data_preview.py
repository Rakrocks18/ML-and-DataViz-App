import streamlit as st

st.title("Live Data preview")
if "df" in st.session_state:
    st.subheader("Original Data")
    rws = st.slider("Preview Rows", 5, 25, 10)
    st.dataframe(st.session_state.df.head(rws))

if "preprocessed_df" in st.session_state:
    st.subheader("Preprocessed Data")
    rws1 = st.slider("Preview Preprocessed Rows", 5, 25, 10)
    st.dataframe(st.session_state.preprocessed_df.head(rws1))