import streamlit as st
# from data_upload.data_upload import data_upload_page

st.set_page_config(page_title="Data Preprocessing, Visualization and Modelling App", layout="wide")
st.title("Data Preprocessing, Visualization and Modelling App")
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

upload_data = st.Page("data_upload\\data_upload.py", title="Upload Data", icon=":material/upload_file:")

linear_model = st.Page("models\\regressions\\linear_regression.py", title="Linear Regression")
polynomial = st.Page("models\\regressions\\polynomial_regression.py", title="Polynomial Regression")

logistic_regression = st.Page("models\\classifications\\logistic_regression.py", title="Logistic Regression")

if "df" in st.session_state:
    pg = st.navigation(
        {
        "Data": [upload_data],
        "Regrssion Models": [linear_model, polynomial],
        "Classification Models": [logistic_regression]
        }
    )
    pg.run()
else:
    
    pg = st.navigation({
        "Data": [upload_data]
    })
    pg.run()
