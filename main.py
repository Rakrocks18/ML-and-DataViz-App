import streamlit as st

st.set_page_config(page_title="Data Preprocessing, Visualization and Modelling App", layout="wide")
st.title("Data Preprocessing, Visualization and Modelling App")

# Initialize session state
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# Define pages
upload_data = st.Page("data_upload/data_upload.py", title="Upload Data", icon="📤")

# Regression Models
linear_model = st.Page("models/regressions/linear_regression.py", title="Linear Regression")
polynomial = st.Page("models/regressions/polynomial_regression.py", title="Polynomial Regression")

# Classification Models
logistic_regression = st.Page("models/classifications/logistic_regression.py", title="Logistic Regression")

# Sequential Learning Models
hmm_model = st.Page("models/sequential_learning/hmm_model.py", title="Hidden Markov Model")

# Navigation based on data availability
if "df" in st.session_state:
    pg = st.navigation({
        "Data": [upload_data],
        "Regression Models": [linear_model, polynomial],
        "Classification Models": [logistic_regression],
        "Sequential Learning": [hmm_model]
    })
else:
    pg = st.navigation({
        "Data": [upload_data]
    })

pg.run()
