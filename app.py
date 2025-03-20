import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Data Loading Functions
# ------------------------------
def load_data(file) -> pl.DataFrame:
    """Load a CSV, TSV, or XLSX file into a Polars DataFrame."""
    ext = file.name.split('.')[-1].lower()
    if ext == "csv":
        df = pl.read_csv(file)
    elif ext == "tsv":
        df = pl.read_csv(file, separator="\t")
    elif ext in ["xlsx", "xls"]:
        # Polars does not read Excel natively, so we use pandas then convert.
        df_pd = pd.read_excel(file)
        df = pl.from_pandas(df_pd)
    else:
        st.error("Unsupported file type!")
        df = None
    return df

# ------------------------------
# Preprocessing Functions
# ------------------------------
def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess the data based on the user's selection.
    Operations include dropping missing values, filling missing values,
    or converting selected columns to numeric.
    """
    st.subheader("Preprocessing Options")
    operation = st.selectbox(
        "Select Preprocessing Operation", 
        ["None", "Drop Missing Values", "Fill Missing Values", "Convert Columns to Numeric"]
    )
    
    if operation == "None":
        st.info("No preprocessing applied.")
        return df

    if operation == "Drop Missing Values":
        st.write("Dropping rows with missing values.")
        return df.drop_nulls()

    if operation == "Fill Missing Values":
        fill_val = st.text_input("Fill missing values with", value="0")
        st.write(f"Filling missing values with: {fill_val}")
        return df.fill_null(fill_val)

    if operation == "Convert Columns to Numeric":
        columns_to_convert = st.multiselect("Select columns to convert to numeric", df.columns)
        for col in columns_to_convert:
            try:
                df = df.with_column(pl.col(col).cast(pl.Float64))
                st.write(f"Converted column '{col}' to numeric type.")
            except Exception as e:
                st.error(f"Error converting column {col}: {e}")
        return df

# ------------------------------
# Plotting Functions
# ------------------------------
def get_plot(df: pl.DataFrame, plot_type: str, x_col: str, y_col: str):
    """
    Generate a plot using Plotly Express.
    Easily extendable by adding more plot types.
    """
    # Convert Polars DataFrame to pandas for Plotly
    pdf = df.to_pandas()

    if plot_type == "Line Plot":
        fig = px.line(pdf, x=x_col, y=y_col, title=f"Line Plot: {x_col} vs {y_col}")
    elif plot_type == "Scatter Plot":
        fig = px.scatter(pdf, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
    elif plot_type == "Bar Chart":
        fig = px.bar(pdf, x=x_col, y=y_col, title=f"Bar Chart: {x_col} vs {y_col}")
    else:
        st.error("Selected plot type is not supported!")
        fig = None
    return fig

# ------------------------------
# ML Model Functions
# ------------------------------
def run_grid_search(df: pl.DataFrame, features: list, target: str, model_info: dict):
    """
    Run GridSearchCV on the provided dataset.
    'model_info' is a dictionary containing the sklearn estimator and parameter grid.
    """
    # Convert Polars DataFrame to pandas for scikit-learn
    pdf = df.to_pandas()
    X = pdf[features]
    y = pdf[target]

    grid_search = GridSearchCV(model_info["estimator"], model_info["param_grid"], cv=3)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

# ------------------------------
# Main Application
# ------------------------------
def main():
    st.title("Data Visualization, Preprocessing & ML Modeling App")

    # Ensure session state for preprocessed data exists.
    if "preprocessed_df" not in st.session_state:
        st.session_state.preprocessed_df = None

    uploaded_file = st.file_uploader("Upload a .csv, .tsv, or .xlsx file", type=["csv", "tsv", "xlsx", "xls"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            st.error("There was an error loading your file!")
            return

        st.subheader("Raw Data Preview")
        st.dataframe(df.head().to_pandas())

        # Sidebar: Choose the action
        action = st.sidebar.selectbox("Select Action", ["Preprocessing", "Plotting", "ML Model"])

        # If preprocessing was done earlier, allow using it for plotting or modeling.
        use_preprocessed = st.sidebar.checkbox("Use preprocessed data if available", value=True)

        if action == "Preprocessing":
            st.header("Data Preprocessing")
            preprocessed = preprocess_data(df)
            st.session_state.preprocessed_df = preprocessed
            st.subheader("Preprocessed Data Preview")
            st.dataframe(preprocessed.head().to_pandas())

        elif action == "Plotting":
            st.header("Data Visualization")
            # If preprocessed data is available and the user opts to use it.
            if use_preprocessed and st.session_state.preprocessed_df is not None:
                df = st.session_state.preprocessed_df
                st.info("Using preprocessed data for plotting.")
            # Define available plot types. You can add more plot functions here.
            plot_types = ["Line Plot", "Scatter Plot", "Bar Chart"]
            plot_type = st.selectbox("Select Plot Type", plot_types)
            cols = df.columns
            x_col = st.selectbox("Select X-axis column", cols)
            y_col = st.selectbox("Select Y-axis column", cols)
            if st.button("Generate Plot"):
                fig = get_plot(df, plot_type, x_col, y_col)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

        elif action == "ML Model":
            st.header("ML Model using GridSearchCV")
            # Use preprocessed data if available and opted.
            if use_preprocessed and st.session_state.preprocessed_df is not None:
                df = st.session_state.preprocessed_df
                st.info("Using preprocessed data for modeling.")
            cols = df.columns
            features = st.multiselect("Select Feature Columns", cols)
            target = st.selectbox("Select Target Column", cols)

            # Define available models and their GridSearchCV parameters.
            # Extend this dictionary to add more models or parameters.
            models_dict = {
                "Logistic Regression": {
                    "estimator": LogisticRegression(),
                    "param_grid": {"C": [0.1, 1, 10], "max_iter": [100, 200]},
                },
                "Random Forest": {
                    "estimator": RandomForestClassifier(),
                    "param_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
                },
            }

            model_choice = st.selectbox("Select Model", list(models_dict.keys()))
            if st.button("Run GridSearchCV"):
                if features and target:
                    best_model, best_params, best_score = run_grid_search(df, features, target, models_dict[model_choice])
                    st.subheader("Best Model & Parameters")
                    st.write("Best Model:", best_model)
                    st.write("Best Parameters:", best_params)
                    st.write("Best CV Score:", best_score)
                else:
                    st.error("Please select the feature(s) and target column.")

    else:
        st.info("Awaiting file upload...")

if __name__ == "__main__":
    main()
