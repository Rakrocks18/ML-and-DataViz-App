import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from hmmlearn import hmm
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# Define the models dictionary at module level
models_dict = {
    "Regression Models": {
        "Lasso Regression": {
            "estimator": Lasso(),
            "param_grid": {
                "alpha": [0.1, 1.0, 10.0],
                "max_iter": [1000]
            },
            "type": "regression"
        },
        "Random Forest Regressor": {
            "estimator": RandomForestRegressor(),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]
            },
            "type": "regression"
        },
        "SVR": {
            "estimator": SVR(),
            "param_grid": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"]
            },
            "type": "regression"
        },
        "KNN Regressor": {
            "estimator": KNeighborsRegressor(),
            "param_grid": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            },
            "type": "regression"
        }
    },
    "Classification Models": {
        "Logistic Regression": {
            "estimator": LogisticRegression(),
            "param_grid": {
                "C": [0.1, 1, 10],
                "max_iter": [1000]
            },
            "type": "classification"
        },
        "Random Forest Classifier": {
            "estimator": RandomForestClassifier(),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]
            },
            "type": "classification"
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(),
            "param_grid": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10]
            },
            "type": "classification"
        },
        "XGBoost": {
            "estimator": xgb.XGBClassifier(),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1]
            },
            "type": "classification"
        },
        "SVM": {
            "estimator": SVC(),
            "param_grid": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"]
            },
            "type": "classification"
        },
        "KNN Classifier": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            },
            "type": "classification"
        }
    },
    "Dimensionality Reduction": {
        "PCA": {
            "estimator": PCA(n_components=0.95),
            "param_grid": {},
            "type": "dimensionality_reduction"
        }
    },
    "Sequential Learning": {
        "HMM": {
            "estimator": hmm.GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=100,
                random_state=42
            ),
            "param_grid": {
                "n_components": [2, 3, 4],
                "covariance_type": ["full", "tied", "diag", "spherical"],
                "n_iter": [50, 100, 200]
            },
            "type": "sequential"
        }
    }
}

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
                df = df.with_columns(pl.col(col).cast(pl.Float64))
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
    Run GridSearchCV on the provided dataset with proper preprocessing of categorical features.
    'model_info' is a dictionary containing the sklearn estimator and parameter grid.
    """
    try:
        # Convert Polars DataFrame to pandas for scikit-learn
        pdf = df.to_pandas()
        X = pdf[features].copy()
        y = pdf[target].copy()
        
        # Handle categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        if model_info.get("type") == "sequential":
            # Handle sequential models (HMM)
            # Reshape data for HMM (samples, timesteps, features)
            X_reshaped = X.reshape(-1, 1, X.shape[1])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_reshaped, y, test_size=0.2, random_state=42
            )
            
            # Create and train model
            model = model_info["estimator"]
            model.fit(X_train.reshape(-1, X_train.shape[-1]))
            
            # Get predictions and probabilities
            hidden_states = model.predict(X_test.reshape(-1, X_test.shape[-1]))
            log_likelihood = model.score(X_test.reshape(-1, X_test.shape[-1]))
            
            # Calculate metrics
            mse = mean_squared_error(y_test, hidden_states)
            r2 = r2_score(y_test, hidden_states)
            
            # Format results
            best_params = {
                "n_components": model.n_components,
                "covariance_type": model.covariance_type,
                "n_iter": model.n_iter
            }
            
            # Calculate normalized score
            max_log_likelihood = 0  # theoretical maximum
            min_log_likelihood = -1000  # reasonable minimum
            normalized_score = (log_likelihood - min_log_likelihood) / (max_log_likelihood - min_log_likelihood)
            normalized_score = max(0, min(1, normalized_score))  # clip between 0 and 1
            
            return model, best_params, normalized_score
            
        elif model_info.get("type") == "dimensionality_reduction":
            # Handle PCA
            model = model_info["estimator"]
            model.fit(X)
            explained_variance = model.explained_variance_ratio_
            return model, {"n_components": model.n_components_}, sum(explained_variance)
        else:
            # Regular supervised learning models
            # Calculate appropriate number of CV splits
            min_samples = min(np.bincount(y)) if model_info.get("type") == "classification" else len(y) // 2
            max_splits = min_samples // 2
            n_splits = max(2, min(3, max_splits))
            
            grid_search = GridSearchCV(model_info["estimator"], model_info["param_grid"], cv=n_splits)
            grid_search.fit(X, y)
            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
            
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        st.info("Try selecting different features or adjusting the model parameters.")
        return None, None, None

def run_q_learning(X, y, model_info, episodes=1000):
    """
    Simple Q-learning implementation for tabular data
    """
    n_states = len(X)
    n_actions = len(np.unique(y))
    Q = np.zeros((n_states, n_actions))
    
    # Q-learning parameters
    alpha = 0.1  # learning rate
    gamma = 0.95  # discount factor
    epsilon = 0.1  # exploration rate
    
    for episode in range(episodes):
        state = np.random.randint(n_states)
        done = False
        
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])
            
            next_state = (state + 1) % n_states
            reward = 1 if action == y[state] else 0
            
            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            
            state = next_state
            done = state == n_states - 1
    
    # Evaluate performance
    predictions = np.array([np.argmax(Q[i]) for i in range(n_states)])
    accuracy = accuracy_score(y, predictions)
    
    return Q, {"episodes": episodes, "alpha": alpha, "gamma": gamma}, accuracy

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
            st.header("ML Model Selection and Training")
            
            # First select the model category
            model_category = st.selectbox("Select Model Category", list(models_dict.keys()))
            
            # Then select the specific model from that category
            model_choice = st.selectbox("Select Model", list(models_dict[model_category].keys()))
            
            cols = df.columns
            features = st.multiselect("Select Feature Columns", cols)
            target = st.selectbox("Select Target Column", cols)

            if st.button("Train Model"):
                if features and target:
                    model_info = models_dict[model_category][model_choice]
                    best_model, best_params, best_score = run_grid_search(df, features, target, model_info)
                    
                    if best_model is not None:
                        st.subheader("Model Results")
                        st.write("Best Model:", best_model)
                        st.write("Best Parameters:", best_params)
                        st.write("Best Score:", best_score)
                        
                        # Additional information for PCA
                        if model_choice == "PCA":
                            st.write("Explained Variance Ratio:", best_model.explained_variance_ratio_)
                            st.line_chart(np.cumsum(best_model.explained_variance_ratio_))
                else:
                    st.error("Please select feature(s) and target column.")

    else:
        st.info("Awaiting file upload...")

if __name__ == "__main__":
    main()
