from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import polars as pl
import streamlit as st

def run_grid_search(X, y, model_info: dict, cv: int = 3):
    """
    Run GridSearchCV on the provided dataset.
    'model_info' is a dictionary containing the sklearn estimator and parameter grid.
    """
    grid_search = GridSearchCV(model_info["estimator"], model_info["param_grid"], cv=cv)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

st.subheader("Logistic Regression Model")

df = st.session_state.df

if df is not None:
    # Let the user choose features and target column
    cols = df.columns
    features = st.multiselect("Select Feature Columns", cols)
    target = st.selectbox("Select Target Column", cols)

    # Convert chosen columns to numpy arrays
    X = df[features].to_numpy()
    y = df[target].to_numpy()

    # Test-train split parameters
    test_size = st.number_input("Enter Test Size", min_value=0.05, max_value=1.0, value=0.1, step=0.05, format="%.2f")
    random_state = st.number_input("Enter Random State", min_value=0, max_value=100, value=42)
    shuffle = st.checkbox("Shuffle Data", value=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, shuffle=shuffle)

    # Choose between Manual and GridSearch CV options
    tuning_method = st.radio("Choose Parameter Tuning Method", ["Manual", "Grid Search CV"])

    if tuning_method == "Manual":
        st.markdown("### Manual Parameter Entry")
        # Let the user input hyperparameters manually
        C = st.number_input("Regularization Strength (C)", value=1.0, step=0.1)
        max_iter = st.number_input("Maximum Iterations", value=100, step=10)

        if st.button("Train Model Manually"):
            # Create and train the model using manual parameters
            model = LogisticRegression(C=C, max_iter=int(max_iter))
            model.fit(X_train, y_train)
            st.success("Model trained successfully with manual parameters.")

            # Model Evaluation
            st.subheader("Model Evaluation")
            y_pred = model.predict(X_test)
            pred_df = pl.DataFrame({
                "Actual Data": y_test,
                "Prediction": y_pred
            })
            st.dataframe(pred_df.head(20).to_pandas())

            st.markdown("### Classification Report")
            report = classification_report(y_test, y_pred)
            st.markdown(f"```\n{report}\n```")

            st.markdown("### Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            st.markdown(f"```\n{conf_matrix}\n```")

    elif tuning_method == "Grid Search CV":
        st.markdown("### Grid SearchCV Parameters")
        cv = st.slider("Number of CV folds", min_value=2, max_value=10, value=3, step=1)
        # Define parameter grid options (you can add more parameters as needed)
        models_dict = {
            "estimator": LogisticRegression(),
            "param_grid": {"C": [0.1, 1, 10], "max_iter": [100, 200]},
        }
        if st.button("Train with Grid Search CV"):
            best_model, best_params, best_score = run_grid_search(X_train, y_train, models_dict, cv)
            st.subheader("Best Model & Parameters")
            st.write("Best Model:", best_model)
            st.write("Best Parameters:", best_params)
            st.write("Best CV Score:", best_score)

            # Model Evaluation
            y_pred = best_model.predict(X_test)
            pred_df = pl.DataFrame({
                "Actual Data": y_test,
                "Prediction": y_pred
            })
            st.dataframe(pred_df.head(20).to_pandas())

            st.markdown("### Classification Report")
            report = classification_report(y_test, y_pred)
            st.markdown(f"```\n{report}\n```")

            st.markdown("### Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            st.markdown(f"```\n{conf_matrix}\n```")

