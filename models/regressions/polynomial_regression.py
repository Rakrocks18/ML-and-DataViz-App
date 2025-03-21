from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import polars as pl
import numpy as np

st.subheader("Polynomial Regression Model")
df = st.session_state.df

if df is not None:
    cols = df.columns
    features = st.multiselect("Select Feature Columns", cols)
    target = st.selectbox("Select Target Column", cols)

    X = df[features].to_numpy()
    y = df[target].to_numpy()

    test_size = st.number_input("Enter Test Size", min_value=0.05, max_value=1.0, value=0.1, step=0.05, format="%.2f")
    random_state = st.number_input("Enter Random State", min_value=0, max_value=100)
    shuffle = st.checkbox("Shuffle Data", value=True)

    degree_option = st.radio("Choose Polynomial Degree Mode", ("Manual", "Grid Search"))
    
    if degree_option == "Manual":
        degree = st.number_input("Enter Polynomial Degree", min_value=1, max_value=10, value=2, step=1)
    else:
        min_degree = st.number_input("Min Degree", min_value=1, max_value=10, value=1, step=1)
        max_degree = st.number_input("Max Degree", min_value=1, max_value=10, value=5, step=1)
        cv_folds = st.number_input("Cross-Validation Folds", min_value=2, max_value=10, value=5, step=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    if st.button("Train Model"):
        if degree_option == "Manual":
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X_train, y_train)
        else:
            param_grid = {"polynomialfeatures__degree": np.arange(min_degree, max_degree + 1)}
            pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            degree = grid_search.best_params_["polynomialfeatures__degree"]
            st.write(f"Optimal Polynomial Degree: {degree}")

        st.success("Model trained successfully.")
        
        st.write("Model Coefficients: ", model.named_steps["linearregression"].coef_.reshape(1, -1))
        st.write("Model Intercept: ", model.named_steps["linearregression"].intercept_)
        
        st.subheader("Model Evaluation")
        y_pred = model.predict(X_test)
        pred_df = pl.DataFrame({"Actual Data": y_test, "Prediction": y_pred})
        st.dataframe(pred_df.head(20).to_pandas())

        st.write("Test Score: ")
        st.write("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
        st.write("Mean Squared Error: ", mean_squared_error(y_test, y_pred))