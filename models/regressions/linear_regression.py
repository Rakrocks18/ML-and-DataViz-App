from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

import streamlit as st

import polars as pl

st.subheader("Linear Regression Model")
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    if st.button("Train Model"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        st.success("Model trained successfully.")
        
        st.write("Model Coefficients: ", model.coef_.reshape(1, -1))
        st.write("Model Intercept: ", model.intercept_)
        
        st.subheader("Model Evaluation")
                
        y_pred = model.predict(X_test)
        pred_df = pl.DataFrame({
            "Actual Data": y_test,
            "Prediction": y_pred 
        })
        st.dataframe(pred_df.head(20).to_pandas())

        st.write("Test Score: ")
        st.write("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
        st.write("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

