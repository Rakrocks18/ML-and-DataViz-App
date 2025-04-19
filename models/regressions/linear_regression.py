import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

def linear_regression_page():
    st.title("Linear Regression")
    
    if "df" not in st.session_state:
        st.error("Please upload data first.")
        return
    
    df = st.session_state.df
    
    # Feature selection
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) == 0:
        st.error("No numeric columns found in the dataset. Linear regression requires numeric data.")
        return
        
    features = st.multiselect("Select features (numeric only)", numeric_columns)
    target = st.selectbox("Select target variable (numeric only)", numeric_columns)
    
    if not features or not target:
        st.warning("Please select both features and target variable to proceed.")
        return
        
    if target in features:
        features.remove(target)
        st.info(f"Removed {target} from features as it's the target variable.")
    
    if features and target:
        # Prepare the data
        X = df[features].values
        y = df[target].values
        
        # Model parameters
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", min_value=0, max_value=999, value=42)
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Train model button
            if st.button("Train Model"):
                # Create and train the model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate scores
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Display results
                st.subheader("Model Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training R² Score", f"{train_score:.4f}")
                with col2:
                    st.metric("Testing R² Score", f"{test_score:.4f}")
                
                # Display coefficients
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': model.coef_
                })
                st.dataframe(coef_df)
                
                # Plot actual vs predicted
                fig = px.scatter(x=y_test, y=test_pred, 
                               labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                               title='Actual vs Predicted Values (Test Set)')
                fig.add_scatter(x=[y_test.min(), y_test.max()], 
                              y=[y_test.min(), y_test.max()],
                              name='Perfect Prediction',
                              line=dict(dash='dash'))
                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please make sure all selected features are numeric and contain valid data.")

