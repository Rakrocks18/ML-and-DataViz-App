import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

def lasso_regression_page():
    st.title("Lasso Regression")
    
    if "df" not in st.session_state:
        st.error("Please upload data first.")
        return
    
    df = st.session_state.df
    
    # Feature selection
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) == 0:
        st.error("No numeric columns found in the dataset. Lasso regression requires numeric data.")
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
        alpha = st.slider("Alpha (Regularization Strength)", 0.0001, 1.0, 0.01, format="%.4f")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", min_value=0, max_value=999, value=42)
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Train model button
            if st.button("Train Model"):
                # Create and train the model
                model = Lasso(alpha=alpha, random_state=random_state)
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate scores
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                # Display results
                st.subheader("Model Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training R² Score", f"{train_score:.4f}")
                    st.metric("Training MSE", f"{train_mse:.4f}")
                with col2:
                    st.metric("Testing R² Score", f"{test_score:.4f}")
                    st.metric("Testing MSE", f"{test_mse:.4f}")
                
                # Display coefficients
                st.subheader("Feature Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', ascending=False)
                
                # Coefficient plot
                fig_coef = px.bar(coef_df, x='Feature', y='Coefficient',
                                title='Lasso Regression Coefficients',
                                labels={'Coefficient': 'Coefficient Value'})
                st.plotly_chart(fig_coef)
                
                # Actual vs Predicted Plot
                fig_pred = go.Figure()
                
                # Training data
                fig_pred.add_trace(go.Scatter(
                    x=y_train,
                    y=train_pred,
                    mode='markers',
                    name='Training Data',
                    marker=dict(color='blue', size=8, opacity=0.6)
                ))
                
                # Testing data
                fig_pred.add_trace(go.Scatter(
                    x=y_test,
                    y=test_pred,
                    mode='markers',
                    name='Testing Data',
                    marker=dict(color='red', size=8, opacity=0.6)
                ))
                
                # Perfect prediction line
                min_val = min(min(y_train), min(y_test))
                max_val = max(max(y_train), max(y_test))
                fig_pred.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig_pred.update_layout(
                    title='Actual vs Predicted Values',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values'
                )
                st.plotly_chart(fig_pred)
                
                # Residuals Plot
                residuals_train = y_train - train_pred
                residuals_test = y_test - test_pred
                
                fig_resid = go.Figure()
                
                # Training residuals
                fig_resid.add_trace(go.Scatter(
                    x=train_pred,
                    y=residuals_train,
                    mode='markers',
                    name='Training Residuals',
                    marker=dict(color='blue', size=8, opacity=0.6)
                ))
                
                # Testing residuals
                fig_resid.add_trace(go.Scatter(
                    x=test_pred,
                    y=residuals_test,
                    mode='markers',
                    name='Testing Residuals',
                    marker=dict(color='red', size=8, opacity=0.6)
                ))
                
                fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig_resid.update_layout(
                    title='Residuals Plot',
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals'
                )
                st.plotly_chart(fig_resid)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please make sure all selected features are numeric and contain valid data.") 