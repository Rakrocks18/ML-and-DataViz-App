import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from analysis.model_metrics_container import ModelMetric, ModelMetricsContainer

st.title("Linear Regression Analysis")

if "X_train" not in st.session_state:
    st.warning("Please split your data first using the 'Split Data' page!")
else:
    # Get data from session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    target_name = y_train.name

    # Feature selection
    available_features = X_train.columns.tolist()
    selected_features = st.multiselect(
        "Select Features for Regression",
        options=available_features,
        help="Choose 1 feature for simple linear regression or multiple for multiple regression"
    )
    
    if not selected_features:
        st.error("Please select at least one feature!")
    else:
        # Identify categorical and numerical features
        categorical_features = [feat for feat in selected_features if X_train[feat].dtype == 'object']
        numerical_features = [feat for feat in selected_features if feat not in categorical_features]
        
        # Handle categorical features with one-hot encoding
        if categorical_features:
            X_train_cat = pd.get_dummies(X_train[categorical_features], prefix=categorical_features)
            X_test_cat = pd.get_dummies(X_test[categorical_features], prefix=categorical_features)
            # Align train and test encoded features to ensure consistent columns
            X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='outer', axis=1, fill_value=0)
        else:
            X_train_cat = pd.DataFrame(index=X_train.index)
            X_test_cat = pd.DataFrame(index=X_test.index)
        
        # Combine numerical and encoded categorical features
        X_train_final = pd.concat([X_train[numerical_features], X_train_cat], axis=1)
        X_test_final = pd.concat([X_test[numerical_features], X_test_cat], axis=1)
        
        # Train model with the processed features
        model = LinearRegression()
        model.fit(X_train_final, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_final)
        y_test_pred = model.predict(X_test_final)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        


        # Display metrics
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Training R² Score", f"{train_r2:.3f}")
        col2.metric("Testing R² Score", f"{test_r2:.3f}")
        col1.metric("Training MSE", f"{train_mse:.3f}")
        col2.metric("Testing MSE", f"{test_mse:.3f}")
        
        # Model comparison functionality (your addition)
        st.subheader("Save Model for Comparison")
        model_name = st.text_input("Model Name", f"Linear Regression ({', '.join(selected_features)})")
        
        if st.button("Save Model Metrics"):
            model_metrics = ModelMetric(model_name=model_name, metrics={
            "Train R²": train_r2, "Test R²": test_r2,
            "Train MSE": train_mse, "Test MSE": test_mse, "RMSE": np.sqrt(test_mse)})

            if "model_metrics" not in st.session_state:
                st.session_state.model_metrics = ModelMetricsContainer(model_metrics)
            else:
                st.session_state.model_metrics.append(model_metrics)
            
            st.success(f"Model '{model_name}' saved for comparison!")
            st.info("Go to the Model Comparison page to compare with other models.")
        
        # Show coefficients using the final feature set
        st.subheader("Model Coefficients")
        coeff_df = pd.DataFrame({
            "Feature": X_train_final.columns,
            "Coefficient": model.coef_
        })
        coeff_df.loc[len(coeff_df)] = ["Intercept", model.intercept_]
        st.dataframe(coeff_df, hide_index=True)
        
        # Visualization section
        st.subheader("Visualization")
        
        if len(numerical_features) == 1 and len(categorical_features) == 0:
            # Simple regression plot (only for a single numerical feature)
            feature = numerical_features[0]
            fig = go.Figure()
            
            # Add training data
            fig.add_trace(go.Scatter(
                x=X_train[feature],
                y=y_train,
                mode='markers',
                name='Training Data',
                marker=dict(color='blue', opacity=0.5)
            ))
            
            # Add test data
            fig.add_trace(go.Scatter(
                x=X_test[feature],
                y=y_test,
                mode='markers',
                name='Test Data',
                marker=dict(color='red', opacity=0.5)
            ))
            
            # Add regression line
            x_line = np.linspace(X_train[feature].min(), X_train[feature].max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))  # Works since only one numerical feature
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name='Regression Line',
                line=dict(color='black', width=3)
            ))
            
            fig.update_layout(
                title=f"{target_name} vs {feature}",
                xaxis_title=feature,
                yaxis_title=target_name,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Multiple regression visualization (for multiple features or any categorical features)
            # Residual plot
            residuals = y_test - y_test_pred
            fig1 = px.scatter(
                x=y_test_pred,
                y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title='Residual Plot'
            )
            fig1.add_hline(y=0, line_dash="dot", line_color="red")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Feature coefficients
            fig2 = px.bar(
                x=model.coef_,
                y=X_train_final.columns,
                orientation='h',
                labels={'x': 'Coefficient Value', 'y': 'Feature'},
                title='Feature Coefficients'
            )
            st.plotly_chart(fig2, use_container_width=True)