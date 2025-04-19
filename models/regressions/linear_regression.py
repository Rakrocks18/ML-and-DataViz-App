import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

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
        # Train model
        model = LinearRegression()
        model.fit(X_train[selected_features], y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train[selected_features])
        y_test_pred = model.predict(X_test[selected_features])
        
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
        
        # Show coefficients
        st.subheader("Model Coefficients")
        coeff_df = pd.DataFrame({
            "Feature": selected_features,
            "Coefficient": model.coef_
        })
        coeff_df.loc[len(coeff_df)] = ["Intercept", model.intercept_]
        st.dataframe(coeff_df, hide_index=True)
        
        # Visualization section
        st.subheader("Visualization")
        
        if len(selected_features) == 1:  # Simple regression plot
            # Create figure
            fig = go.Figure()
            
            # Add training data
            fig.add_trace(go.Scatter(
                x=X_train[selected_features[0]],
                y=y_train,
                mode='markers',
                name='Training Data',
                marker=dict(color='blue', opacity=0.5)
            ))
            
            # Add test data
            fig.add_trace(go.Scatter(
                x=X_test[selected_features[0]],
                y=y_test,
                mode='markers',
                name='Test Data',
                marker=dict(color='red', opacity=0.5)
            ))
            
            # Add regression line
            x_line = np.linspace(
                X_train[selected_features[0]].min(),
                X_train[selected_features[0]].max(),
                100
            )
            y_line = model.predict(x_line.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name='Regression Line',
                line=dict(color='black', width=3)
            ))
            
            fig.update_layout(
                title=f"{target_name} vs {selected_features[0]}",
                xaxis_title=selected_features[0],
                yaxis_title=target_name,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Multiple regression visualization
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
            
            # Feature importance
            fig2 = px.bar(
                x=model.coef_,
                y=selected_features,
                orientation='h',
                labels={'x': 'Coefficient Value', 'y': 'Feature'},
                title='Feature Coefficients'
            )
            st.plotly_chart(fig2, use_container_width=True)