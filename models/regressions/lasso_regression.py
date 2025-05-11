import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error


from analysis.model_metrics_container import ModelMetric, ModelMetricsContainer

st.title("Lasso Regression Analysis")

if "X_train" not in st.session_state:
    st.warning("Please split your data first using the 'Split Data' page!")
else:
    # Get data from session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    target_name = y_train.name

    st.session_state.task_type = "regression"  # Set task type to regression

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Manual Feature Selection", "GridSearchCV Tuning"])

    with tab1:
        st.header("Manual Feature Selection")
        
        # Feature selection
        available_features = X_train.columns.tolist()
        selected_features = st.multiselect(
            "Select Features (Manual Mode)",
            options=available_features,
            help="Select features for Lasso regression"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.number_input("Regularization Strength (α)", 
                                  min_value=0.001, 
                                  max_value=1000.0,
                                  value=1.0,
                                  step=0.1,
                                  format="%.3f")
        with col2:
            max_iter = st.number_input("Maximum Iterations", 
                                     min_value=100, 
                                     max_value=100000,
                                     value=1000)

        if st.button("Train Lasso Model") and selected_features:
            # Train model
            model = Lasso(alpha=alpha, max_iter=max_iter)
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
            st.subheader("Feature Coefficients")
            coeff_df = pd.DataFrame({
                "Feature": selected_features,
                "Coefficient": model.coef_
            })
            coeff_df = coeff_df[coeff_df['Coefficient'].abs() > 1e-6]  # Filter zero coefficients
            coeff_df.loc[len(coeff_df)] = ["Intercept", model.intercept_]

            # Save model metrics for comparison
            st.subheader("Save Model for Comparison")
            model_name = st.text_input("Model Name", f"Lasso Regression, {', '.join(selected_features)})")
            
            if st.button("Save Model Metrics"):
                # Initialize model_metrics in session state if it doesn't exist
                model_metrics = ModelMetric(model_name=model_name, metrics={
                    "Train R²": train_r2, "Test R²": test_r2,
                    "Train MSE": train_mse, "Test MSE": test_mse, "RMSE": np.sqrt(test_mse)})

                if "model_metrics" not in st.session_state:
                    st.session_state.model_metrics = ModelMetricsContainer(model_metrics)
                else:
                    st.session_state.model_metrics.append(model_metrics)
                st.success(f"Model '{model_name}' saved for comparison!")
                st.info("Go to the Model Comparison page to compare with other models.")
            
            if not coeff_df.empty:
                st.dataframe(coeff_df, hide_index=True)
                
                # Coefficient visualization
                fig = px.bar(
                    coeff_df[:-1],  # Exclude intercept
                    x='Feature',
                    y='Coefficient',
                    title='Feature Coefficients Magnitude'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("All coefficients shrunk to zero! Try reducing alpha.")

    with tab2:
        st.header("GridSearchCV Tuning")
        
        # Feature selection for GridSearch
        gs_features = st.multiselect(
            "Select Features (GridSearch Mode)",
            options=X_train.columns.tolist(),
            help="Select features for automated tuning"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            cv_folds = st.number_input("CV Folds", 
                                     min_value=2, 
                                     max_value=10,
                                     value=5)
        with col2:
            alphas_input = st.text_input("Alpha Values (comma separated)", 
                                       value="0.001,0.01,0.1,1.0,10.0,100.0")
            alphas = [float(a.strip()) for a in alphas_input.split(',') if a.strip()]

        if st.button("Run GridSearchCV") and gs_features and alphas:
            # Setup parameter grid
            param_grid = {'alpha': alphas}
            
            # Create and fit GridSearchCV
            model = Lasso(max_iter=10000)
            grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='r2')
            grid_search.fit(X_train[gs_features], y_train)

            # Get best model
            best_model = grid_search.best_estimator_
            
            # Display results
            st.subheader("Best Parameters")
            st.write(f"**Best Alpha:** {grid_search.best_params_['alpha']:.4f}")
            st.write(f"**Best R² Score (CV):** {grid_search.best_score_:.3f}")

            # Test set performance
            y_test_pred = best_model.predict(X_test[gs_features])
            test_r2 = r2_score(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            st.write(f"**Test R² Score:** {test_r2:.3f}")
            st.write(f"**Test MSE:** {test_mse:.3f}")

            st.subheader("Save Model for Comparison")
            model_name_gs = st.text_input("Model Name", f"Lasso Regression, {', '.join(selected_features)})")
            
            if st.button("Save Model Metrics"):
                # Initialize model_metrics in session state if it doesn't exist
                model_metrics = ModelMetric(model_name=model_name_gs, metrics={
                   "Test R²": test_r2, "Test MSE": test_mse, "RMSE": np.sqrt(test_mse)})

                if "model_metrics" not in st.session_state:
                    st.session_state.model_metrics = ModelMetricsContainer(model_metrics)
                else:
                    st.session_state.model_metrics.append(model_metrics)
                st.success(f"Model '{model_name_gs}' saved for comparison!")
                st.info("Go to the Model Comparison page to compare with other models.")

            # Validation curve visualization
            results = pd.DataFrame(grid_search.cv_results_)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['param_alpha'],
                y=results['mean_test_score'],
                error_y=dict(
                    type='data',
                    array=results['std_test_score'],
                    visible=True
                ),
                mode='markers+lines',
                name='CV Performance'
            ))
            fig.update_layout(
                title='Validation Curve',
                xaxis_title='Alpha (Log Scale)',
                yaxis_title='R² Score',
                xaxis_type='log'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            coeff_df = pd.DataFrame({
                'Feature': gs_features,
                'Coefficient': best_model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            fig2 = px.bar(
                coeff_df,
                x='Feature',
                y='Coefficient',
                title='Feature Importance from Best Model'
            )
            st.plotly_chart(fig2, use_container_width=True)

        elif not gs_features:
            st.error("Please select features for GridSearch!")
        elif not alphas:
            st.error("Please enter valid alpha values!")