import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer

st.title("Polynomial Regression Analysis")

if "X_train" not in st.session_state:
    st.warning("Please split your data first using the 'Split Data' page!")
else:
    # Get data from session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    target_name = y_train.name

    # Create tabs
    tab1, tab2 = st.tabs(["Manual Mode", "GridSearchCV"])

    with tab1:
        # Feature and degree selection
        col1, col2 = st.columns(2)
        with col1:
            available_features = X_train.columns.tolist()
            selected_features = st.multiselect(
                "Select Features for Regression",
                options=available_features,
                help="Choose 1 feature for polynomial regression or multiple for multivariate"
            )
        with col2:
            degree = st.number_input("Polynomial Degree", 
                                    min_value=1, 
                                    max_value=5, 
                                    value=2,
                                    help="Higher degrees may cause overfitting!")

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
                # Align train and test encoded features
                X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='outer', axis=1, fill_value=0)
            else:
                X_train_cat = pd.DataFrame(index=X_train.index)
                X_test_cat = pd.DataFrame(index=X_test.index)
            
            # Apply polynomial features to numerical features
            if numerical_features:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_train_num_poly = poly.fit_transform(X_train[numerical_features])
                X_test_num_poly = poly.transform(X_test[numerical_features])
                feature_names_num = poly.get_feature_names_out(numerical_features)
                X_train_num_poly_df = pd.DataFrame(X_train_num_poly, columns=feature_names_num, index=X_train.index)
                X_test_num_poly_df = pd.DataFrame(X_test_num_poly, columns=feature_names_num, index=X_test.index)
            else:
                X_train_num_poly_df = pd.DataFrame(index=X_train.index)
                X_test_num_poly_df = pd.DataFrame(index=X_test.index)
            
            # Combine numerical polynomial and categorical encoded features
            X_train_final = pd.concat([X_train_num_poly_df, X_train_cat], axis=1)
            X_test_final = pd.concat([X_test_num_poly_df, X_test_cat], axis=1)
            
            # Train model
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
            
            # Save model metrics for comparison
            st.subheader("Save Model for Comparison")
            model_name = st.text_input("Model Name", f"Polynomial Regression (Degree {degree}, {', '.join(selected_features)})")
            
            if st.button("Save Model Metrics"):
                # Initialize model_metrics in session state if it doesn't exist
                if "model_metrics" not in st.session_state:
                    st.session_state.model_metrics = {}
                
                # Set task type for proper comparison
                st.session_state.task_type = "regression"
                
                # Save metrics to session state
                st.session_state.model_metrics[model_name] = {
                    "Test R²": test_r2,
                    "Train R²": train_r2,
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "RMSE": np.sqrt(test_mse)
                }
                
                st.success(f"Model '{model_name}' saved for comparison!")
                st.info("Go to the Model Comparison page to compare with other models.")
            
            # Show coefficients
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
                # Single numerical feature: plot polynomial curve
                feature = numerical_features[0]
                x_curve = np.linspace(
                    X_train[feature].min(),
                    X_train[feature].max(),
                    300
                ).reshape(-1, 1)
                
                # Transform x_curve to polynomial features
                x_curve_poly = poly.transform(x_curve)
                x_curve_poly_df = pd.DataFrame(x_curve_poly, columns=feature_names_num)
                
                # Since no categorical features, proceed directly
                y_curve = model.predict(x_curve_poly_df)
                
                fig = go.Figure()
                # Training data
                fig.add_trace(go.Scatter(
                    x=X_train[feature],
                    y=y_train,
                    mode='markers',
                    name='Training Data',
                    marker=dict(color='blue', opacity=0.5)
                ))
                
                # Test data
                fig.add_trace(go.Scatter(
                    x=X_test[feature],
                    y=y_test,
                    mode='markers',
                    name='Test Data',
                    marker=dict(color='red', opacity=0.5)
                ))
                
                # Polynomial curve
                fig.add_trace(go.Scatter(
                    x=x_curve.flatten(),
                    y=y_curve,
                    mode='lines',
                    name=f'Degree {degree} Fit',
                    line=dict(color='black', width=3)
                ))
                
                fig.update_layout(
                    title=f"{target_name} vs {feature} (Degree {degree})",
                    xaxis_title=feature,
                    yaxis_title=target_name
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Multiple features or categorical features: show residual and coefficient plots
                residuals = y_test - y_test_pred
                fig1 = px.scatter(
                    x=y_test_pred,
                    y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title='Residual Plot'
                )
                fig1.add_hline(y=0, line_dash="dot", line_color="red")
                st.plotly_chart(fig1, use_container_width=True)
                
                fig2 = px.bar(
                    x=coeff_df["Coefficient"][:-1],
                    y=coeff_df["Feature"][:-1],
                    orientation='h',
                    labels={'x': 'Coefficient Value', 'y': 'Feature'},
                    title='Polynomial Feature Coefficients'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Model complexity warning
            if degree > 3 and test_r2 < train_r2:
                st.warning("Warning: Potential overfitting detected! Test performance is worse than training performance.")
    
    with tab2:
        st.header("Automated Polynomial Degree Selection")
        
        selected_features = st.multiselect(
            "Select Features for Polynomial Regression",
            options=X_train.columns.tolist()
        )
        
        if selected_features:
            # Identify categorical and numerical features
            categorical_features = [feat for feat in selected_features if X_train[feat].dtype == 'object']
            numerical_features = [feat for feat in selected_features if feat not in categorical_features]
            
            # Define ColumnTransformer
            col_transformer = ColumnTransformer(
                transformers=[
                    ('num', PolynomialFeatures(), numerical_features),
                    ('cat', OneHotEncoder(), categorical_features)
                ]
            )
            
            # Create pipeline
            pipeline = Pipeline([
                ('col_trans', col_transformer),
                ('linear', LinearRegression())
            ])
            
            # Parameter grid
            param_grid = {
                'col_trans__num__degree': list(range(1, 6)),
                'col_trans__num__interaction_only': [True, False]
            }
            
            # GridSearchCV setup
            gs = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            
            if st.button("Run GridSearchCV"):
                gs.fit(X_train[selected_features], y_train)
                
                # Best model results
                best_degree = gs.best_params_['col_trans__num__degree']
                best_interaction = gs.best_params_['col_trans__num__interaction_only']
                
                # Test set performance
                y_pred = gs.predict(X_test[selected_features])
                test_r2 = r2_score(y_test, y_pred)
                test_mse = mean_squared_error(y_test, y_pred)
                train_r2 = r2_score(y_train, gs.predict(X_train[selected_features]))
                train_mse = mean_squared_error(y_train, gs.predict(X_train[selected_features]))
                
                st.subheader("Best Parameters")
                st.write(f"**Optimal Degree:** {best_degree}")
                st.write(f"**Interaction Only:** {best_interaction}")
                st.write(f"**Best R² Score (CV):** {gs.best_score_:.3f}")
                st.write(f"**Test R²:** {test_r2:.3f}")
                st.write(f"**Test MSE:** {test_mse:.3f}")
                
                # Save model metrics for comparison
                st.subheader("Save Model for Comparison")
                model_name_gs = st.text_input("Model Name (GridSearch)", f"Polynomial GridSearch (Degree {best_degree}, {', '.join(selected_features)})")
                
                if st.button("Save GridSearch Model Metrics"):
                    # Initialize model_metrics in session state if it doesn't exist
                    if "model_metrics" not in st.session_state:
                        st.session_state.model_metrics = {}
                    
                    # Set task type for proper comparison
                    st.session_state.task_type = "regression"
                    
                    # Save metrics to session state
                    st.session_state.model_metrics[model_name_gs] = {
                        "Test R²": test_r2,
                        "Train R²": train_r2,
                        "Test MSE": test_mse,
                        "Train MSE": train_mse,
                        "RMSE": np.sqrt(test_mse)
                    }
                    
                    st.success(f"Model '{model_name_gs}' saved for comparison!")
                    st.info("Go to the Model Comparison page to compare with other models.")
                
                # Validation curve visualization
                results = pd.DataFrame(gs.cv_results_)
                fig = px.line(
                    results[results['param_col_trans__num__interaction_only'] == best_interaction],
                    x='param_col_trans__num__degree',
                    y='mean_test_score',
                    error_y='std_test_score',
                    title='Validation Curve for Polynomial Degree'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance visualization
                best_col_trans = gs.best_estimator_.named_steps['col_trans']
                feature_names = best_col_trans.get_feature_names_out()
                coefficients = gs.best_estimator_.named_steps['linear'].coef_
                
                fig2 = px.bar(
                    x=feature_names,
                    y=coefficients,
                    labels={'x': 'Features', 'y': 'Coefficient Value'},
                    title='Feature Coefficients'
                )
                st.plotly_chart(fig2, use_container_width=True)