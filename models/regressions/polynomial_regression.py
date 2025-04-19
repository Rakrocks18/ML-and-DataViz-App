import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

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
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train[selected_features])
            X_test_poly = poly.transform(X_test[selected_features])
            feature_names = poly.get_feature_names_out(selected_features)

            # Train model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_poly)
            y_test_pred = model.predict(X_test_poly)

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
                "Feature": feature_names,
                "Coefficient": model.coef_
            })
            coeff_df.loc[len(coeff_df)] = ["Intercept", model.intercept_]
            st.dataframe(coeff_df, hide_index=True)

            # Visualization section
            st.subheader("Visualization")
            
            if len(selected_features) == 1:  # Single feature visualization
                # Generate smooth curve
                x_curve = np.linspace(
                    X_train[selected_features[0]].min(),
                    X_train[selected_features[0]].max(),
                    300
                ).reshape(-1, 1)
                
                x_curve_poly = poly.transform(x_curve)
                y_curve = model.predict(x_curve_poly)

                fig = go.Figure()
                # Training data
                fig.add_trace(go.Scatter(
                    x=X_train[selected_features[0]],
                    y=y_train,
                    mode='markers',
                    name='Training Data',
                    marker=dict(color='blue', opacity=0.5)
                ))
                
                # Test data
                fig.add_trace(go.Scatter(
                    x=X_test[selected_features[0]],
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
                    title=f"{target_name} vs {selected_features[0]} (Degree {degree})",
                    xaxis_title=selected_features[0],
                    yaxis_title=target_name
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Multiple features visualization
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

                # Coefficient plot
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
            # Create pipeline
            pipeline = Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LinearRegression())
            ])

            # Parameter grid
            param_grid = {
                'poly__degree': list(range(1, 6)),
                'poly__interaction_only': [True, False]
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
                best_degree = gs.best_params_['poly__degree']
                best_interaction = gs.best_params_['poly__interaction_only']
                
                st.subheader("Best Parameters")
                st.write(f"**Optimal Degree:** {best_degree}")
                st.write(f"**Interaction Only:** {best_interaction}")
                st.write(f"**Best R² Score (CV):** {gs.best_score_:.3f}")

                # Test set performance
                y_pred = gs.predict(X_test[selected_features])
                test_r2 = r2_score(y_test, y_pred)
                test_mse = mean_squared_error(y_test, y_pred)
                st.write(f"**Test R²:** {test_r2:.3f}")
                st.write(f"**Test MSE:** {test_mse:.3f}")

                # Validation curve visualization
                results = pd.DataFrame(gs.cv_results_)
                fig = px.line(
                    results[results['param_poly__interaction_only'] == best_interaction],
                    x='param_poly__degree',
                    y='mean_test_score',
                    error_y='std_test_score',
                    title='Validation Curve for Polynomial Degree'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance visualization
                poly = gs.best_estimator_.named_steps['poly']
                feature_names = poly.get_feature_names_out(selected_features)
                coefficients = gs.best_estimator_.named_steps['linear'].coef_
                
                fig2 = px.bar(
                    x=feature_names,
                    y=coefficients,
                    labels={'x': 'Features', 'y': 'Coefficient Value'},
                    title='Feature Coefficients'
                )
                st.plotly_chart(fig2, use_container_width=True)