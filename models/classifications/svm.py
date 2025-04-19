import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance

st.title("Support Vector Machine Analysis")

if "X_train" not in st.session_state:
    st.warning("Please split your data first using the 'Split Data' page!")
else:
    # Get data from session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    
    # Determine problem type
    problem_type = 'classification' if y_train.nunique() < 10 else 'regression'
    
    tab1, tab2 = st.tabs(["Manual Configuration", "GridSearchCV Tuning"])

    with tab1:
        st.header("Manual Configuration")
        selected_features = st.multiselect("Select Features (Max 2 for visualization)", 
                                         X_train.columns)
        
        if selected_features:
            col1, col2 = st.columns(2)
            with col1:
                kernel = st.selectbox("Kernel Type", ['linear', 'rbf', 'poly', 'sigmoid'])
                C = st.number_input("Regularization (C)", 0.01, 100.0, 1.0, 0.1)
                gamma = st.selectbox("Gamma", ['scale', 'auto']) 
                if kernel == 'poly':
                    degree = st.number_input("Polynomial Degree", 1, 10, 3)
            with col2:
                if kernel in ['rbf', 'poly', 'sigmoid']:
                    gamma = st.selectbox("Gamma Type", ['scale', 'auto', 'custom'])
                    if gamma == 'custom':
                        gamma = st.number_input("Custom Gamma Value", 0.0001, 10.0, 0.1)
                if problem_type == 'regression':
                    epsilon = st.number_input("Epsilon", 0.01, 1.0, 0.1)

            if st.button("Train SVM Model"):
                if problem_type == 'classification':
                    model = SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        degree=degree if kernel == 'poly' else 3,
                        probability=True
                    )
                else:
                    model = SVR(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        degree=degree if kernel == 'poly' else 3,
                        epsilon=epsilon
                    )
                
                model.fit(X_train[selected_features], y_train)
                y_pred = model.predict(X_test[selected_features])
                
                # Performance metrics
                st.subheader("Model Performance")
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.3f}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual"),
                        x=model.classes_,
                        y=model.classes_,
                        text_auto=True
                    )
                    st.plotly_chart(fig_cm)

                    # Decision Boundary Visualization
                    if len(selected_features) == 2:
                        st.subheader("Decision Boundaries")
                        
                        # Create mesh grid
                        x_min, x_max = X_train[selected_features[0]].min()-1, X_train[selected_features[0]].max()+1
                        y_min, y_max = X_train[selected_features[1]].min()-1, X_train[selected_features[1]].max()+1
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                           np.linspace(y_min, y_max, 100))
                        
                        # Predict probabilities
                        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                        Z = Z.reshape(xx.shape)
                        
                        # Create plot
                        fig = px.scatter(
                            X_test, 
                            x=selected_features[0], 
                            y=selected_features[1], 
                            color=y_pred.astype(str),
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        fig.add_contour(
                            x=xx[0], 
                            y=yy[:,0], 
                            z=Z,
                            colorscale='RdBu',
                            opacity=0.5,
                            showscale=False
                        )
                        st.plotly_chart(fig)
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    col1, col2 = st.columns(2)
                    col1.metric("MSE", f"{mse:.3f}")
                    col2.metric("R² Score", f"{r2:.3f}")

                    # Residual Plot
                    fig = px.scatter(
                        x=y_pred,
                        y=y_test - y_pred,
                        labels={'x': 'Predicted Values', 'y': 'Residuals'},
                        title='Residual Plot'
                    )
                    fig.add_hline(y=0, line_dash="dot", line_color="red")
                    st.plotly_chart(fig)

                # Feature Importance
                st.subheader("Feature Importance")
                result = permutation_importance(model, X_test[selected_features], y_test, n_repeats=10)
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': result.importances_mean
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Permutation Importance'
                )
                st.plotly_chart(fig_imp)

    with tab2:
        st.header("Automated Hyperparameter Tuning")
        selected_features_gs = st.multiselect(
            "Select Features for Tuning",
            X_train.columns
        )
        
        if selected_features_gs:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
            
            if st.button("Run GridSearchCV"):
                if problem_type == 'classification':
                    model = SVC(probability=True)
                    scoring = 'accuracy'
                else:
                    model = SVR()
                    scoring = 'neg_mean_squared_error'
                
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                gs.fit(X_train[selected_features_gs], y_train)
                
                # Best model results
                best_model = gs.best_estimator_
                st.subheader("Best Parameters")
                st.write(f"**C:** {gs.best_params_['C']}")
                st.write(f"**Kernel:** {gs.best_params_['kernel']}")
                st.write(f"**Gamma:** {gs.best_params_['gamma']}")
                st.write(f"**Best Score:** {gs.best_score_:.3f}")

                # Validation results visualization
                st.subheader("Hyperparameter Performance")
                results = pd.DataFrame(gs.cv_results_)
                
                fig = px.parallel_coordinates(
                    results,
                    color='mean_test_score',
                    dimensions=['param_C', 'param_kernel', 'param_gamma', 'mean_test_score'],
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig)

                # Feature importance from best model
                st.subheader("Best Model Feature Importance")
                result = permutation_importance(best_model, X_test[selected_features_gs], y_test)
                importance_df = pd.DataFrame({
                    'Feature': selected_features_gs,
                    'Importance': result.importances_mean
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Permutation Importance (Best Model)'
                )
                st.plotly_chart(fig_imp)

                # Final model performance
                st.subheader("Test Set Performance")
                y_pred = best_model.predict(X_test[selected_features_gs])
                
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Test Accuracy", f"{accuracy:.3f}")
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    col1, col2 = st.columns(2)
                    col1.metric("Test MSE", f"{mse:.3f}")
                    col2.metric("Test R²", f"{r2:.3f}")

                # Learning curve (example implementation)
                try:
                    from sklearn.model_selection import learning_curve
                    train_sizes, train_scores, test_scores = learning_curve(
                        best_model,
                        X_train[selected_features_gs],
                        y_train,
                        cv=5,
                        scoring=scoring
                    )
                    
                    fig_lc = go.Figure()
                    fig_lc.add_trace(go.Scatter(
                        x=train_sizes,
                        y=train_scores.mean(axis=1),
                        name='Training Score'
                    ))
                    fig_lc.add_trace(go.Scatter(
                        x=train_sizes,
                        y=test_scores.mean(axis=1),
                        name='Validation Score'
                    ))
                    fig_lc.update_layout(title='Learning Curve')
                    st.plotly_chart(fig_lc)
                except ImportError:
                    pass