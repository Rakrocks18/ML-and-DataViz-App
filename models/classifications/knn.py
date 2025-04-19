import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            mean_squared_error, r2_score)

st.title("K-Nearest Neighbors Analysis")

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
                n_neighbors = st.slider("Number of Neighbors (k)", 1, 50, 5)
                weights = st.selectbox("Weighting", ['uniform', 'distance'])
            with col2:
                p = st.selectbox("Distance Metric", 
                                [('Euclidean (p=2)', 2),
                                 ('Manhattan (p=1)', 1)],
                                format_func=lambda x: x[0])[1]
                algorithm = st.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
            
            if st.button("Train KNN Model"):
                if problem_type == 'classification':
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        p=p,
                        algorithm=algorithm
                    )
                else:
                    model = KNeighborsRegressor(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        p=p,
                        algorithm=algorithm
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
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    col1, col2 = st.columns(2)
                    col1.metric("MSE", f"{mse:.3f}")
                    col2.metric("R² Score", f"{r2:.3f}")

                # Decision Boundary Visualization (for 2 features)
                if len(selected_features) == 2 and problem_type == 'classification':
                    st.subheader("Decision Boundaries")
                    
                    # Create mesh grid
                    x_min, x_max = X_train[selected_features[0]].min()-1, X_train[selected_features[0]].max()+1
                    y_min, y_max = X_train[selected_features[1]].min()-1, X_train[selected_features[1]].max()+1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                       np.arange(y_min, y_max, 0.02))
                    
                    # Predict on mesh grid
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    
                    # Create plot
                    fig = px.scatter(
                        X_test, 
                        x=selected_features[0], 
                        y=selected_features[1], 
                        color=y_pred,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
                    fig.add_contour(
                        x=xx[0], 
                        y=yy[:,0], 
                        z=Z,
                        colorscale=px.colors.qualitative.Plotly,
                        showscale=False,
                        opacity=0.3
                    )
                    st.plotly_chart(fig)

                elif problem_type == 'regression':
                    st.subheader("Actual vs Predicted")
                    fig = px.scatter(
                        x=y_test, 
                        y=y_pred, 
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        trendline="ols"
                    )
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                x0=y_test.min(), y0=y_test.min(),
                                x1=y_test.max(), y1=y_test.max())
                    st.plotly_chart(fig)

    with tab2:
        st.header("Automated Hyperparameter Tuning")
        selected_features_gs = st.multiselect(
            "Select Features for Tuning",
            X_train.columns
        )
        
        if selected_features_gs:
            param_grid = {
                'n_neighbors': list(range(1, 31)),
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
            
            if st.button("Run GridSearchCV"):
                if problem_type == 'classification':
                    model = KNeighborsClassifier()
                    scoring = 'accuracy'
                else:
                    model = KNeighborsRegressor()
                    scoring = 'neg_mean_squared_error'
                
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring=scoring,
                    n_jobs=-1
                )
                gs.fit(X_train[selected_features_gs], y_train)
                
                # Best model results
                best_model = gs.best_estimator_
                st.subheader("Best Parameters")
                st.write(f"**Neighbors:** {gs.best_params_['n_neighbors']}")
                st.write(f"**Weights:** {gs.best_params_['weights']}")
                st.write(f"**Distance Metric:** {'Manhattan (p=1)' if gs.best_params_['p'] == 1 else 'Euclidean (p=2)'}")
                st.write(f"**Best Score:** {gs.best_score_:.3f}")

                # Validation Curve
                st.subheader("Validation Curve")
                results = pd.DataFrame(gs.cv_results_)
                fig_val = px.line(
                    results,
                    x='param_n_neighbors',
                    y='mean_test_score',
                    color='param_weights',
                    facet_col='param_p',
                    error_y='std_test_score',
                    title='Performance by Number of Neighbors'
                )
                st.plotly_chart(fig_val)

                # Final Model Visualization
                st.subheader("Best Model Performance")
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

                # Feature importance visualization (permutation importance)
                try:
                    from sklearn.inspection import permutation_importance
                    result = permutation_importance(best_model, X_test[selected_features_gs], y_test)
                    importance_df = pd.DataFrame({
                        'Feature': selected_features_gs,
                        'Importance': result.importances_mean
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Feature',
                        y='Importance',
                        title='Feature Importance (Permutation)'
                    )
                    st.plotly_chart(fig_imp)
                except ImportError:
                    st.info("Permutation importance requires scikit-learn 0.22+")