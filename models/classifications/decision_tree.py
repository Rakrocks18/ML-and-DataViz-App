import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            mean_squared_error, r2_score)

st.title("Decision Tree Analysis")

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
        selected_features = st.multiselect("Select Features", X_train.columns)
        
        if selected_features:
            col1, col2 = st.columns(2)
            with col1:
                max_depth = st.slider("Max Depth", 1, 20, 3)
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            with col2:
                criterion = st.selectbox(
                    "Split Criterion",
                    ['gini', 'entropy'] if problem_type == 'classification' 
                    else ['squared_error', 'friedman_mse', 'absolute_error']
                )
                splitter = st.selectbox("Split Strategy", ['best', 'random'])
            
            if st.button("Train Decision Tree"):
                if problem_type == 'classification':
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        criterion=criterion,
                        splitter=splitter
                    )
                else:
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        criterion=criterion,
                        splitter=splitter
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

                # Feature Importance
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance Scores'
                )
                st.plotly_chart(fig_imp)

                # Tree Visualization
                st.subheader("Decision Tree Structure")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(
                    model,
                    feature_names=selected_features,
                    filled=True,
                    rounded=True,
                    ax=ax
                )
                st.pyplot(fig)

    with tab2:
        st.header("Automated Hyperparameter Tuning")
        selected_features_gs = st.multiselect(
            "Select Features for Tuning",
            X_train.columns
        )
        
        if selected_features_gs:
            param_grid = {
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy'] if problem_type == 'classification'
                            else ['squared_error', 'absolute_error']
            }
            
            if st.button("Run GridSearchCV"):
                if problem_type == 'classification':
                    model = DecisionTreeClassifier()
                    scoring = 'accuracy'
                else:
                    model = DecisionTreeRegressor()
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
                st.write(f"**Max Depth:** {gs.best_params_.get('max_depth', 'Unlimited')}")
                st.write(f"**Min Samples Split:** {gs.best_params_['min_samples_split']}")
                st.write(f"**Criterion:** {gs.best_params_['criterion']}")
                st.write(f"**Best Score:** {gs.best_score_:.3f}")

                # Validation Curve
                st.subheader("Validation Curve")
                results = pd.DataFrame(gs.cv_results_)
                fig_val = px.line(
                    results,
                    x='param_max_depth',
                    y='mean_test_score',
                    error_y='std_test_score',
                    color='param_criterion',
                    title='Performance by Max Depth and Criterion'
                )
                st.plotly_chart(fig_val)

                # Feature Importance from Best Model
                st.subheader("Best Model Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features_gs,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance from Best Model'
                )
                st.plotly_chart(fig_imp)

                # Final Tree Visualization
                st.subheader("Best Model Tree Structure")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(
                    best_model,
                    feature_names=selected_features_gs,
                    filled=True,
                    rounded=True,
                    ax=ax
                )
                st.pyplot(fig)