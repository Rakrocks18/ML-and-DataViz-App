import streamlit as st
import plotly.express as px
# import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve, auc)

st.title("Logistic Regression Analysis")

if "X_train" not in st.session_state:
    st.warning("Please split your data first using the 'Split Data' page!")
else:
    # Get data from session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    
    # Check classification task
    unique_classes = y_train.nunique()
    if unique_classes > 2:
        st.info(f"Multi-class Classification ({unique_classes} classes)")
        multi_class = 'multinomial'
    else:
        st.info("Binary Classification")
        multi_class = 'ovr'

    # Create tabs
    tab1, tab2 = st.tabs(["Manual Mode", "GridSearchCV"])
    
    with tab1:
        st.header("Manual Configuration")
        
        selected_features = st.multiselect(
            "Select Features",
            options=X_train.columns.tolist()
        )
        
        col1, col2 = st.columns(2)
        with col1:
            C = st.number_input("Inverse Regularization (C)", 
                               0.001, 1000.0, 1.0, 0.1)
            penalty = st.selectbox("Penalty", ['l2', 'l1'])
        with col2:
            solver = st.selectbox("Solver", 
                                 ['lbfgs', 'liblinear', 'saga'])
            max_iter = st.number_input("Max Iterations", 100, 10000, 1000)

        if selected_features and st.button("Train Model"):
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
                multi_class=multi_class
            )
            model.fit(X_train[selected_features], y_train)
            
            # Predictions
            y_pred = model.predict(X_test[selected_features])
            y_proba = model.predict_proba(X_test[selected_features])

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader(f"Accuracy: {accuracy:.3f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=model.classes_,
                y=model.classes_,
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # ROC Curve (binary only)
            if unique_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig2 = px.area(
                    x=fpr, y=tpr,
                    title=f'ROC Curve (AUC = {roc_auc:.2f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate')
                )
                fig2.add_shape(type='line', line=dict(dash='dash'), 
                              x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.header("Automated Hyperparameter Tuning")
        
        selected_features = st.multiselect(
            "Select Features for Tuning",
            options=X_train.columns.tolist()
        )
        
        if selected_features:
            param_grid = {
                'C': np.logspace(-3, 3, 7),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

            if st.button("Run GridSearchCV"):
                model = LogisticRegression(
                    max_iter=5000,
                    multi_class=multi_class
                )
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                gs.fit(X_train[selected_features], y_train)
                
                # Best parameters
                st.subheader("Best Parameters")
                st.write(f"**C:** {gs.best_params_['C']:.3f}")
                st.write(f"**Penalty:** {gs.best_params_['penalty']}")
                st.write(f"**Solver:** {gs.best_params_['solver']}")
                st.write(f"**Best Accuracy (CV):** {gs.best_score_:.3f}")

                # Validation results
                results = pd.DataFrame(gs.cv_results_)
                fig = px.parallel_coordinates(
                    results,
                    color='mean_test_score',
                    dimensions=['param_C', 'param_penalty', 'param_solver', 'mean_test_score'],
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)