import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
import os

def plot_learning_curve(model, X, y, cv=5):
    """Plot learning curve to show model performance vs training size."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        name='Training Score',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        name='Cross-validation Score',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training Examples',
        yaxis_title='Score',
        showlegend=True
    )
    
    return fig

def plot_roc_curve(y_test, y_pred_proba, class_names):
    """Plot ROC curve for multiclass classification."""
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig = go.Figure()
    for i in range(n_classes):
        fig.add_trace(go.Scatter(
            x=fpr[i], y=tpr[i],
            name=f'ROC curve (class {class_names[i]}) (AUC = {roc_auc[i]:.2f})',
            line=dict(width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def xgboost_page():
    st.title("XGBoost Classification")
    
    if "df" not in st.session_state:
        st.error("Please upload data first.")
        return
    
    df = st.session_state.df
    
    # Feature selection
    features = st.multiselect("Select features", df.columns)
    target = st.selectbox("Select target variable", df.columns)
    
    if features and target:
        try:
            X = df[features].copy()
            y = df[target].copy()
            
            # Handle categorical variables in features
            label_encoders = {}
            for column in X.select_dtypes(include=['object']).columns:
                label_encoders[column] = LabelEncoder()
                X[column] = label_encoders[column].fit_transform(X[column])
            
            # Handle target variable - ensure consecutive integers starting from 0
            if y.dtype == 'object' or y.dtype == 'int64' or y.dtype == 'float64':
                # First, get unique classes and sort them
                unique_classes = sorted(y.unique())
                # Create a mapping from original classes to consecutive integers
                class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
                # Map the original classes to consecutive integers
                y = y.map(class_mapping)
                # Store the original class names for reference
                class_names = [str(cls) for cls in unique_classes]
                st.info(f"Class mapping: {class_mapping}")
            
            # Data preprocessing options
            with st.expander("Data Preprocessing Options"):
                scale_features = st.checkbox("Scale Features", value=True)
                if scale_features:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    X = pd.DataFrame(X, columns=features)
            
            # Split the data
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Model parameters with advanced options
            with st.expander("Model Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    n_estimators = st.slider("Number of Estimators", 50, 500, 100)
                    max_depth = st.slider("Maximum Depth", 3, 10, 6)
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                with col2:
                    min_child_weight = st.slider("Min Child Weight", 1, 10, 1)
                    subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
                    colsample_bytree = st.slider("Column Sample by Tree", 0.5, 1.0, 1.0, 0.1)
            
            # Create and train model
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Create base model
                    model = xgb.XGBClassifier(
                        objective='multi:softmax',
                        num_class=len(np.unique(y)),
                        random_state=42,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree
                    )
                    
                    # Define parameter grid
                    param_grid = {
                        'n_estimators': [n_estimators],
                        'max_depth': [max_depth],
                        'learning_rate': [learning_rate],
                        'min_child_weight': [min_child_weight],
                        'subsample': [subsample],
                        'colsample_bytree': [colsample_bytree]
                    }
                    
                    # Perform GridSearchCV
                    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = grid_search.predict(X_test)
                    y_pred_proba = grid_search.predict_proba(X_test)
                    
                    # Display results
                    st.subheader("Model Results")
                    st.write("Best Parameters:", grid_search.best_params_)
                    st.write("Best Cross-Validation Score:", f"{grid_search.best_score_:.4f}")
                    
                    # Calculate and display test accuracy
                    test_score = grid_search.score(X_test, y_test)
                    st.write("Test Accuracy:", f"{test_score:.4f}")
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': grid_search.best_estimator_.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.write("Feature Importance:")
                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                title='Feature Importance Plot')
                    st.plotly_chart(fig)
                    
                    # SHAP values for feature importance
                    with st.spinner("Calculating SHAP values..."):
                        explainer = shap.TreeExplainer(grid_search.best_estimator_)
                        shap_values = explainer.shap_values(X_test)
                        
                        if len(np.unique(y)) <= 10:  # Only show SHAP for reasonable number of classes
                            st.subheader("SHAP Values")
                            for i, class_name in enumerate(class_names):
                                fig = shap.summary_plot(shap_values[i], X_test, show=False)
                                st.pyplot(fig)
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm,
                                     labels=dict(x="Predicted", y="Actual"),
                                     title="Confusion Matrix",
                                     color_continuous_scale="Viridis")
                    st.plotly_chart(fig_cm)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, target_names=class_names)
                    st.text(report)
                    
                    # Learning Curve
                    st.subheader("Learning Curve")
                    learning_curve_fig = plot_learning_curve(grid_search.best_estimator_, X_train, y_train)
                    st.plotly_chart(learning_curve_fig)
                    
                    # ROC Curves
                    if len(np.unique(y)) <= 10:  # Only show ROC for reasonable number of classes
                        st.subheader("ROC Curves")
                        roc_fig = plot_roc_curve(y_test, y_pred_proba, class_names)
                        st.plotly_chart(roc_fig)
                    
                    # Model persistence
                    if st.button("Save Model"):
                        model_dir = "saved_models"
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, "xgboost_model.joblib")
                        joblib.dump(grid_search.best_estimator_, model_path)
                        st.success(f"Model saved to {model_path}")
                    
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.info("Try selecting different features or adjusting the parameters.")
    else:
        st.warning("Please select both features and target columns.") 