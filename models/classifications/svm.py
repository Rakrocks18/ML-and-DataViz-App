import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

def svm_page():
    st.title("Support Vector Machine Classification")
    
    if "df" not in st.session_state:
        st.error("Please upload data first.")
        return
    
    df = st.session_state.df
    
    # Feature selection
    features = st.multiselect("Select features", df.columns)
    target = st.selectbox("Select target variable", df.columns)
    
    if features and target:
        X = df[features].copy()
        y = df[target].copy()
        
        # Handle categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Model parameters
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        
        # Create and train model
        if st.button("Train Model"):
            model = SVC()
            
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_scaled, y)
            
            # Display results
            st.subheader("Model Results")
            st.write("Best Parameters:", grid_search.best_params_)
            st.write("Best Score:", grid_search.best_score_)
            
            if kernel == 'linear':
                # For linear kernel, we can get feature importance
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': np.abs(grid_search.best_estimator_.coef_[0])
                }).sort_values('Importance', ascending=False)
                
                st.write("Feature Importance (for linear kernel):")
                st.dataframe(importance_df) 