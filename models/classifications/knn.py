import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

def knn_page():
    st.title("K-Nearest Neighbors Classification")
    
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
        n_neighbors = st.slider("Number of Neighbors (K)", 1, 20, 5)
        weights = st.selectbox("Weight Function", ["uniform", "distance"])
        
        # Create and train model
        if st.button("Train Model"):
            model = KNeighborsClassifier()
            
            # Define parameter grid
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_scaled, y)
            
            # Display results
            st.subheader("Model Results")
            st.write("Best Parameters:", grid_search.best_params_)
            st.write("Best Score:", grid_search.best_score_) 