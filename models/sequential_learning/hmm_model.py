import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from hmmlearn import hmm
import plotly.express as px
import plotly.graph_objects as go

def prepare_hmm_data(X, y):
    """Prepare data for HMM by handling categorical variables and scaling."""
    # Handle categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape data for HMM (samples, timesteps, features)
    X_reshaped = X_scaled.reshape(-1, 1, X_scaled.shape[1])
    
    return X_reshaped, y, scaler, label_encoders

def plot_state_sequences(hidden_states, y_test, title="Hidden State Sequences"):
    """Plot the hidden state sequences and their relationship with the target variable."""
    fig = go.Figure()
    
    # Plot hidden states
    fig.add_trace(go.Scatter(
        y=hidden_states,
        mode='lines',
        name='Hidden States',
        line=dict(color='blue')
    ))
    
    # Plot actual values
    fig.add_trace(go.Scatter(
        y=y_test,
        mode='lines',
        name='Actual Values',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="State / Value",
        showlegend=True
    )
    
    return fig

def plot_state_distributions(state_probabilities, n_states):
    """Plot the distribution of state probabilities."""
    fig = go.Figure()
    
    for state in range(n_states):
        fig.add_trace(go.Box(
            y=state_probabilities[:, state],
            name=f'State {state}',
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title='State Probability Distributions',
        yaxis_title='Probability',
        showlegend=True
    )
    
    return fig

def hmm_page():
    st.title("Hidden Markov Model")
    
    if "df" not in st.session_state:
        st.error("Please upload data first.")
        return
    
    df = st.session_state.df
    
    # Feature selection
    features = st.multiselect("Select features", df.columns)
    target = st.selectbox("Select target variable", df.columns)
    
    if features and target:
        try:
            # Prepare the data
            X = df[features].copy()
            y = df[target].copy()
            
            # Model parameters
            st.subheader("Model Parameters")
            n_components = st.slider("Number of Hidden States", 2, 10, 3)
            covariance_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
            n_iter = st.slider("Number of Iterations", 10, 1000, 100)
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner("Training HMM..."):
                    # Prepare data
                    X_reshaped, y, scaler, label_encoders = prepare_hmm_data(X, y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_reshaped, y, test_size=0.2, random_state=42
                    )
                    
                    # Create and train model
                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        random_state=42
                    )
                    
                    model.fit(X_train.reshape(-1, X_train.shape[-1]))
                    
                    # Get predictions and probabilities
                    hidden_states = model.predict(X_test.reshape(-1, X_test.shape[-1]))
                    state_probabilities = model.predict_proba(X_test.reshape(-1, X_test.shape[-1]))
                    
                    # Calculate metrics
                    log_likelihood = model.score(X_test.reshape(-1, X_test.shape[-1]))
                    mse = mean_squared_error(y_test, hidden_states)
                    r2 = r2_score(y_test, hidden_states)
                    
                    # Display results
                    st.subheader("Model Results")
                    st.write("Log Likelihood:", log_likelihood)
                    st.write("Mean Squared Error:", mse)
                    st.write("R² Score:", r2)
                    
                    # Visualizations
                    st.subheader("Hidden State Sequences")
                    state_seq_fig = plot_state_sequences(hidden_states, y_test)
                    st.plotly_chart(state_seq_fig)
                    
                    st.subheader("State Probability Distributions")
                    state_dist_fig = plot_state_distributions(state_probabilities, n_components)
                    st.plotly_chart(state_dist_fig)
                    
                    # Feature importance based on emission probabilities
                    st.subheader("Feature Importance")
                    means = model.means_
                    importance_df = pd.DataFrame(
                        np.abs(means).mean(axis=0),
                        index=features,
                        columns=['Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x=importance_df.index,
                        y='Importance',
                        title='Feature Importance based on Emission Probabilities'
                    )
                    st.plotly_chart(fig)
                    
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.info("Try selecting different features or adjusting the parameters.")
    else:
        st.warning("Please select both features and target columns.") 