import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, accuracy_score

st.title("Neural Network Analysis")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[64], activation='relu', dropout=0.2):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        self.softmax = nn.Softmax(dim=1) if output_size > 1 else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return self.softmax(x)

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
    output_size = y_train.nunique() if problem_type == 'classification' else 1

    tab1, tab2 = st.tabs(["Manual Configuration", "Hyperparameter Tuning"])

    with tab1:
        st.header("Manual Configuration")
        selected_features = st.multiselect("Select Features", X_train.columns)
        
        if selected_features:
            col1, col2 = st.columns(2)
            with col1:
                hidden_layers = st.text_input("Hidden Layers (comma separated)", "64,32")
                hidden_layers = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
                activation = st.selectbox("Activation Function", ['relu', 'sigmoid'])
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            with col2:
                learning_rate = st.number_input("Learning Rate", 1e-4, 1.0, 1e-3, step=1e-3, format="%.4f")
                epochs = st.number_input("Epochs", 10, 500, 100)
                batch_size = st.number_input("Batch Size", 16, 256, 32)
                optimizer_type = st.selectbox("Optimizer", ['adam', 'sgd'])

            if st.button("Train Neural Network"):
                # Convert data to PyTorch tensors
                X_train_tensor = torch.tensor(X_train[selected_features].values)
                y_train_tensor = torch.tensor(y_train.values)
                
                # Initialize model
                model = NeuralNet(
                    input_size=len(selected_features),
                    output_size=output_size,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    dropout=dropout
                )
                
                # Define loss and optimizer
                criterion = nn.CrossEntropyLoss() if problem_type == 'classification' else nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == 'adam' else optim.SGD(model.parameters(), lr=learning_rate)
                
                # Training loop
                train_losses = []
                val_losses = []
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs.squeeze(), y_train_tensor)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(torch.tensor(X_test[selected_features].values))
                        val_loss = criterion(val_outputs.squeeze(), torch.tensor(y_test.values))
                        val_losses.append(val_loss.item())

                # Plot training curves
                st.subheader("Training Progress")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(epochs)),
                    y=train_losses,
                    name='Training Loss'
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(epochs)),
                    y=val_losses,
                    name='Validation Loss'
                ))
                fig.update_layout(title="Loss Curves", xaxis_title="Epochs", yaxis_title="Loss")
                st.plotly_chart(fig)

                # Evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = np.asarray(model(torch.tensor(X_test[selected_features].values)))

                if problem_type == 'classification':
                    y_pred = np.argmax(y_pred, axis=1)
                    accuracy = (y_pred == y_test).mean()
                    st.metric("Accuracy", f"{accuracy:.3f}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual"),
                        x=np.unique(y_test),
                        y=np.unique(y_test),
                        text_auto=True
                    )
                    st.plotly_chart(fig_cm)
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    col1, col2 = st.columns(2)
                    col1.metric("MSE", f"{mse:.3f}")
                    col2.metric("R² Score", f"{r2:.3f}")
                    
                    # Actual vs Predicted plot
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred.squeeze(),
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        trendline="ols"
                    )
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                x0=y_test.min(), y0=y_test.min(),
                                x1=y_test.max(), y1=y_test.max())
                    st.plotly_chart(fig)

    with tab2:
        st.header("Hyperparameter Tuning")
        selected_features_tune = st.multiselect("Select Features for Tuning", X_train.columns)
        
        if selected_features_tune:
            param_grid = {
                'module__hidden_layers': [[64], [64, 32], [128, 64]],
                'module__activation': ['relu', 'sigmoid'],
                'lr': [0.001, 0.01],
                'batch_size': [32, 64],
                'optimizer__weight_decay': [0, 0.001],
                'max_epochs': [50, 100]
            }

            if st.button("Run Hyperparameter Search"):
                # Create skorch model
                if problem_type == 'classification':
                    net = NeuralNetClassifier(
                        NeuralNet,
                        module__input_size=len(selected_features_tune),
                        module__output_size=output_size,
                        callbacks=[EarlyStopping(patience=5)],
                        verbose=0
                    )
                else:
                    net = NeuralNetRegressor(
                        NeuralNet,
                        module__input_size=len(selected_features_tune),
                        module__output_size=output_size,
                        callbacks=[EarlyStopping(patience=5)],
                        verbose=0
                    )

                gs = GridSearchCV(
                    net,
                    param_grid,
                    cv=3,
                    scoring='accuracy' if problem_type == 'classification' else 'r2',
                    n_jobs=-1
                )

                # Corrected fit call
                gs.fit(
                    X_train[selected_features_tune].values,
                    y_train.values
                )

                # Display results
                st.subheader("Best Parameters")
                results = pd.DataFrame(gs.cv_results_)
                st.write(f"Best Score: {gs.best_score_:.3f}")
                st.write(gs.best_params_)

                # Parallel coordinates plot
                fig = px.parallel_coordinates(
                    results,
                    color='mean_test_score',
                    dimensions=[col for col in results.columns if col.startswith('param_')],
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig)

                # Best model visualization
                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test[selected_features_tune].values)
                
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Test Accuracy", f"{accuracy:.3f}")
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    col1, col2 = st.columns(2)
                    col1.metric("Test MSE", f"{mse:.3f}")
                    col2.metric("Test R²", f"{r2:.3f}")