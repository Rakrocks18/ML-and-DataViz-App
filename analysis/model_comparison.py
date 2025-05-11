import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

st.title("Model Comparison and Best Model Selection")

# Check if models have been trained
if "model_metrics" not in st.session_state or len(st.session_state.model_metrics) < 2:
    st.warning("Please train at least two models to compare!")
else:
    task_type = st.session_state.get("task_type", "regression")  # Default to regression if not set
    model_metrics = st.session_state.model_metrics

    # Display metrics table with improved formatting
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame(model_metrics).T
    
    # Add styling to highlight best metrics
    if task_type == "regression":
        # For regression: higher R² is better, lower MSE/RMSE/MAE is better
        styled_df = metrics_df.style.highlight_max(subset=[col for col in metrics_df.columns if 'r2' in col.lower()])
        styled_df = styled_df.highlight_min(subset=[col for col in metrics_df.columns 
                                                  if any(x in col.lower() for x in ['mse', 'rmse', 'mae'])])
    else:
        # For classification: higher values are generally better
        styled_df = metrics_df.style.highlight_max()
    
    st.dataframe(styled_df)
    
    # Visualization of metrics
    st.subheader("Metrics Visualization")
    
    # Create a melted dataframe for visualization
    viz_df = metrics_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    viz_df.rename(columns={'index': 'Model'}, inplace=True)
    
    # Create bar chart for each metric
    for metric in metrics_df.columns:
        fig = px.bar(
            viz_df[viz_df['Metric'] == metric], 
            x='Model', 
            y='Value',
            title=f"{metric} Comparison",
            color='Model'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Format prompt for LLM with more context
    prompt = f"""You are an expert data scientist analyzing model performance metrics.

Task: Analyze the performance of several {task_type} models trained on the same dataset and determine which model performs best.

Here are the performance metrics:
"""
    for model_name, metrics in model_metrics.items():
        metrics_str = ", ".join([f"{k} = {v:.4f}" for k, v in metrics.items()])
        prompt += f"- {model_name}: {metrics_str}\n"
    
    prompt += f"""
Please provide a comprehensive analysis:
1. Identify the best model for this {task_type} task with clear reasoning
2. Compare the top 2-3 models and explain their trade-offs
3. For regression models, focus on R² (higher is better) and error metrics like MSE/RMSE (lower is better)
4. For classification models, focus on Accuracy, Precision, Recall, and F1-Score (higher is better)
5. Suggest potential next steps for model improvement

Format your response in markdown with clear sections.
"""

    # LLM API Call with Groq using .env
    llm_models = {
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "Llama 3.1 70B": "llama-3.1-70b-versatile",
        "Llama 3.1 405B": "llama-3.1-405b-versatile",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma 7B": "gemma-7b-it"
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox("Select LLM Model", list(llm_models.keys()), index=1)
    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("Analyze Models with LLM"):
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not found in .env file. Please add it to use the LLM analysis.")
                st.info("Create a .env file in the project root with: GROQ_API_KEY=your_api_key")
            else:
                with st.spinner(f"Analyzing models with {selected_model}..."):
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an expert data scientist specializing in model evaluation and selection."},
                            {"role": "user", "content": prompt}
                        ],
                        model=llm_models[selected_model],
                        temperature=temperature,
                        max_tokens=1024
                    )
                    conclusion = response.choices[0].message.content
                
                # Display LLM analysis in an expandable section
                st.subheader("LLM Analysis")
                st.markdown(conclusion)
                
                # Save analysis to session state for reference
                if "model_analyses" not in st.session_state:
                    st.session_state.model_analyses = {}
                st.session_state.model_analyses[f"{selected_model}_{temperature}"] = conclusion
                
        except Exception as e:
            st.error(f"Error contacting LLM API: {e}")
            st.info("Check your API key and internet connection. If the error persists, try a different LLM model.")
    
    # Show previous analyses if available
    if "model_analyses" in st.session_state and st.session_state.model_analyses:
        with st.expander("Previous Analyses"):
            for analysis_key, analysis_text in st.session_state.model_analyses.items():
                st.markdown(f"**{analysis_key}**")
                st.markdown(analysis_text)
                st.divider()