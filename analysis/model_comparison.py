import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

st.title("Model Comparison and Best Model Selection")

# Check if models have been trained
if "model_metrics" not in st.session_state or len(st.session_state.model_metrics) < 2:
    st.warning("Please train at least two models to compare!")
else:
    task_type = st.session_state.get("task_type", "regression")  # Default to regression if not set
    model_metrics = st.session_state.model_metrics

    # Format prompt for LLM
    prompt = f"Here are the performance metrics for several {task_type} models trained on the same dataset:\n\n"
    for model_name, metrics in model_metrics.items():
        metrics_str = ", ".join([f"{k} = {v:.3f}" for k, v in metrics.items()])
        prompt += f"- {model_name}: {metrics_str}\n"
    prompt += f"\nWhich model performs the best for this {task_type} task? Please state the best model and provide a brief explanation. "
    prompt += "For regression, higher R² and lower MSE are better. For classification, higher Accuracy and F1-Score are better."

    # Display metrics table
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame(model_metrics).T
    st.dataframe(metrics_df)

    # LLM API Call with Groq using .env
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        with st.spinner("Analyzing models with LLM..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in data analysis."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile"
            )
            conclusion = response.choices[0].message.content
        st.subheader("LLM Conclusion")
        st.write(conclusion)
    except Exception as e:
        st.error(f"Error contacting LLM API: {e}")