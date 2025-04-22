# ML and Data Visualization App

A comprehensive machine learning and data visualization application built with Streamlit that allows users to preprocess data, visualize patterns, train various machine learning models, and compare their performance.

## Features

### Data Handling
- **Data Upload**: Import your datasets in various formats
- **Live Data Preview**: Examine your data before processing

### Preprocessing
- **Handle Missing Data**: Multiple strategies for dealing with missing values
- **Encode Categorical Variables**: Convert categorical data to numerical format
- **Feature Scaling**: Normalize or standardize your features
- **Data Splitting**: Divide data into training and testing sets

### Visualization
- **Scatter Plots**: Explore relationships between variables
- **Histograms**: Understand data distributions
- **Box Plots**: Identify outliers and compare distributions
- **Correlation Analysis**: Discover relationships between features
- **Pair Plots**: Visualize pairwise relationships in the dataset

### Machine Learning Models

#### Regression Models
- Linear Regression
- Polynomial Regression
- Lasso Regression

#### Classification Models
- Logistic Regression
- Decision Trees
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)

#### Neural Networks
- Simple Neural Network implementation

#### Sequential Learning
- Hidden Markov Models (HMM)

### Analysis
- **Model Comparison**: Compare performance metrics across different models
- **LLM-Powered Analysis**: Get insights on model performance using Groq's LLM API

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ML-and-DataViz-App.git
cd ML-and-DataViz-App

# Install dependencies
pip install -r requirements.txt

# For LLM-powered analysis, create a .env file in the analysis directory with:
GROQ_API_KEY=your_groq_api_key
```

## Usage

```bash
# Run the application
streamlit run main.py
```

The application will open in your default web browser. Follow these steps:

1. Upload your dataset
2. Preprocess the data as needed
3. Visualize patterns and relationships
4. Train and evaluate machine learning models
5. Compare model performance to select the best one

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Polars
- Scikit-learn
- XGBoost
- Plotly
- NumPy
- Matplotlib
- Seaborn
- Groq API (for LLM analysis)

## Project Structure

```
├── main.py                  # Main application entry point
├── data_upload/            # Data loading functionality
├── preprocessing/          # Data preprocessing modules
├── visualization/          # Data visualization components
├── models/                 # Machine learning models
│   ├── classifications/    # Classification algorithms
│   ├── regressions/        # Regression algorithms
│   ├── neural_networks/    # Neural network implementations
│   └── sequential_learning/ # Sequential learning models
└── analysis/              # Model comparison and analysis
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
