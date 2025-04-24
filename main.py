import streamlit as st

st.set_page_config(page_title="Data Preprocessing, Visualization and Modelling App", layout="wide")
st.title("Data Preprocessing, Visualization and Modelling App")

# Initialize session state
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# Define pages
upload_data = st.Page("data_upload/data_upload.py", title="Upload Data", icon="📤")
live_data = st.Page("data_upload/live_data_preview.py", title="Live Data Preview")

# Preprocessing Pages
handle_missing = st.Page("preprocessing/handle_missing.py", title="Handle Missing Data")
encode_categorical = st.Page("preprocessing/encode_categorical.py", title="Encode Categorical")
feature_scaling = st.Page("preprocessing/feature_scaling.py", title="Feature Scaling")
split_data = st.Page("preprocessing/split_data.py", title="Split Data")

# Visualization Pages
scatter_plots = st.Page("visualization/scatter.py", title="Scatter Plots", icon="📈")
histograms = st.Page("visualization/histogram.py", title="Distributions", icon="📊")
box_plots = st.Page("visualization/box_plots.py", title="Box Plots", icon="📦")
correlation = st.Page("visualization/correlation.py", title="Correlations", icon="🔗")
pair_plots = st.Page("visualization/pair_plots.py", title="Pair Plots", icon="🔄")
# Visualization Pages
line_plots = st.Page("visualization/line_plot.py", title="Line Plots", icon="📉")
pie_charts = st.Page("visualization/pie_chart.py", title="Pie Charts", icon="🥧")
treemap = st.Page("visualization/treemap.py", title="Treemap", icon="🌳")
waffle_charts = st.Page("visualization/waffle_chart.py", title="Waffle Charts", icon="🧇")

# Regression Models
linear_model = st.Page("models/regressions/linear_regression.py", title="Linear Regression")
polynomial = st.Page("models/regressions/polynomial_regression.py", title="Polynomial Regression")
lasso = st.Page("models/regressions/lasso_regression.py", title="Lasso Regression")

# Classification Models
logistic_regression = st.Page("models/classifications/logistic_regression.py", title="Logistic Regression")

#Tree Models
decision_tree = st.Page("models/classifications/decision_tree.py", title="Decision Tree Classifier")

#Neighbor Models
knn = st.Page("models/classifications/knn.py", title="K-Nearest Neighbors")

#SVM
svm_model = st.Page("models/classifications/svm.py", title="Support Vector Machine")

#Neural Networks
simple_nn = st.Page("models/neural_networks/simple_nn.py", title="Simple Neural Network")

# Clustering Models
kmeans = st.Page("models/clustering/kmeans.py", title="K-Means")
dbscan = st.Page("models/clustering/dbscan.py", title="DBSCAN")
hierarchical = st.Page("models/clustering/hierarchical.py", title="Hierarchical")
gmm = st.Page("models/clustering/gmm.py", title="Gaussian Mixture Models")

# Sequential Learning Models
hmm_model = st.Page("models/sequential_learning/hmm_model.py", title="Hidden Markov Model")

#model comparision 
model_comparison = st.Page("analysis/model_comparison.py", title="Model Comparison", icon="⚖️")
# Add to navigation dictionary under "Analysis" or at the end

# Navigation based on data availability
if "df" in st.session_state:
    pg = st.navigation({
        "Data": [upload_data, live_data],
        "Preprocessing": [handle_missing, encode_categorical, feature_scaling, split_data],
        "Visualization": [scatter_plots, histograms, box_plots, line_plots, correlation, pair_plots, pie_charts, treemap, waffle_charts],
        "Regression Models": [linear_model, polynomial, lasso],
        "Classification Models": [logistic_regression],
        "Trees": [decision_tree],
        "Neighbors": [knn],
        "SVM": [svm_model],
        "Neural Networks": [simple_nn],
        "Clustering": [kmeans, dbscan, hierarchical, gmm],
        "Sequential Learning": [hmm_model],
        "Analysis": [model_comparison]
    })
else:
    pg = st.navigation({
        "Data": [upload_data]
    })

pg.run()
