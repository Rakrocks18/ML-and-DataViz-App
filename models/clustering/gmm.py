import streamlit as st
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import polars as pl

st.title("Gaussian Mixture Models")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.select(pl.col(pl.NUMERIC_DTYPES))
    
    features = st.multiselect("Select Features", df.columns)
    n_components = st.slider("Number of Components", 1, 10, 3)
    covariance_type = st.selectbox("Covariance Type", 
                                ['full', 'tied', 'diag', 'spherical'])
    
    if st.button("Run GMM") and features:
        X = StandardScaler().fit_transform(df[features])
        gmm = GaussianMixture(n_components=n_components, 
                            covariance_type=covariance_type)
        labels = gmm.fit_predict(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
                        title="GMM Clustering Results",
                        labels={'x': 'PC1', 'y': 'PC2'})
        st.plotly_chart(fig)
        
        st.subheader("Model Metrics")
        col1, col2 = st.columns(2)
        col1.metric("BIC Score", f"{gmm.bic(X):.2f}")
        col2.metric("Log Likelihood", f"{gmm.lower_bound_:.2f}")