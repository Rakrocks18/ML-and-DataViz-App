import streamlit as st
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import polars as pl

st.title("DBSCAN Clustering")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.select(pl.col(pl.NUMERIC_DTYPES))
    
    features = st.multiselect("Select Features", df.columns)
    eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, step=0.1)
    min_samples = st.slider("Minimum Samples", 1, 20, 5)
    
    if st.button("Run DBSCAN") and features:
        X = StandardScaler().fit_transform(df[features])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
                        title="DBSCAN Clustering Results",
                        labels={'x': 'PC1', 'y': 'PC2'})
        st.plotly_chart(fig)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.subheader("Cluster Metrics")
        st.metric("Number of Clusters Found", n_clusters)
        st.metric("Noise Points", f"{(labels == -1).sum()} points")