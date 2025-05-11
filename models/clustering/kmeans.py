import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import polars as pl

st.title("K-Means Clustering")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.select(pl.col(pl.NUMERIC_DTYPES))
    
    tab1, tab2 = st.tabs(["Manual Configuration", "Elbow Method Analysis"])
    
    with tab1:
        st.header("Manual Clustering")
        features = st.multiselect("Select Features", df.columns)
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        max_iter = st.slider("Max Iterations", 100, 1000, 300)
        
        if st.button("Run K-Means") and features:
            X = df[features]
            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
            labels = kmeans.fit_predict(X)
            
            # Reduce dimensions for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
                            title="PCA Visualization of Clusters",
                            labels={'x': 'PC1', 'y': 'PC2'})
            st.plotly_chart(fig)
            
            st.subheader("Cluster Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Inertia", f"{kmeans.inertia_:.2f}")
            col2.metric("Silhouette Score", f"{silhouette_score(X, labels):.2f}")
    
    with tab2:
        st.header("Elbow Method Analysis")
        features_elbow = st.multiselect("Select Features for Analysis", df.columns)
        max_clusters = st.slider("Max Clusters to Test", 2, 15, 8)
        
        if st.button("Run Elbow Analysis") and features_elbow:
            X = df[features_elbow]
            inertias = []
            s_scores = []
            
            for k in range(2, max_clusters+1):
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                s_scores.append(silhouette_score(X, kmeans.labels_))
            
            fig = px.line(x=range(2, max_clusters+1), y=inertias, 
                        title="Elbow Method - Inertia vs Number of Clusters",
                        labels={'x': 'Number of Clusters', 'y': 'Inertia'})
            fig.add_scatter(x=list(range(2, max_clusters+1)), y=s_scores, mode='lines', 
                           name='Silhouette Score', secondary_y=False)
            st.plotly_chart(fig)