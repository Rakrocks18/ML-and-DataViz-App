import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import polars as pl

st.title("Hierarchical Clustering")

if "df" not in st.session_state:
    st.warning("Please upload data first!")
else:
    df = st.session_state.df.select(pl.col(pl.NUMERIC_DTYPES))
    
    features = st.multiselect("Select Features", df.columns)
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    linkage_method = st.selectbox("Linkage Method", 
                                ['ward', 'complete', 'average', 'single'])
    
    if st.button("Run Clustering") and features:
        X = df[features]
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X)
        
        # Dendrogram visualization
        st.subheader("Dendrogram")
        Z = linkage(X, method=linkage_method)
        fig = ff.create_dendrogram(Z)
        fig.update_layout(title=f"{linkage_method.title()} Linkage Dendrogram")
        st.plotly_chart(fig)
        
        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig_clusters = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
                                title="Cluster Visualization",
                                labels={'x': 'PC1', 'y': 'PC2'})
        st.plotly_chart(fig_clusters)