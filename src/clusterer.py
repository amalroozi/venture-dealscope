from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

def cluster_and_reduce(filtered_df, filtered_embeddings, n_clusters=5):
    n_clusters = min(n_clusters, len(filtered_df) // 10)
    n_clusters = max(2, n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    filtered_df['cluster'] = kmeans.fit_predict(filtered_embeddings)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(filtered_embeddings)
    filtered_df['x'] = coords[:, 0]
    filtered_df['y'] = coords[:, 1]
    
    return filtered_df