from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

def kmeans_pipeline(
    X,
    k_range = range(2 , 31),
    n_init = 20,
    random_state= 45,
):
    best_score = -1
    best_model = None
    best_K = None

    for k in k_range:
        pipeline = Pipeline([
            ('scaler' , StandardScaler()),
            ('kmeans' , KMeans(n_clusters = k , n_init = n_init, random_state = random_state))
        ])

        # fit the pipeline
        labels = pipeline.fit_predict(X)

        # transform X using same scaler
        X_scaled = pipeline.named_steps['scaler'].transform(X)

        score = silhouette_score(X_scaled , labels)
       
        if score > best_score:
            best_score = score
            best_model = pipeline
            best_k = k

    print(f"Best K: {best_k}")
    print(f"Best Silhouette Score: {best_score:.4f}")

    return best_model

# t-SNE model implementation
def tsne_model(X_data):
    model = TSNE()
    model_transformed = model.fit_transform(X_data)

    return model_transformed

# Hierarchical clustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster

def Hierarchical_cluster_scp(X_data, method_s):
    X_scaled = StandardScaler().fit_transform(X_data)

    model = linkage(X_scaled, method= method_s)

    for k in range(2 , 11):
        labels = fcluster(model, t = k, criterion= 'maxclust')
        print(f"K : {k} , silhouette Score : {silhouette_score(X_scaled, labels)}\n\
                \tcalinski harabasz score : {calinski_harabasz_score(X_scaled, labels)}\n\
                \tdavies bouldin score : {davies_bouldin_score(X_scaled, labels)}\n\n")

    return model