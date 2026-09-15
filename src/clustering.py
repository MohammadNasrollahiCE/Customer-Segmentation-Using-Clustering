from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

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