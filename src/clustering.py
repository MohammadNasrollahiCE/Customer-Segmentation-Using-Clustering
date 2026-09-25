from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def kmeans_pipeline(
    X,
    k_range = range(2 , 31),
    n_init = 20,
    random_state= 45,
    init= 'k-means++'
):
    best_score = -1
    best_model = None
    best_k = None
    K_Values = []
    Inertias = []

    for k in k_range:
        pipeline = Pipeline([
            ('scaler' , StandardScaler()),
            ('kmeans' , KMeans(n_clusters = k , n_init = n_init,
             random_state = random_state, init= init))
        ])

        # fit the pipeline
        labels = pipeline.fit_predict(X)

        Inertias.append(pipeline.named_steps['kmeans'].inertia_)

        # transform X using same scaler
        X_scaled = pipeline.named_steps['scaler'].transform(X)

        score = silhouette_score(X_scaled , labels)

        # Sets for storing values of K and score for ploting Elbow Methode
        K_Values.append(k)

        print(f"K : {k}, Score : {score}")
       
        if score > best_score:
            best_score = score
            best_model = pipeline
            best_k = k

    print("-----------------------------------------------------------")
    print(f"Best K: {best_k}")
    print(f"Best Silhouette Score: {best_score:.4f}")

    # Ploting Elbow Method
    plt.plot(K_Values, Inertias, marker= 'o')
    plt.xlabel('K Values')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

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

# DBSCAN 
# K-Distance Graph 

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
 
def k_distance_graph(X_data):
    # Scaling the data
    X_scaleed = StandardScaler().fit_transform(X_data)
    min_samples = 2 * X_scaleed.shape[1]

    # Finding the nearest neighbors
    nearest_neighbors = NearestNeighbors(n_neighbors= min_samples)
    nearest_neighbors.fit(X_scaleed)

    # Calculate the different amung nearest neighbors
    dist, _ = nearest_neighbors.kneighbors(X_scaleed)
    kdist = np.sort(dist[:, -1])

    # drow k distance graph
    plt.plot(kdist)
    plt.ylabel(f"{min_samples}-th neighbor distance")
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Distance "eps select"')
    plt.title('K-Distance Graph')
    plt.grid()
    plt.show()


# DBSCAN algorithm implementation
def dbscan_grid(X_data, eps_range):
    X_scaled = StandardScaler().fit_transform(X_data)
    min_samples = 2 * X_scaled.shape[1]
    best_score, best_labels, best_eps = -1, None, None

    for eps in eps_range:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        noise_ratio = 1 - mask.mean()

        if n_clusters >= 2:
            score = silhouette_score(X_scaled[mask], labels[mask])
            print(f"eps={eps:.2f}  k={n_clusters}  noise={noise_ratio:.1%}  sil={score:.3f}")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_eps = eps
        else:
            print(f"eps={eps:.2f}  k={n_clusters}  noise={noise_ratio:.1%}  (Just One Cluster)")

    print("------------------------------------------------------------------------")
    if best_eps is None:
        print("eps is None. One cluster")
    else:
        best_mask = best_labels != -1
        print(f"\nBest: eps={best_eps:.2f}, k={len(set(best_labels[best_mask]))}, "
              f"noise={1 - best_mask.mean():.1%}, sil={best_score:.3f}")

    return best_labels

# HDBSCAN algorithm implementation
from sklearn.cluster import HDBSCAN

def hdbscan_grid(X_data, min_cluster_size, min_samples):
    X_scaled = StandardScaler().fit_transform(X_data)
    best_score, best_labels, best_params = -1, None, None

    for mcs in min_cluster_size:
        for ms in min_samples:
            model = HDBSCAN(min_cluster_size=mcs, min_samples=ms, copy= True)
            labels = model.fit_predict(X_scaled)

            mask = labels != -1
            k = len(set(labels[mask]))
            noise = 1 - mask.mean()

            if k >= 2:
                score = silhouette_score(X_scaled[mask], labels[mask])
                print(f"mcs={mcs} ms={ms} k={k} noise={noise:.1%} sil={score:.3f}")
                if score > best_score:
                    best_score, best_labels, best_params = score, labels, (mcs, ms)

    print(f"\nBest: mcs={best_params[0]}, ms={best_params[1]}, sil={best_score:.3f}")
    return best_labels

# GMM Model implementation
from sklearn.mixture import GaussianMixture
import pandas as pd

def GMM_model(X_data, return_type):
    X_scaled = StandardScaler().fit_transform(X_data)
    best_score_b, best_score_a, best_labels, best_params = np.inf, np.inf, None, None
    best_model= None
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    results = []

    for cov in covariance_types:
        for k in range(2,11):
            model = GaussianMixture(n_components= k, covariance_type= cov, random_state= 42, n_init= 30)
            labels = model.fit_predict(X_scaled)

            if k >= 2:
                score_BIC = model.bic(X_scaled)
                score_AIC = model.aic(X_scaled)
                print(f"k={k}, cov={cov}, BIC={score_BIC:.2f}, AIC={score_AIC:.2f}")

                results.append({
                    "n_components": k,
                    "covariance_type": cov,
                    "BIC": score_BIC,
                    "AIC": score_AIC
                })

                if score_BIC < best_score_b:
                    best_score_a = score_AIC
                    best_score_b = score_BIC
                    best_labels = labels
                    best_params = (k, cov)
                    best_model = model

    print("----------------------------------------------------------------------------------")
    print(f"\nBest: best k={best_params[0]}, best cov={best_params[1]}, best BIC={best_score_b:.2f}, best AIC={best_score_a:.2f}")

    df_results = pd.DataFrame(results)
    for cov in df_results['covariance_type'].unique():
        df_cov = df_results[df_results['covariance_type'] == cov]

        plt.plot(df_cov['n_components'], df_cov['BIC'], marker='o', label=cov)

    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.xticks(range(2, 11))
    plt.grid()
    plt.legend()
    plt.show()

    if return_type == 'proba':
        proba = best_model.predict_proba(X_scaled)
        proba_df = pd.DataFrame(proba, columns=[f'Cluster_{i}_Prob'
                for i in range(best_model.n_components)])

        return proba_df

    elif return_type == 'labels':
        return best_labels
    elif return_type == 'model':
        return best_model