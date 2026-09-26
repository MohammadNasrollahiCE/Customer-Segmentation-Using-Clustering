from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# PCA API implementation
def pca_maker(X_data):
    X_scaled = StandardScaler().fit_transform(X_data)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # print PCx results
    features_pca = range(pca.n_components_)
    variance = pca.explained_variance_

    for feat, var in zip(features_pca, pca.explained_variance_):
        print(f"PC{feat} : {var:.2f} ({((var * 100) / variance.sum()):.2f}%)")

    plt.bar(features_pca, pca.explained_variance_)
    plt.xticks(features_pca)
    plt.xlabel('PCA Features')
    plt.ylabel('Variance')
    plt.grid()
    plt.show()

    return pca, X_pca