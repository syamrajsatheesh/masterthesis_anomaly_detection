import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def generate_clusters(n_clusters, n_samples_per_cluster, cluster_centers, n_anomalies):
    # Generate clusters
    X, y = make_blobs(n_samples=n_clusters * n_samples_per_cluster, centers=cluster_centers, random_state=42)

    # Generate anomalies
    anomalies = np.random.rand(n_anomalies, len(cluster_centers[0]))
    for i in range(len(cluster_centers[0])):
        anomalies[:, i] = anomalies[:, i] * (max(X[:, i]) - min(X[:, i])) + min(X[:, i])

    # Concatenate clusters and anomalies
    X = np.concatenate([X, anomalies])
    y = np.concatenate([y, np.ones(n_anomalies) * -1])  # -1 label for anomalies

    return X, y

def plot_clusters(X, y, cluster_centers):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('Custom Clusters with Anomalies')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Parameters
n_clusters = 3
n_samples_per_cluster = 50
cluster_centers = [[-5, 0], [0, 5], [5, 0]]
n_anomalies = 10

# Generate clusters and anomalies
X, y = generate_clusters(n_clusters, n_samples_per_cluster, cluster_centers, n_anomalies)

# Plot the clusters and anomalies
plot_clusters(X, y, np.array(cluster_centers))


