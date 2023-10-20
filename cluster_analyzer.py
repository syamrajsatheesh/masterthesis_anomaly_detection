import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusterAnalyzer:
    def __init__(self, data):
        self.data = data
        self.cluster_centers = None
        self.cluster_labels = None

    def find_optimal_clusters(self, max_clusters=10):
        scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(self.data)
            score = silhouette_score(self.data, kmeans.labels_)
            scores.append(score)

        optimal_clusters = np.argmax(scores) + 2  # +2 because we started from k=2
        return optimal_clusters

    def analyze_clusters(self, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(self.data)
        self.cluster_centers = kmeans.cluster_centers_
        self.cluster_labels = kmeans.labels_

    def calculate_density(self):
        unique_labels = np.unique(self.cluster_labels)
        cluster_density = {}
        for label in unique_labels:
            cluster_density[label] = np.sum(self.cluster_labels == label)
        return cluster_density

    def calculate_velocity(self, previous_centers, time_interval=1):
        if previous_centers is None:
            return None
        velocity = (self.cluster_centers - previous_centers) / time_interval
        return velocity

if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    data = np.random.rand(200, 2) * 10

    # Create a ClusterAnalyzer
    cluster_analyzer = ClusterAnalyzer(data)

    # Find the optimal number of clusters
    num_clusters = cluster_analyzer.find_optimal_clusters(max_clusters=10)
    print(f"The optimal number of clusters is: {num_clusters}")

    # Analyze the clusters
    cluster_analyzer.analyze_clusters(num_clusters)

    # Calculate cluster density
    cluster_density = cluster_analyzer.calculate_density()
    print(f"Cluster Density: {cluster_density}")

    # Calculate cluster velocity (assuming a time interval of 1)
    previous_centers = None  # Set previous_centers to the centers at the previous time step
    cluster_velocity = cluster_analyzer.calculate_velocity(previous_centers)
    print(f"Cluster Velocity: {cluster_velocity}")