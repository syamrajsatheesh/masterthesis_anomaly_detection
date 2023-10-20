import numpy as np

class ClusterGenerator:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.clusters = []

    def add_cluster(self, center, velocity, num_points):
        cluster = {
            "center": center,
            "velocity": velocity,
            "num_points": num_points,
        }
        self.clusters.append(cluster)

    def generate_clusters(self):
        data = []
        for cluster in self.clusters:
            center = cluster["center"]
            velocity = cluster["velocity"]
            num_points = cluster["num_points"]
            cluster_data = self.generate_cluster_data(center, velocity, num_points)
            data.extend(cluster_data)
        return data

    def generate_cluster_data(self, center, velocity, num_points):
        data = []
        for _ in range(num_points):
            point = center + np.random.normal(0, 0.1, 2)  # Add some random noise
            data.append(point)
            center += velocity  # Move the center according to the velocity
        return data

