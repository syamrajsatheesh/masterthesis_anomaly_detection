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

if __name__ == "__main__":
    cluster_gen = ClusterGenerator(num_clusters=3)

    # Add clusters with their centers, velocities, and number of data points
    cluster_gen.add_cluster(center=[0, 0], velocity=[0.02, 0.02], num_points=100)
    cluster_gen.add_cluster(center=[5, 5], velocity=[-0.02, -0.02], num_points=100)
    cluster_gen.add_cluster(center=[-5, 5], velocity=[0.01, -0.01], num_points=100)

    generated_data = cluster_gen.generate_clusters()

    # Print the generated data
    for point in generated_data:
        print(point)