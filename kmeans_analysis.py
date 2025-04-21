import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine, jaccard
import time
from collections import Counter
import matplotlib.pyplot as plt
import os

class KMeans:
    def __init__(self, n_clusters, distance_metric='euclidean', max_iter=500, tol=1e-4):
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.sse_history = []
        self.n_iterations = 0
        self.convergence_time = 0

    def compute_distance(self, X, centroids):
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        elif self.distance_metric == 'cosine':
            distances = np.array([[cosine(x, c) for c in centroids] for x in X])
        elif self.distance_metric == 'jaccard':
            distances = np.array([[jaccard(x, c) for c in centroids] for x in X])
        return distances

    def compute_sse(self, X, labels, centroids):
        sse = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = centroids[i]
                if self.distance_metric == 'euclidean':
                    sse += np.sum((cluster_points - centroid) ** 2)
                elif self.distance_metric == 'cosine':
                    sse += np.sum([cosine(point, centroid) ** 2 for point in cluster_points])
                elif self.distance_metric == 'jaccard':
                    sse += np.sum([jaccard(point, centroid) ** 2 for point in cluster_points])
        return sse

    def fit(self, X):
        start_time = time.time()
        
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        prev_centroids = None
        self.n_iterations = 0
        
        while self.n_iterations < self.max_iter:
            # Assign points to clusters
            distances = self.compute_distance(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) 
                                    for i in range(self.n_clusters)])
            
            # Calculate SSE
            current_sse = self.compute_sse(X, self.labels_, new_centroids)
            self.sse_history.append(current_sse)
            
            # Check convergence conditions
            if prev_centroids is not None:
                centroid_shift = np.sum(np.abs(new_centroids - prev_centroids))
                if centroid_shift < self.tol:  # No change in centroid position
                    break
                if len(self.sse_history) > 1 and self.sse_history[-1] > self.sse_history[-2]:  # SSE increases
                    break
            
            self.centroids = new_centroids
            prev_centroids = new_centroids.copy()
            self.n_iterations += 1
        
        self.convergence_time = time.time() - start_time
        return self

def get_majority_label(cluster_points, true_labels):
    if len(cluster_points) == 0:
        return -1
    counter = Counter(true_labels[cluster_points])
    return counter.most_common(1)[0][0]

def evaluate_kmeans(X, y, n_clusters, distance_metric):
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, distance_metric=distance_metric)
    kmeans.fit(X)
    
    # Get cluster assignments and final SSE
    cluster_labels = kmeans.labels_
    final_sse = kmeans.sse_history[-1]
    
    # Assign majority class labels to clusters
    predicted_labels = np.zeros_like(cluster_labels)
    for i in range(n_clusters):
        cluster_points = np.where(cluster_labels == i)[0]
        majority_label = get_majority_label(cluster_points, y)
        predicted_labels[cluster_points] = majority_label
    
    # Calculate accuracy
    accuracy = accuracy_score(y, predicted_labels)
    
    return {
        'sse': final_sse,
        'accuracy': accuracy,
        'iterations': kmeans.n_iterations,
        'convergence_time': kmeans.convergence_time,
        'sse_history': kmeans.sse_history
    }

def plot_results(results, metrics):
    # Set style for better visualization
    plt.style.use('seaborn-v0_8')
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    accuracies = [results[m]['accuracy'] * 100 for m in metrics]  # Convert to percentage
    plt.bar(metrics, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.title('Clustering Accuracy by Distance Metric', pad=20, fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.ylim(0, 100)  # Set y-axis from 0 to 100%
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Iterations Comparison
    plt.figure(figsize=(10, 6))
    iterations = [results[m]['iterations'] for m in metrics]
    plt.bar(metrics, iterations, color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.title('Number of Iterations to Converge by Distance Metric', pad=20, fontsize=12)
    plt.ylabel('Number of Iterations', fontsize=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/iterations_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Convergence Time Comparison
    plt.figure(figsize=(10, 6))
    times = [results[m]['convergence_time'] for m in metrics]
    plt.bar(metrics, times, color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.title('Convergence Time by Distance Metric', pad=20, fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/convergence_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Read the data and labels
    data = pd.read_csv('data/data.csv', header=None)
    labels = pd.read_csv('data/label.csv', header=None)
    
    X = data.values
    y = labels.values.ravel()
    n_clusters = len(np.unique(y))
    
    # Run experiments with different distance metrics
    metrics = ['euclidean', 'cosine', 'jaccard']
    results = {}
    
    print("\nK-means Clustering Analysis with Different Distance Metrics")
    print("=" * 60)
    
    for metric in metrics:
        print(f"\nRunning K-means with {metric} distance:")
        results[metric] = evaluate_kmeans(X, y, n_clusters, metric)
        print(f"SSE: {results[metric]['sse']:.4f}")
        print(f"Accuracy: {results[metric]['accuracy']*100:.2f}%")
        print(f"Iterations to converge: {results[metric]['iterations']}")
        print(f"Convergence time: {results[metric]['convergence_time']:.4f} seconds")
    
    # Generate plots
    plot_results(results, metrics)
    print("\nVisualization files have been saved in the 'figures' directory.")

if __name__ == "__main__":
    main()
