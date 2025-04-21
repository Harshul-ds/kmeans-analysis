import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine, jaccard
import time
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    # Color scheme
    colors = {'euclidean': '#2ecc71', 'cosine': '#3498db', 'jaccard': '#e74c3c'}
    
    # 1. Accuracy Comparison
    fig = go.Figure()
    accuracies = [results[m]['accuracy'] * 100 for m in metrics]
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=accuracies,
            text=[f'{x:.1f}%' for x in accuracies],
            textposition='auto',
            marker_color=[colors[m] for m in metrics],
            hovertemplate='%{y:.1f}%<extra></extra>'
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Clustering Accuracy by Distance Metric',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100],
        template='plotly_white',
        height=600,
        width=1000,
        showlegend=False,
        annotations=[
            dict(
                x=metric,
                y=acc,
                text=f'{acc:.1f}%',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=14)
            ) for metric, acc in zip(metrics, accuracies)
        ]
    )
    
    fig.update_xaxes(tickangle=0, tickfont=dict(size=14))
    fig.update_yaxes(gridwidth=1, gridcolor='LightGray')
    
    fig.write_html("figures/accuracy_comparison.html")
    fig.write_image("figures/accuracy_comparison.png", scale=2)
    
    # 2. Performance Metrics
    fig = go.Figure()
    
    times = [results[m]['convergence_time'] for m in metrics]
    iterations = [results[m]['iterations'] for m in metrics]
    
    # Add iterations bars
    fig.add_trace(
        go.Bar(
            name='Iterations',
            x=metrics,
            y=iterations,
            text=iterations,
            textposition='auto',
            marker_color=[colors[m] for m in metrics],
            opacity=0.9,
            hovertemplate='Iterations: %{y}<extra></extra>'
        )
    )
    
    # Add convergence time line and markers
    fig.add_trace(
        go.Scatter(
            name='Time (seconds)',
            x=metrics,
            y=times,
            text=[f'{t:.1f}s' for t in times],
            mode='lines+markers+text',
            marker=dict(size=12),
            line=dict(width=3, dash='dot'),
            textposition='top center',
            yaxis='y2',
            hovertemplate='Time: %{y:.1f}s<extra></extra>'
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Convergence Performance by Distance Metric',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        yaxis=dict(
            title='Number of Iterations',
            gridwidth=1,
            gridcolor='LightGray',
            side='left'
        ),
        yaxis2=dict(
            title='Time (seconds)',
            overlaying='y',
            side='right',
            gridwidth=1,
            gridcolor='LightGray'
        ),
        template='plotly_white',
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.write_html("figures/performance_metrics.html")
    fig.write_image("figures/performance_metrics.png", scale=2)
    
    # 3. SSE Convergence History (normalized)
    fig = go.Figure()
    
    for metric in metrics:
        history = results[metric]['sse_history']
        # Normalize SSE values relative to initial SSE
        normalized_history = [sse/history[0] for sse in history]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(history))),
                y=normalized_history,
                name=metric.capitalize(),
                line=dict(color=colors[metric], width=2),
                mode='lines',
                hovertemplate='Relative SSE: %{y:.3f}<br>Iteration: %{x}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title={
            'text': 'Normalized SSE Convergence History',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Iteration",
        yaxis_title="Relative SSE (normalized)",
        yaxis_type="log",
        template='plotly_white',
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    fig.write_html("figures/convergence_history.html")
    fig.write_image("figures/convergence_history.png", scale=2)

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
    print("\nVisualization files have been saved in the 'figures' directory:")
    print("1. accuracy_comparison.html (interactive) and .png")
    print("2. performance_metrics.html (interactive) and .png")
    print("3. convergence_history.html (interactive) and .png")

if __name__ == "__main__":
    main()
