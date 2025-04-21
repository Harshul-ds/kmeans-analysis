import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
# Remove original StandardScaler import, it will be added later in main if needed
# from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
# Keep scipy's euclidean import as well, though it might not be used directly now
from scipy.spatial.distance import euclidean as scipy_euclidean
# Import jaccard directly if needed elsewhere, but the custom one is primary
# from scipy.spatial.distance import jaccard as scipy_jaccard
from scipy.optimize import linear_sum_assignment # For optimal label mapping (more robust)

# ... rest of your imports (time, Counter, plotly, os, tabulate, warnings)
import time
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tabulate # Ensure tabulate is imported
import warnings # To suppress potential division warnings if needed

# Add StandardScaler import here, closer to where it might be used (or keep in main)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# --- Improved Distance Functions ---
def euclidean_distances(X, Y=None, squared=False):
    """Compute pairwise Euclidean distances between points."""
    # Ensure X and Y are numpy arrays
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)

    # Ensure inputs are 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # Handle single point comparisons efficiently
    if X.shape[0] == 1 and Y.shape[0] == 1:
         diff = X - Y
         dist_sq = np.sum(diff * diff, axis=1) # Result will be array of size 1
    elif X.shape[0] == 1:
        # Optimized case for a single point vs multiple points
        diff = X - Y
        dist_sq = np.sum(diff * diff, axis=1)
    elif Y.shape[0] == 1:
         # Optimized case for multiple points vs single point
        diff = X - Y
        dist_sq = np.sum(diff * diff, axis=1)
    else:
        # General case using dot product trick for efficiency
        # ||x - y||^2 = ||x||^2 - 2<x, y> + ||y||^2
        XX = np.sum(X * X, axis=1)[:, np.newaxis]
        YY = np.sum(Y * Y, axis=1) # Keep as 1D array for broadcasting
        distances = -2 * np.dot(X, Y.T)
        distances += XX
        distances += YY
        # Handle potential numerical errors leading to small negative values
        np.maximum(distances, 0, out=distances)
        dist_sq = distances

    if squared:
        return dist_sq
    else:
        # Avoid sqrt(0) warnings if possible, although maximum handles negative cases
        # return np.sqrt(dist_sq)
        # Use np.sqrt safely, potentially masking zeros if needed, but maximum should suffice
        return np.sqrt(dist_sq, out=np.zeros_like(dist_sq), where=dist_sq>0)

def cosine_distances(X, Y=None):
    """Compute pairwise cosine distances between points."""
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)

    # Ensure inputs are 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # Normalize vectors for cosine similarity
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors. Replace norm with 1; similarity will be 0.
    X_normalized = np.divide(X, X_norm, out=np.zeros_like(X), where=X_norm!=0)

    if Y is X:
        Y_normalized = X_normalized
        Y_norm = X_norm # Reuse norm if Y is X
    else:
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        Y_normalized = np.divide(Y, Y_norm, out=np.zeros_like(Y), where=Y_norm!=0)

    # Cosine similarity using dot product of normalized vectors
    similarities = np.dot(X_normalized, Y_normalized.T)

    # Clip to [-1, 1] to handle potential floating-point inaccuracies
    similarities = np.clip(similarities, -1.0, 1.0)

    # Cosine distance = 1 - cosine similarity
    distances = 1.0 - similarities

    # Ensure distance is non-negative (should be due to clip, but defensive)
    np.maximum(distances, 0, out=distances)

    return distances

def generalized_jaccard_distance(u, v):
    """Compute generalized Jaccard distance (Tanimoto coefficient) between two vectors."""
    u = np.asarray(u)
    v = np.asarray(v)

    # Ensure vectors are non-negative for standard Jaccard/Tanimoto interpretation
    # If your data can be negative, this needs careful consideration of the definition.
    # Shifting might alter relationships if negative values have meaning.
    min_val = min(np.min(u) if u.size > 0 else 0, np.min(v) if v.size > 0 else 0, 0)
    if min_val < 0:
        # print("Warning: Jaccard distance encountered negative values. Shifting data to be non-negative.")
        u_shifted = u - min_val
        v_shifted = v - min_val
    else:
        u_shifted = u
        v_shifted = v

    # Use shifted vectors for calculation
    numerator = np.sum(np.minimum(u_shifted, v_shifted))
    denominator = np.sum(np.maximum(u_shifted, v_shifted))

    # Handle edge cases
    if denominator == 0:
        # If both vectors (after shift) consist of only zeros
        if numerator == 0:
            return 0.0 # Jaccard distance is 0 if both vectors are identical (zero vector)
        else:
            # This case should technically not happen if numerator is sum of minimums
            return 1.0 # Or raise error, indicates inconsistency
    else:
        # Generalized Jaccard Index (Tanimoto coefficient)
        jaccard_index = numerator / denominator
        # Jaccard Distance = 1 - Jaccard Index
        return 1.0 - jaccard_index

def jaccard_distances(X, Y=None):
    """Compute pairwise Jaccard distances between rows of X and Y."""
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)

    # Ensure inputs are 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n_x = X.shape[0]
    n_y = Y.shape[0]
    distances = np.zeros((n_x, n_y))

    # Check for negative values once before the loop if applicable
    min_X = np.min(X) if X.size > 0 else 0
    min_Y = np.min(Y) if Y.size > 0 else 0
    if min(min_X, min_Y) < 0:
        print("Warning: Jaccard distance computation encountered negative values in input matrix/matrices."
              " Applying non-negativity shift internally for each pairwise calculation.")

    for i in range(n_x):
        for j in range(n_y):
            # The generalized_jaccard_distance function handles internal shifting if needed
            distances[i, j] = generalized_jaccard_distance(X[i], Y[j])

    return distances

# --- Improved K-Means++ Initialization ---
def kmeans_plusplus_init(X, n_clusters, random_state=None, n_local_trials=None):
    """Initialize cluster centers using k-means++ algorithm with local optimization.

    Modified from scikit-learn's implementation for clarity and robustness.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape

    # Set n_local_trials based on scikit-learn recommendation
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Choose the first center randomly from the data points
    center_id = np.random.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize distances using squared Euclidean distance for stability
    # Use the updated euclidean_distances function
    closest_dist_sq = euclidean_distances(centers[0:1], X, squared=True).flatten() # Ensure 1D
    # Ensure closest_dist_sq is non-negative
    closest_dist_sq = np.maximum(closest_dist_sq, 0)

    # Calculate the initial potential (sum of closest squared distances)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 centers
    for c in range(1, n_clusters):
        if current_pot == 0:
             # All points are coincident with the current centers.
             # Handle this edge case: pick remaining centers randomly but without duplicates if possible
             print(f"Warning: K-means++ potential is zero at step {c+1}. Points may be duplicates.")
             current_center_indices = [np.where((X == center).all(axis=1))[0][0] for center in centers[:c] if np.any((X == center).all(axis=1))]
             remaining_indices = np.setdiff1d(np.arange(n_samples), current_center_indices)
             if len(remaining_indices) >= (n_clusters - c):
                 candidate_ids = np.random.choice(remaining_indices, n_local_trials, replace=False)
             else: # Not enough unique points left, sample with replacement from remaining
                 candidate_ids = np.random.choice(remaining_indices, n_local_trials, replace=True)
        else:
             # Choose center candidates by sampling with probability proportional to distance squared
             # Ensure probabilities sum to 1 (or close) for sampling
             prob = closest_dist_sq / current_pot
             # Handle potential floating point inaccuracies
             prob = np.maximum(prob, 0)
             prob_sum = prob.sum()
             if prob_sum == 0: # All remaining points are at distance 0
                  prob = np.ones(n_samples) / n_samples # Assign equal probability
             else:
                  prob /= prob_sum

             # Use np.random.choice for sampling based on probabilities
             candidate_ids = np.random.choice(n_samples, size=n_local_trials, p=prob)

        # Decide which candidate is the best among the n_local_trials
        best_candidate = -1 # Use -1 to indicate no candidate chosen yet
        best_pot = current_pot # Initialize with current potential
        best_dist_sq = None # To store distances for the chosen candidate

        for trial in range(n_local_trials):
             # Compute distances (squared Euclidean) to the candidate
             new_dist_sq = euclidean_distances(X[candidate_ids[trial]].reshape(1, -1), X, squared=True).flatten() # Ensure 1D
             new_dist_sq = np.maximum(new_dist_sq, 0) # Ensure non-negative

             # Compute potential reduction if this candidate were chosen
             temp_closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
             new_pot = temp_closest_dist_sq.sum()

             # Store the best candidate found so far
             if best_candidate == -1 or new_pot < best_pot:
                 best_candidate = candidate_ids[trial]
                 best_pot = new_pot
                 best_dist_sq = new_dist_sq # Store the distances for the best candidate

        # Add the best candidate found among trials to the centers
        if best_candidate != -1: # Check if a best candidate was actually found
             centers[c] = X[best_candidate]
             current_pot = best_pot
             # Update closest distances using the stored distances for the best candidate
             if best_dist_sq is not None:
                closest_dist_sq = np.minimum(closest_dist_sq, best_dist_sq)
             else:
                 # This case means the loop didn't run or assign best_dist_sq (shouldn't happen)
                 # Recalculate defensively
                 new_center_dist_sq = euclidean_distances(centers[c:c+1], X, squared=True).flatten()
                 new_center_dist_sq = np.maximum(new_center_dist_sq, 0)
                 closest_dist_sq = np.minimum(closest_dist_sq, new_center_dist_sq)
                 current_pot = closest_dist_sq.sum() # Recalculate potential
        else:
             # Handle the unlikely case where no candidate improved potential (e.g., all trials landed on existing centers)
             print(f"Warning: K-means++ potential did not improve at step {c+1}. Picking random center.")
             current_center_indices = [np.where((X == center).all(axis=1))[0][0] for center in centers[:c] if np.any((X == center).all(axis=1))]
             available_indices = np.setdiff1d(np.arange(n_samples), current_center_indices)
             if len(available_indices) > 0:
                 centers[c] = X[np.random.choice(available_indices)]
             else: # All points are already centers (highly unlikely unless k >= n_samples)
                 centers[c] = X[np.random.randint(n_samples)] # Pick any point

             # Recalculate distances and potential after fallback
             new_center_dist_sq = euclidean_distances(centers[c:c+1], X, squared=True).flatten()
             new_center_dist_sq = np.maximum(new_center_dist_sq, 0)
             closest_dist_sq = np.minimum(closest_dist_sq, new_center_dist_sq)
             current_pot = closest_dist_sq.sum()

    return centers

# --- Enhanced K-Means Class ---
class KMeans:
    def __init__(self, n_clusters, distance_metric='euclidean', max_iter=300, tol=1e-4,
                 random_state=None, init_method='kmeans++', init_centroids=None): # Added init_centroids
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.max_iter = max_iter # Max iterations for the combined rule or 'max_iter_only'
        self.tol = tol # Tolerance for centroid stability
        self.random_state = random_state
        self.init_method = init_method
        self.init_centroids = init_centroids # Allow passing initial centroids
        self.centroids = None
        self.labels_ = None
        self.sse_history = [] # Tracks NATIVE SSE history (sum of distances, or squared for Euclidean)
        self.inertia_ = None # Final NATIVE SSE
        self.sse_euclidean_ = None # Final EUCLIDEAN SSE (sum of squared Euclidean distances) for comparison
        self.n_iter_ = 0
        self.stop_reason = "Not Run"

    def _compute_distances(self, X, centroids):
        """Compute distances between data points and centroids based on the chosen metric."""
        try:
            if self.distance_metric == 'euclidean':
                # For Euclidean, K-Means minimizes sum of SQUARED distances
                # So, return squared distances for internal use (inertia, stability check)
                # But the function itself can return non-squared if needed elsewhere
                return euclidean_distances(X, centroids, squared=True)
            elif self.distance_metric == 'cosine':
                # Cosine K-Means uses cosine distance (1 - similarity)
                return cosine_distances(X, centroids)
            elif self.distance_metric == 'jaccard':
                # Jaccard K-Means uses Jaccard distance
                # Ensure Jaccard gets non-negative data if necessary (handled in jaccard_distances)
                return jaccard_distances(X, centroids)
            else:
                print(f"Warning: Unknown distance metric '{self.distance_metric}'. Using Euclidean (squared).")
                return euclidean_distances(X, centroids, squared=True)
        except Exception as e:
            print(f"Error computing {self.distance_metric} distances: {e}")
            # Fallback or re-raise, here using Euclidean (squared) as fallback
            print("Falling back to squared Euclidean distances.")
            return euclidean_distances(X, centroids, squared=True)

    def _compute_inertia(self, X, centroids, labels):
        """Compute the native inertia (objective function value) for the current clustering.

        For Euclidean: Sum of squared distances to closest centroid.
        For Cosine/Jaccard: Sum of distances to closest centroid.
        """
        inertia = 0.0
        if centroids is None or labels is None:
             return np.inf # Or some indicator of invalid state

        # Get distances according to the *native* metric
        # Note: _compute_distances returns squared Euclidean, but raw Cosine/Jaccard
        distances = self._compute_distances(X, centroids)

        # Ensure distances are valid numbers
        if not np.all(np.isfinite(distances)):
             print(f"Warning: Non-finite distances encountered during inertia calculation ({self.distance_metric}).")
             # Handle non-finite values, e.g., replace with a large number or skip
             distances = np.nan_to_num(distances, nan=np.inf, posinf=np.inf, neginf=0) # Treat -inf as 0 distance?

        try:
            for k in range(self.n_clusters):
                cluster_points_mask = (labels == k)
                if np.any(cluster_points_mask):
                    # Get distances for points in cluster k to centroid k
                    point_distances = distances[cluster_points_mask, k]

                    # Sum distances (or squared distances for Euclidean)
                    # The distances from _compute_distances are already squared for Euclidean
                    inertia += np.sum(point_distances)
        except IndexError:
             print(f"Error: Indexing failed during inertia calculation. Distances shape: {distances.shape}, Labels unique: {np.unique(labels)}")
             return np.inf # Indicate error
        except Exception as e:
            print(f"Error during inertia calculation: {e}")
            return np.inf

        return inertia

    def _compute_sse_euclidean(self, X, centroids, labels):
        """Compute SSE using Euclidean distance ONLY, regardless of native metric, for comparison.

        SSE is defined as the sum of SQUARED Euclidean distances.
        """
        sse_euclidean = 0.0
        if centroids is None or labels is None or centroids.shape[0] != self.n_clusters:
            # print("Debug: Cannot compute Euclidean SSE - invalid centroids or labels.")
            return np.inf

        # Always use Euclidean distance here, ensure it's SQUARED for SSE definition
        try:
            euclidean_dists_sq = euclidean_distances(X, centroids, squared=True)

            if not np.all(np.isfinite(euclidean_dists_sq)):
                print("Warning: Non-finite squared Euclidean distances encountered during comparison SSE calculation.")
                euclidean_dists_sq = np.nan_to_num(euclidean_dists_sq, nan=np.inf, posinf=np.inf, neginf=0)

            for k in range(self.n_clusters):
                cluster_points_mask = (labels == k)
                if np.any(cluster_points_mask):
                    # Sum the *squared* Euclidean distances for points in this cluster to their centroid k
                    sse_euclidean += np.sum(euclidean_dists_sq[cluster_points_mask, k])

        except IndexError:
             print(f"Error: Indexing failed during Euclidean SSE calculation. Distances shape: {euclidean_dists_sq.shape}, Labels unique: {np.unique(labels)}")
             return np.inf # Indicate error
        except Exception as e:
            print(f"Error during Euclidean SSE calculation: {e}")
            return np.inf

        return sse_euclidean

    def fit(self, X, stopping_criterion='combined'):
        """
        Compute K-Means clustering.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster.
        stopping_criterion : {'combined', 'stable', 'sse_increase', 'max_iter_only'}
            Determines which condition(s) terminate the algorithm:
            - 'combined': Stop on stability (tol), SSE increase (optional check), or max_iter.
            - 'stable': Stop only when centroids are stable (tol) or max_iter.
            - 'sse_increase': Stop only on significant SSE increase or max_iter.
            - 'max_iter_only': Stop only when max_iter is reached.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.sse_history = [] # Reset history for this fit
        self.centroids = None # Ensure centroids are re-initialized if fit is called again
        self.labels_ = None
        self.inertia_ = None
        self.sse_euclidean_ = None
        self.n_iter_ = 0
        self.stop_reason = "Not Run"


        # 1. Initialization
        if self.init_centroids is not None:
            # Use provided initial centroids
            if self.init_centroids.shape == (self.n_clusters, n_features):
                self.centroids = np.copy(self.init_centroids)
                print("Using provided initial centroids.")
            else:
                print("Error: Provided init_centroids have incorrect shape. Falling back to init_method.")
                self.init_centroids = None # Clear invalid centroids

        if self.centroids is None: # If not provided or invalid
            if self.init_method == 'kmeans++':
                 self.centroids = kmeans_plusplus_init(X, self.n_clusters, random_state=self.random_state)
            elif self.init_method == 'random':
                 indices = np.random.choice(n_samples, self.n_clusters, replace=False)
                 self.centroids = X[indices]
            else: # Default to kmeans++
                 print(f"Warning: Unknown init_method '{self.init_method}'. Using kmeans++.")
                 self.centroids = kmeans_plusplus_init(X, self.n_clusters, random_state=self.random_state)


        # Check for valid initial centroids
        if self.centroids is None or self.centroids.shape != (self.n_clusters, n_features) or np.any(np.isnan(self.centroids)):
             print("Error: Initialization failed to produce valid centroids.")
             self.stop_reason = "Initialization Failed"
             self.n_iter_ = 0
             self.labels_ = np.zeros(n_samples, dtype=int) # Assign dummy labels
             self.inertia_ = np.inf
             self.sse_euclidean_ = np.inf
             return self # Return self to indicate failure

        # Initial assignment needed to calculate first inertia
        initial_distances = self._compute_distances(X, self.centroids)
        self.labels_ = np.argmin(initial_distances, axis=1)

        # Calculate initial inertia and SSE
        initial_inertia = self._compute_inertia(X, self.centroids, self.labels_)
        initial_sse_euclidean = self._compute_sse_euclidean(X, self.centroids, self.labels_)

        # Check for non-finite initial inertia/sse
        if not np.isfinite(initial_inertia):
             print("Warning: Initial native inertia is non-finite. Check distance metric or data.")
             initial_inertia = np.inf # Set to infinity if non-finite
        if not np.isfinite(initial_sse_euclidean):
             print("Warning: Initial Euclidean SSE is non-finite. Check data.")
             initial_sse_euclidean = np.inf

        self.sse_history.append(initial_inertia)
        self.inertia_ = initial_inertia
        self.sse_euclidean_ = initial_sse_euclidean # Store initial Euclidean SSE

        old_centroids = np.copy(self.centroids)
        stable = False # Initialize stable flag

        # 2. Iterative Refinement
        for i in range(self.max_iter):
            self.n_iter_ = i + 1 # Record iteration count (1-based)
            iteration_start_time = time.time() # Optional: time each iteration

            # --- Assignment Step ---
            distances = self._compute_distances(X, self.centroids)

            # Handle potential non-finite distances from computation
            if not np.all(np.isfinite(distances)):
                 print(f"Warning: Non-finite distances encountered at iteration {self.n_iter_} ({self.distance_metric}). Assigning points based on available finite distances.")
                 # Assign based on minimum finite distance, or assign to cluster 0 if all are infinite
                 finite_distances = np.where(np.isfinite(distances), distances, np.inf)
                 new_labels = np.argmin(finite_distances, axis=1)
                 # Check if any row had all infinite distances
                 all_inf_mask = np.all(np.isinf(finite_distances), axis=1)
                 if np.any(all_inf_mask):
                      print(f"  {np.sum(all_inf_mask)} points had infinite distance to all centroids. Assigning to cluster 0.")
                      new_labels[all_inf_mask] = 0 # Arbitrary assignment
            else:
                 new_labels = np.argmin(distances, axis=1)

            self.labels_ = new_labels

            # --- Update Step ---
            new_centroids = np.zeros_like(self.centroids)
            points_in_cluster = np.zeros(self.n_clusters, dtype=bool)

            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                if cluster_points.shape[0] > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                    points_in_cluster[k] = True
                else:
                    # Handle empty cluster: Keep the old centroid position
                    print(f"Warning: Cluster {k} became empty at iteration {self.n_iter_}. Keeping previous centroid.")
                    new_centroids[k] = old_centroids[k]
                    points_in_cluster[k] = False # Mark cluster as effectively empty

            # Check for NaN centroids after update
            if np.any(np.isnan(new_centroids)):
                 print(f"Error: NaN centroid detected at iteration {self.n_iter_}. Stopping.")
                 self.stop_reason = "NaN Centroid Encountered"
                 self.centroids = old_centroids # Revert to last good centroids
                 # Recompute labels based on old_centroids for consistency
                 old_distances = self._compute_distances(X, self.centroids)
                 self.labels_ = np.argmin(old_distances, axis=1)
                 self.inertia_ = self.sse_history[-1] if self.sse_history and np.isfinite(self.sse_history[-1]) else np.inf
                 self.sse_euclidean_ = self._compute_sse_euclidean(X, self.centroids, self.labels_)
                 break # Stop iteration

            self.centroids = new_centroids

            # --- Calculate Native Inertia for this iteration ---
            current_inertia = self._compute_inertia(X, self.centroids, self.labels_)

            if not np.isfinite(current_inertia):
                 print(f"Warning: Non-finite native inertia ({current_inertia}) calculated at iteration {self.n_iter_}. Using last finite value or infinity.")
                 current_inertia = self.sse_history[-1] if self.sse_history and np.isfinite(self.sse_history[-1]) else np.inf

            self.sse_history.append(current_inertia)
            self.inertia_ = current_inertia # Update final inertia each iteration

            # --- Calculate Comparison Euclidean SSE for this iteration ---
            self.sse_euclidean_ = self._compute_sse_euclidean(X, self.centroids, self.labels_)

            # --- Check Stopping Criteria ---
            # 1. Centroid Stability
            centroid_shift_sq = np.sum((self.centroids - old_centroids)**2)
            stable = centroid_shift_sq < self.tol

            # 2. SSE Increase/Plateau
            sse_increased = False
            if len(self.sse_history) > 1 and np.isfinite(self.sse_history[-2]) and np.isfinite(self.sse_history[-1]):
                sse_change = self.sse_history[-1] - self.sse_history[-2] # Positive if SSE increased
                if sse_change > 1e-9: # Stop on significant increase (allow tiny FP fluctuations)
                    sse_increased = True
                    print(f"Warning: Native Inertia increased at iteration {self.n_iter_} ({self.distance_metric}). "
                          f"From {self.sse_history[-2]:.4e} to {self.sse_history[-1]:.4e}")

            # Apply stopping logic
            stop = False
            if stopping_criterion == 'combined':
                if stable:
                    self.stop_reason = "Centroids Stable"
                    stop = True
                # Optional: Stop on SSE increase? Usually just warn.
            elif stopping_criterion == 'stable':
                if stable:
                    self.stop_reason = "Centroids Stable"
                    stop = True
            elif stopping_criterion == 'sse_increase':
                 if sse_increased and self.n_iter_ > 1: # Don't stop on first iter increase
                    self.stop_reason = "SSE Increased"
                    stop = True
            # 'max_iter_only' handled by loop limit

            if stop:
                 break # Exit the loop

            old_centroids = np.copy(self.centroids)
            # End of loop iteration

        else: # Loop finished without break (hit max_iter)
            if self.stop_reason == "Not Run": # Check if a stop reason wasn't set inside loop
                if stopping_criterion == 'max_iter_only':
                    self.stop_reason = "Max Iterations Reached"
                elif stable: # If stable happened on last iter for 'combined' or 'stable'
                     self.stop_reason = "Centroids Stable"
                else: # Otherwise max_iter was the reason
                     self.stop_reason = "Max Iterations Reached"


        # Final checks (value already set, just ensure finite)
        if self.inertia_ is None or not np.isfinite(self.inertia_):
             self.inertia_ = np.inf
        if self.sse_euclidean_ is None or not np.isfinite(self.sse_euclidean_):
             self.sse_euclidean_ = np.inf

        return self

# --- Improved Label Assignment ---
def assign_labels_to_clusters(kmeans_labels, true_labels, n_clusters):
    """
    Assign class labels to clusters using the Hungarian algorithm for optimal matching.
    """
    # Create a mapping matrix: rows=clusters, cols=classes
    mapping_matrix = np.zeros((n_clusters, n_clusters))
    
    # Count occurrences of each class in each cluster
    for i in range(n_clusters):
        for j in range(n_clusters):
            mapping_matrix[i, j] = np.sum((kmeans_labels == i) & (true_labels == j))
    
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-mapping_matrix)
    
    # Create a dictionary to map cluster indices to class labels
    cluster_to_label = {row_ind[i]: col_ind[i] for i in range(n_clusters)}
    
    # Assign labels to clusters
    inferred_labels = np.array([cluster_to_label[label] for label in kmeans_labels])
    
    return inferred_labels

def evaluate_kmeans(X, y_true, n_clusters, distance_metrics, stopping_criteria, n_init=10, max_iter=300, tol=1e-4, random_state=None):
    """
    Evaluates KMeans performance with different distance metrics and stopping criteria.

    Parameters:
        X (array-like): Data points.
        y_true (array-like): True labels (if available, for accuracy calculation).
        n_clusters (int): Number of clusters.
        distance_metrics (list): List of distance metric names to evaluate.
        stopping_criteria (list): List of stopping criteria strings to evaluate.
        n_init (int): Number of times to run KMeans with different centroid seeds.
        max_iter (int): Maximum iterations for a single run.
        tol (float): Tolerance for centroid stability check.
        random_state (int): Seed for reproducibility.

    Returns:
        list: A list of dictionaries, each containing results for one combination
              of distance metric and stopping criterion.
    """
    results = []
    rng = np.random.RandomState(random_state) # Use a RandomState object for seeding runs

    print(f"--- Starting K-Means Evaluation ---")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Distance Metrics: {distance_metrics}")
    print(f"Stopping Criteria: {stopping_criteria}")
    print(f"n_init: {n_init}, max_iter: {max_iter}, tol: {tol:.1e}")
    print("-" * 30)

    for metric in distance_metrics:
        for criterion in stopping_criteria:
            print(f"Evaluating Metric: '{metric}', Stopping Criterion: '{criterion}'...")
            best_inertia = np.inf
            best_sse_euclidean = np.inf
            best_model = None
            run_times = []
            all_n_iters = []
            all_stop_reasons = []

            for i in range(n_init):
                init_seed = rng.randint(np.iinfo(np.int32).max) # Generate seed for this specific init run
                model = KMeans(n_clusters=n_clusters,
                               distance_metric=metric,
                               max_iter=max_iter,
                               tol=tol,
                               random_state=init_seed, # Use specific seed for this run
                               init_method='kmeans++') # Always use kmeans++ for fair comparison across runs

                start_time = time.time()
                model.fit(X, stopping_criterion=criterion)
                end_time = time.time()
                run_time = end_time - start_time
                run_times.append(run_time)
                all_n_iters.append(model.n_iter_)
                all_stop_reasons.append(model.stop_reason)


                # Check if this run is better (lower native inertia)
                # Handle potential inf inertia
                current_inertia = model.inertia_ if np.isfinite(model.inertia_) else np.inf

                if current_inertia < best_inertia:
                    best_inertia = current_inertia
                    best_sse_euclidean = model.sse_euclidean_ if np.isfinite(model.sse_euclidean_) else np.inf
                    best_model = model
                    #print(f"  (Run {i+1}/{n_init}: New best inertia: {best_inertia:.4e} in {model.n_iter_} iters, reason: {model.stop_reason})")


            if best_model is None:
                print(f"  Warning: No valid model found for {metric}/{criterion} after {n_init} initializations.")
                # Store placeholder results
                result_entry = {
                    'metric': metric,
                    'criterion': criterion,
                    'inertia': np.inf,
                    'sse_euclidean': np.inf,
                    'n_iter': np.nan,
                    'stop_reason': "All Inits Failed",
                    'accuracy': np.nan,
                    'avg_runtime': np.mean(run_times) if run_times else np.nan,
                    'avg_n_iter': np.mean(all_n_iters) if all_n_iters else np.nan,
                    'stop_reasons_summary': dict(Counter(all_stop_reasons))
                }
            else:
                # Calculate accuracy if y_true is provided
                accuracy = np.nan
                if y_true is not None:
                     # Simplified accuracy: check if majority class in cluster matches true label
                     # Note: This is a basic measure and may not be ideal for all scenarios.
                     # A better approach uses adjusted rand index or normalized mutual information.
                     pred_labels = best_model.labels_
                     # Map cluster labels to true labels (requires scipy)
                     try:
                         from scipy.optimize import linear_sum_assignment
                         from sklearn import metrics # Need metrics here
                         contingency_matrix = metrics.cluster.contingency_matrix(y_true, pred_labels)
                         row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
                         accuracy = contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)
                     except ImportError:
                          print("Warning: Scipy not found. Cannot compute optimal label mapping for accuracy.")
                          # Basic majority vote accuracy (less reliable)
                          correct = 0
                          for k in range(n_clusters):
                              cluster_mask = (pred_labels == k)
                              if np.any(cluster_mask):
                                  true_labels_in_cluster = y_true[cluster_mask]
                                  # most_common_true_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
                                  # This isn't quite right - needs proper mapping. Placeholder.
                                  pass # Cannot compute accuracy without mapping

                          print("  Using placeholder NaN for accuracy due to missing scipy or simple method.")
                          accuracy = np.nan # Fallback

                accuracy_str = f"{accuracy:.4f}" if not np.isnan(accuracy) else 'NaN'
                result_entry = {
                    'metric': metric,
                    'criterion': criterion,
                    'inertia': best_inertia, # Native SSE from the best run
                    'sse_euclidean': best_sse_euclidean, # Comparison SSE from the best run
                    'n_iter': best_model.n_iter_, # Iterations of the best run
                    'stop_reason': best_model.stop_reason, # Stop reason of the best run
                    'accuracy': accuracy,
                    'avg_runtime': np.mean(run_times),
                    'avg_n_iter': np.mean(all_n_iters),
                    'stop_reasons_summary': dict(Counter(all_stop_reasons)),
                    'best_model': best_model # Keep the best model itself for potential plotting/analysis
                }
                print(f"  Finished {metric}/{criterion}. Best Inertia: {best_inertia:.4e}, "
                      f"Euclidean SSE: {best_sse_euclidean:.4e}, Acc: {accuracy_str}, Avg Iters: {result_entry['avg_n_iter']:.1f}")


            results.append(result_entry)
        print("-" * 20) # Separator between metrics

    print("--- K-Means Evaluation Complete ---")
    return results

def create_tables(results):
    """
    Creates pandas DataFrames summarizing the K-Means evaluation results.

    Parameters:
        results (list): The list of dictionaries returned by evaluate_kmeans.

    Returns:
        dict: A dictionary containing various pandas DataFrames ('summary', 'sse_comparison', 'runtime_iterations').
    """
    if not results:
        print("Warning: No results provided to create_tables.")
        return {'summary': pd.DataFrame(), 'sse_comparison': pd.DataFrame(), 'runtime_iterations': pd.DataFrame(), 'stop_reasons': pd.DataFrame()}

    df = pd.DataFrame(results)

    # --- Summary Table ---
    # Select and rename columns for clarity
    summary_df = df[['metric', 'criterion', 'inertia', 'sse_euclidean', 'accuracy', 'n_iter', 'stop_reason', 'avg_runtime', 'avg_n_iter']].copy()
    summary_df.rename(columns={
        'metric': 'Distance Metric',
        'criterion': 'Stopping Criterion',
        'inertia': 'Native SSE (Best Run)',
        'sse_euclidean': 'Euclidean SSE (Best Run)',
        'accuracy': 'Accuracy (Best Run)',
        'n_iter': 'Iterations (Best Run)',
        'stop_reason': 'Stop Reason (Best Run)',
        'avg_runtime': 'Avg. Runtime (s)',
        'avg_n_iter': 'Avg. Iterations'
    }, inplace=True)

    # Sort for better readability
    summary_df.sort_values(by=['Distance Metric', 'Stopping Criterion'], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)


    # --- SSE Comparison Table ---
    # Pivot table to compare Native vs Euclidean SSE across metrics and criteria
    try:
        sse_pivot = pd.pivot_table(df,
                                   values=['inertia', 'sse_euclidean'],
                                   index='metric',
                                   columns='criterion',
                                   aggfunc='first') # 'first' assumes one result per metric/criterion combo
        sse_pivot.columns = [f"{sse_type}_{crit}" for sse_type, crit in sse_pivot.columns] # Flatten MultiIndex
        sse_pivot.rename(columns=lambda x: x.replace('inertia', 'NativeSSE').replace('sse_euclidean', 'EuclideanSSE'), inplace=True)
    except Exception as e:
         print(f"Could not create SSE pivot table: {e}")
         sse_pivot = pd.DataFrame() # Empty fallback


    # --- Runtime and Iterations Table ---
    # Pivot table to compare runtime and iterations
    try:
        runtime_iter_pivot = pd.pivot_table(df,
                                            values=['avg_runtime', 'avg_n_iter', 'n_iter'],
                                            index='metric',
                                            columns='criterion',
                                            aggfunc='first')
        runtime_iter_pivot.columns = [f"{val_type}_{crit}" for val_type, crit in runtime_iter_pivot.columns] # Flatten MultiIndex
        runtime_iter_pivot.rename(columns=lambda x: x.replace('avg_runtime', 'AvgRuntime')
                                                  .replace('avg_n_iter', 'AvgIters')
                                                  .replace('n_iter', 'BestRunIters'), inplace=True)
    except Exception as e:
         print(f"Could not create Runtime/Iterations pivot table: {e}")
         runtime_iter_pivot = pd.DataFrame() # Empty fallback


    # --- Stop Reason Summary Table ---
    # Requires extracting the summary dicts
    stop_reasons_list = []
    for _, row in df.iterrows():
        entry = {'Metric': row['metric'], 'Criterion': row['criterion']}
        entry.update(row['stop_reasons_summary']) # Add counts from the dict
        stop_reasons_list.append(entry)

    stop_reasons_df = pd.DataFrame(stop_reasons_list).fillna(0) # Fill NaN with 0 counts
    # Ensure standard columns exist even if no runs triggered them
    standard_reasons = ['Centroids Stable', 'Max Iterations Reached', 'SSE Increased', 'Initialization Failed', 'NaN Centroid Encountered', 'All Inits Failed']
    for reason in standard_reasons:
        if reason not in stop_reasons_df.columns:
            stop_reasons_df[reason] = 0
    # Reorder columns
    cols_order = ['Metric', 'Criterion'] + [r for r in standard_reasons if r in stop_reasons_df.columns]
    stop_reasons_df = stop_reasons_df[cols_order]
    stop_reasons_df.sort_values(by=['Metric', 'Criterion'], inplace=True)
    stop_reasons_df.reset_index(drop=True, inplace=True)


    print("\n--- Generated Tables ---")
    print("\nSummary Results:")
    print(summary_df.to_string())
    if not sse_pivot.empty:
        print("\nSSE Comparison (Native vs Euclidean):")
        print(sse_pivot.to_string())
    if not runtime_iter_pivot.empty:
        print("\nRuntime and Iterations Comparison:")
        print(runtime_iter_pivot.to_string())
    print("\nStop Reason Summary (Counts over n_init runs):")
    print(stop_reasons_df.to_string())
    print("-" * 25)

    return {
        'summary': summary_df,
        'sse_comparison': sse_pivot,
        'runtime_iterations': runtime_iter_pivot,
        'stop_reasons': stop_reasons_df
    }

def plot_results(results, tables):
    """
    Generates plots visualizing the K-Means evaluation results using Plotly.

    Parameters:
        results (list): The list of dictionaries returned by evaluate_kmeans.
        tables (dict): The dictionary of pandas DataFrames returned by create_tables.
    """
    # Check if results are empty or the essential 'summary' table is missing/empty
    if not results or 'summary' not in tables or tables['summary'].empty:
        print("Warning: Insufficient data provided to plot_results. Skipping plot generation.")
        return

    # Create figures directory if it doesn't exist
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created directory: {figures_dir}")

    # Proceed with plotting if data is valid
    df = tables['summary']
    df_full = pd.DataFrame(results) # Need the full results for SSE history

    # --- Plot 1: Native SSE vs. Euclidean SSE (Bar Chart) ---
    fig1 = go.Figure()
    metrics = df['Distance Metric'].unique()
    criteria = df['Stopping Criterion'].unique()

    # Create grouped bars for Native and Euclidean SSE
    for criterion in criteria:
        df_crit = df[df['Stopping Criterion'] == criterion]
        fig1.add_trace(go.Bar(
            name=f'Native SSE ({criterion})',
            x=df_crit['Distance Metric'],
            y=df_crit['Native SSE (Best Run)'],
            text=df_crit['Native SSE (Best Run)'].apply(lambda x: f'{x:.2e}'),
            textposition='auto'
        ))
        fig1.add_trace(go.Bar(
            name=f'Euclidean SSE ({criterion})',
            x=df_crit['Distance Metric'],
            y=df_crit['Euclidean SSE (Best Run)'],
            text=df_crit['Euclidean SSE (Best Run)'].apply(lambda x: f'{x:.2e}'),
            textposition='auto'
        ))

    fig1.update_layout(
        title='Comparison of Final SSE (Native vs. Euclidean) by Metric and Criterion',
        xaxis_title='Distance Metric',
        yaxis_title='Sum of Squared Errors (SSE)',
        barmode='group',
        yaxis_type="log", # Use log scale if SSE values vary widely
        legend_title='SSE Type (Stopping Criterion)'
    )
    # Save the figure to the figures directory
    fig1.write_html(os.path.join(figures_dir, "sse_comparison.html"))
    fig1.write_image(os.path.join(figures_dir, "sse_comparison.png"))
    fig1.show()


    # --- Plot 2: Accuracy vs. Distance Metric (Grouped by Criterion) ---
    if 'Accuracy (Best Run)' in df.columns and df['Accuracy (Best Run)'].notna().any():
        fig2 = go.Figure()
        for criterion in criteria:
             df_crit = df[df['Stopping Criterion'] == criterion]
             fig2.add_trace(go.Bar(
                 name=criterion,
                 x=df_crit['Distance Metric'],
                 y=df_crit['Accuracy (Best Run)'],
                 text=df_crit['Accuracy (Best Run)'].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'NaN'),
                 textposition='auto'
             ))

        fig2.update_layout(
            title='Clustering Accuracy by Distance Metric and Stopping Criterion',
            xaxis_title='Distance Metric',
            yaxis_title='Accuracy (Higher is Better)',
            barmode='group',
            yaxis_range=[0, 1.05], # Set range for accuracy
            legend_title='Stopping Criterion'
        )
        # Save the figure to the figures directory
        fig2.write_html(os.path.join(figures_dir, "accuracy_comparison.html"))
        fig2.write_image(os.path.join(figures_dir, "accuracy_comparison.png"))
        fig2.show()
    else:
         print("Skipping Accuracy plot: No valid accuracy data found.")


    # --- Plot 3: SSE Convergence Curves (Native SSE) ---
    fig3 = go.Figure()
    for index, row in df_full.iterrows():
         metric = row['metric']
         criterion = row['criterion']
         model = row.get('best_model') # Get the model saved in results
         if model and hasattr(model, 'sse_history') and model.sse_history:
             # Ensure history is finite and starts from iteration 0 (or 1 if preferred)
             history = np.array(model.sse_history)
             history = history[np.isfinite(history)] # Filter out non-finite values
             iterations = list(range(1, len(history) + 1)) # Iterations 1 to N

             # Add trace if history is valid
             if len(iterations) > 0:
                 fig3.add_trace(go.Scatter(
                     x=iterations,
                     y=history,
                     mode='lines+markers',
                     name=f'{metric} ({criterion}) - {model.stop_reason}'
                 ))

    fig3.update_layout(
        title='SSE Convergence Curves (Native SSE from Best Run)',
        xaxis_title='Iteration Number',
        yaxis_title='Native Sum of Squared Errors (SSE)',
        yaxis_type="log", # Often useful for SSE convergence
        legend_title='Metric (Criterion) - Stop Reason'
    )
    # Save the figure to the figures directory
    fig3.write_html(os.path.join(figures_dir, "sse_convergence.html"))
    fig3.write_image(os.path.join(figures_dir, "sse_convergence.png"))
    fig3.show()


    # --- Plot 4: Average Runtime vs. Metric/Criterion ---
    fig4 = go.Figure()
    for criterion in criteria:
        df_crit = df[df['Stopping Criterion'] == criterion]
        fig4.add_trace(go.Bar(
            name=criterion,
            x=df_crit['Distance Metric'],
            y=df_crit['Avg. Runtime (s)'],
            text=df_crit['Avg. Runtime (s)'].apply(lambda x: f'{x:.3f}s'),
            textposition='auto'
        ))

    fig4.update_layout(
        title='Average Runtime by Distance Metric and Stopping Criterion',
        xaxis_title='Distance Metric',
        yaxis_title='Average Runtime (seconds)',
        barmode='group',
        legend_title='Stopping Criterion'
    )
    # Save the figure to the figures directory
    fig4.write_html(os.path.join(figures_dir, "runtime_comparison.html"))
    fig4.write_image(os.path.join(figures_dir, "runtime_comparison.png"))
    fig4.show()
    
    # Also save the data visualization to the figures directory
    if os.path.exists('data_visualization.png'):
        import shutil
        shutil.copy('data_visualization.png', os.path.join(figures_dir, 'data_visualization.png'))
        print(f"Copied data visualization to {figures_dir}")
    
    print(f"All plots saved to {figures_dir} directory")

# --- Main Execution Block --- (Updated)
if __name__ == "__main__":
    # --- Configuration ---
    N_CLUSTERS = 4
    N_FEATURES = 2  # Reduced to 2 features for better visualization
    N_SAMPLES = 1000  # Increased sample size
    RANDOM_STATE = 42  # For reproducibility of data generation and runs

    # Create a more complex dataset that will favor different distance metrics
    from sklearn.datasets import make_blobs, make_moons, make_circles
    import matplotlib.pyplot as plt
    
    # Generate different types of clusters
    # 1. Standard Gaussian clusters (favors Euclidean)
    X1, y1 = make_blobs(n_samples=int(N_SAMPLES*0.4),
                        centers=2,
                        n_features=N_FEATURES,
                        cluster_std=0.5,
                        random_state=RANDOM_STATE)
    
    # 2. Moon-shaped clusters (favors non-Euclidean metrics)
    X2, y2 = make_moons(n_samples=int(N_SAMPLES*0.3), 
                        noise=0.08, 
                        random_state=RANDOM_STATE)
    # Shift the moons to separate them from the blobs
    X2[:, 0] += 5
    X2[:, 1] += 5
    y2 += 2  # Labels 2, 3
    
    # 3. Concentric circles (favors non-Euclidean metrics)
    X3, y3 = make_circles(n_samples=int(N_SAMPLES*0.3), 
                         noise=0.05, 
                         factor=0.5, 
                         random_state=RANDOM_STATE)
    # Shift the circles to separate them from other clusters
    X3[:, 0] -= 5
    X3[:, 1] -= 5
    y3 += 4  # Labels 4, 5
    
    # Combine all datasets
    X = np.vstack([X1, X2, X3])
    y_true = np.concatenate([y1, y2, y3])
    
    # Optional: Visualize the generated data
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.7)
    plt.title('Generated Dataset with Mixed Cluster Shapes')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.savefig('data_visualization.png')
    
    # Scale features (important for distance metrics)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()  # Changed to StandardScaler for better performance with mixed shapes
    X_scaled = scaler.fit_transform(X)
    
    # Use scaled data for clustering
    data_to_cluster = X_scaled
    
    # Save the data for future reference
    data_df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    data_df['true_label'] = y_true
    data_df.to_csv('data/data.csv', index=False)

    # Define metrics and criteria to test
    DISTANCE_METRICS = ['euclidean', 'cosine', 'jaccard']
    STOPPING_CRITERIA = ['combined', 'stable', 'max_iter_only']
    
    # Adjust n_clusters to match our dataset (we have 6 actual clusters)
    ACTUAL_CLUSTERS = len(np.unique(y_true))
    print(f"Dataset contains {ACTUAL_CLUSTERS} actual clusters")
    
    # --- Evaluation ---
    evaluation_results = evaluate_kmeans(
        X=data_to_cluster,
        y_true=y_true,  # Provide true labels for accuracy calculation
        n_clusters=ACTUAL_CLUSTERS,  # Use the actual number of clusters
        distance_metrics=DISTANCE_METRICS,
        stopping_criteria=STOPPING_CRITERIA,
        n_init=10,  # Increased for better chance of finding optimal solution
        max_iter=200,  # Increased for more complex data
        tol=1e-4,  # Tolerance for stability check
        random_state=RANDOM_STATE  # Seed for the evaluation function's RNG
    )

    # --- Reporting ---
    if evaluation_results:
        # Create summary tables
        result_tables = create_tables(evaluation_results)

        # Generate plots
        plot_results(evaluation_results, result_tables)
    else:
        print("Evaluation did not produce any results.")

    print("\n--- Script Finished ---")