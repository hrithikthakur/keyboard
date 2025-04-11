# import torch
import cupy as cp
# import triton
import numpy as np
import time
# import json
import argparse
import importlib
import test
import math
from testing import Testing
importlib.reload(test)
from test import testdata_kmeans, testdata_knn, testdata_ann

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Select device: cpu or cuda")
Testing.set_cfg_via_argparse(parser)
args = parser.parse_args()
device = args.device

t = Testing(args)

print(f"Using device: {device}")

def process_distance_func(arg):
    if arg == "cosine":
        return distance_cosine
    elif arg == "l2":
        return distance_l2
    elif arg == "dot":
        return distance_dot
    elif arg == "manhattan":
        return distance_manhattan
    else:
        raise ValueError("Unknown distance function specified")

block_dim = (16, 16)
# ---------------------------------------------------------------------------
# Cosine distance kernel:
# For each pair (i,j), compute:
#    dot = sum_d( X[i,d] * Y[j,d] )
#    normX = sqrt(sum_d( X[i,d]^2 ))
#    normY = sqrt(sum_d( Y[j,d]^2 ))
# and then output = 1 - dot/(normX*normY)
# ---------------------------------------------------------------------------
# Cosine distance kernel (with unrolling)
cosine_kernel = cp.RawKernel(r'''
extern "C" __global__
void cosine_distance(const float* X, const float* Y, float* out, int N, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // index over X rows
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // index over Y rows
    if (i < N && j < K) {
        float dot = 0.0f, normX = 0.0f, normY = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            float a = X[i * D + d];
            float b = Y[j * D + d];
            dot += a * b;
            normX += a * a;
            normY += b * b;
        }
        normX = sqrtf(normX);
        normY = sqrtf(normY);
        if (normX < 1e-6f) normX = 1e-6f;
        if (normY < 1e-6f) normY = 1e-6f;
        float cos_sim = dot / (normX * normY);
        out[i * K + j] = 1.0f - cos_sim;
    }
}
''', 'cosine_distance')

# L2 (Euclidean) distance kernel (computing squared distances; no sqrt)
l2s_kernel = cp.RawKernel(r'''
extern "C" __global__
void l2s_distance(const float* X, const float* Y, float* out, int N, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < N && j < K) {
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            float diff = X[i * D + d] - Y[j * D + d];
            sum += diff * diff;
        }
        out[i * K + j] = sqrtf(sum);
    }
}
''', 'l2s_distance')

# Dot product kernel (with negative sign)
dot_kernel = cp.RawKernel(r'''
extern "C" __global__
void dot_distance(const float* X, const float* Y, float* out, int N, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < N && j < K) {
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            sum += X[i * D + d] * Y[j * D + d];
        }
        out[i * K + j] = -sum;
    }
}
''', 'dot_distance')

# Manhattan (L1) distance kernel
manhattan_kernel = cp.RawKernel(r'''
extern "C" __global__
void manhattan_distance(const float* X, const float* Y, float* out, int N, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < N && j < K) {
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            float diff = X[i * D + d] - Y[j * D + d];
            sum += fabsf(diff);
        }
        out[i * K + j] = sum;
    }
}
''', 'manhattan_distance')

# -----------------------------
# Distance Function Wrappers
# -----------------------------
def distance_cosine(X, Y):
    # Ensure inputs are float32 on GPU
    N, D = X.shape
    K, D2 = Y.shape
    assert D == D2, "Dimension mismatch"
    X = X.astype(cp.float32, copy=False)
    Y = Y.astype(cp.float32, copy=False)
    out = cp.empty((N, K), dtype=cp.float32)
    grid_dim = ((N + block_dim[0] - 1) // block_dim[0],
                (K + block_dim[1] - 1) // block_dim[1])
    cosine_kernel(grid_dim, block_dim, (X, Y, out, cp.int32(N), cp.int32(K), cp.int32(D)))
    return out

def distance_l2(X, Y):
    # Compute squared Euclidean distances
    N, D = X.shape
    K, D2 = Y.shape
    assert D == D2, "Dimension mismatch"
    X = X.astype(cp.float32, copy=False)
    Y = Y.astype(cp.float32, copy=False)
    out = cp.empty((N, K), dtype=cp.float32)
    grid_dim = ((N + block_dim[0] - 1) // block_dim[0],
                (K + block_dim[1] - 1) // block_dim[1])
    l2s_kernel(grid_dim, block_dim, (X, Y, out, cp.int32(N), cp.int32(K), cp.int32(D)))
    # Note: for k-NN and k-means, argmin is the same with squared distances.
    return out

def distance_dot(X, Y):
    N, D = X.shape
    K, D2 = Y.shape
    assert D == D2, "Dimension mismatch"
    X = X.astype(cp.float32, copy=False)
    Y = Y.astype(cp.float32, copy=False)
    out = cp.empty((N, K), dtype=cp.float32)
    grid_dim = ((N + block_dim[0] - 1) // block_dim[0],
                (K + block_dim[1] - 1) // block_dim[1])
    dot_kernel(grid_dim, block_dim, (X, Y, out, cp.int32(N), cp.int32(K), cp.int32(D)))
    return out

def distance_manhattan(X, Y):
    N, D = X.shape
    K, D2 = Y.shape
    assert D == D2, "Dimension mismatch"
    X = X.astype(cp.float32, copy=False)
    Y = Y.astype(cp.float32, copy=False)
    out = cp.empty((N, K), dtype=cp.float32)
    grid_dim = ((N + block_dim[0] - 1) // block_dim[0],
                (K + block_dim[1] - 1) // block_dim[1])
    manhattan_kernel(grid_dim, block_dim, (X, Y, out, cp.int32(N), cp.int32(K), cp.int32(D)))
    return out

def top_k_efficient(dists, k):
    """
    Retrieve the indices of the 'k' smallest distances in each column of 'dists'.
    
    Parameters:
      dists (cp.ndarray): Distances, shape (N, M), where:
          - N is the number of data points
          - M is the number of query vectors
      k (int): Number of neighbors to retrieve
    
    Returns:
      cp.ndarray: A 2D array of shape (k, M), containing
                  the indices of the k smallest distances for each column.
    """
    """
    NOTE FOR THE REPORT:
    CuPy  sorting solution uses library functions like cp.argpartition and cp.argsort that are implemented in highly optimized CUDA code. 
    These functions internally launch GPU kernels that process data in parallel across thousands of threads.

    However in our case, this approach proved to be a major bottleneck, ESPECIALLY when dealing with a lot of vectors
    so a custom quicksellect algorithm kernel was written. (Look into quicksellect more)


    
    """
    N, M = dists.shape
    k = min(N, k)
    # 1. Use cp.argpartition to find the k smallest distances per column
    k = min(dists.shape[0], k)
    # argpartition returns the indices of the k-smallest items in no guaranteed order
    #cp.argpartition uses a GPU-accelerated partition algorithm under the hood. 
    # This step is executed as a GPU kernel that sorts (or partially sorts) large chunks of data in parallel.
    topK_unsorted = cp.argpartition(dists, k - 1, axis=0)[:k, :]

    # 2. Sort these k items according to actual distance in each column
    # gather their distances
    topK_dists = dists[topK_unsorted, cp.arange(dists.shape[1])[None, :]]
    # argsort them within the k-sized slice
    sorted_order = cp.argsort(topK_dists, axis=0)
    # re-index the topK_unsorted
    topK_sorted = cp.take_along_axis(topK_unsorted, sorted_order, axis=0)

    if M == 1:
        return topK_sorted[:, 0]
    
    return topK_sorted

block_topk_kernel = cp.RawKernel(r'''
extern "C" __global__
void block_topk(const float* dists, const int* indices, float* out_vals, int* out_indices, const int N, const int K) {
    const int BLOCK_SIZE = blockDim.x;
    int start = blockIdx.x * BLOCK_SIZE;
    int tid = threadIdx.x;
    
    // Allocate shared memory: first BLOCK_SIZE floats for values, then BLOCK_SIZE ints for indices.
    extern __shared__ float sdata[];
    int* sidx = (int*)(&sdata[BLOCK_SIZE]);

    // Compute the number of valid elements in this block.
    int valid_count = (start + BLOCK_SIZE <= N) ? BLOCK_SIZE : (N - start);
    
    int idx = start + tid;
    float val = (idx < N) ? dists[idx] : 1e20f;  // Use a large number as "infinity"
    sdata[tid] = val;
    sidx[tid] = (idx < N) ? indices[idx] : -1;
    __syncthreads();

    // For K iterations, select the minimum in the block and mark it as processed.
    // Only iterate over the valid_count; if K > valid_count, fill remaining with defaults.
    for (int i = 0; i < K; i++) {
        __syncthreads();
        float block_min = 1e20f;
        int block_min_idx = -1;
        if (tid == 0) {
            if (i < valid_count) {
                // Search only among the valid_count entries.
                for (int j = 0; j < valid_count; j++) {
                    if (sdata[j] < block_min) {
                        block_min = sdata[j];
                        block_min_idx = j;
                    }
                }
                out_vals[blockIdx.x * K + i] = block_min;
                out_indices[blockIdx.x * K + i] = sidx[block_min_idx];
                sdata[block_min_idx] = 1e20f;  // Mark as processed.
            } else {
                // If no valid candidate remains, output default values.
                out_vals[blockIdx.x * K + i] = 1e20f;
                out_indices[blockIdx.x * K + i] = -1;
            }
        }
        __syncthreads();
    }
}
''', 'block_topk')


def top_k_efficient_quicksellect(dists, k, block_size=128):
    """
    Custom top-K selection using a block-wise quickselect kernel.
    Assumes dists is a 1D CuPy array (single query).
    
    Returns:
        (cp.ndarray, cp.ndarray): top-K values and their indices.
    """
    N = dists.shape[0]
    # Create an indices array [0, 1, 2, ..., N-1].
    indices = cp.arange(N, dtype=cp.int32)
    
    # Determine the number of blocks.
    num_blocks = (N + block_size - 1) // block_size
    
    # Allocate output arrays for each block's top-K candidates.
    out_vals = cp.empty(num_blocks * k, dtype=cp.float32)
    out_indices = cp.empty(num_blocks * k, dtype=cp.int32)
    
    # Calculate shared memory size: block_size floats + block_size ints.
    shared_mem = block_size * (cp.dtype(cp.float32).itemsize + cp.dtype(cp.int32).itemsize)
    
    # Launch the kernel. Grid is one-dimensional with num_blocks blocks; each block has block_size threads.
    grid = (num_blocks,)
    block = (block_size,)
    block_topk_kernel(grid, block, 
                      (dists, indices, out_vals, out_indices, np.int32(N), np.int32(k)),
                      shared_mem=shared_mem)
    
    # Now, out_vals and out_indices each contain (num_blocks * k) candidates.
    # Perform a global top-K selection on these candidates using CuPy’s built-in routines.
    # (Since num_blocks*k is small compared to N, this final step is fast.)
    global_topk_unsorted = cp.argpartition(out_vals, k - 1)[:k]
    # Sort the selected candidates.
    sorted_order = cp.argsort(out_vals[global_topk_unsorted])
    global_topk_vals = out_vals[global_topk_unsorted][sorted_order]
    global_topk_indices = out_indices[global_topk_unsorted][sorted_order]
    
    return global_topk_vals, global_topk_indices




def our_knn(N, D, A, X, K):
    """
    Computes K-Nearest Neighbors using efficient batch distance computations.
    
    Parameters:
        N (int): Number of vectors in A
        D (int): Dimensionality of vectors
        A (ndarray): Data points array of shape (N, D) 
        X (ndarray): Query vector (D)
        K (int): Number of nearest neighbors to retrieve
        
    Returns:
        np.ndarray: Indices of K nearest neighbors to X
    """

    distance_func = process_distance_func(args.dist)

    # NOTE: uncomment all of the tracking stuff when writing the "top-k retrieval" section of the report. This tracking provides great insight into bottlenecks and how we dealt wtith them
    # start = cp.cuda.Event()
    # mid = cp.cuda.Event()
    # end = cp.cuda.Event()

    A_gpu = cp.asarray(A, dtype=cp.float32)
    X_gpu = cp.asarray(X, dtype=cp.float32)

    if t.capture_knn_and_kmeans:
        t.start_capture() #Start capture after data is loaded


    if X_gpu.ndim == 1:
        X_gpu = X_gpu[None, :]  # shape -> (1, D)

    #start.record()
    
    # 1. Compute pairwise distances, shape (N, M) => N data points, M queries
    dists = distance_func(A_gpu, X_gpu)

    #mid.record()

    # 2. Retrieve the top-k neighbors using the separate function
    #topK_indices = top_k_efficient(dists, K) #NOTE: older less efficient, uncomment when writting report
    topk_vals, topk_indices = top_k_efficient_quicksellect(dists, K, block_size=128) #NOTE more efficient. verify with profiler

    #end.record()
    #end.synchronize()
    # to_mid = cp.cuda.get_elapsed_time(start, mid)
    # from_mid = cp.cuda.get_elapsed_time(mid, end)
    # print(f"Distance calculation took (ms): {to_mid}")
    # print(f"Sorting took (ms): {from_mid}")

    return cp.asnumpy(topk_indices)


def our_kmeans(N, D, A, K):

    #NOTE: typically only used via the L2 distance
    """
    Computes K-Nearest Neighbors using efficient batch distance computations.
    
    Parameters:
        N (int): Number of vectors in A
        D (int): Dimensionality of vectors
        A (ndarray): Data points array of shape (N, D) 
        K (int): Number of clusters
        
    Returns:
        np.ndarray: A label for each vector in a, showing which cluster is it assigned to
    """

    distance_func = process_distance_func(args.dist)

    max_iter = 1000
    tol = 1e-4
    # Transfer A to GPU if not already a Cupy array.
    A_gpu = cp.asarray(A, dtype=cp.float32)

    if t.capture_knn_and_kmeans:
        t.start_capture() #Start capture after data is loaded


    # Initialize centroids by randomly selecting K points from A.
    indices = cp.random.choice(N, size=K, replace=False)
    centroids = A_gpu[indices]

    labels = cp.empty(N, dtype=cp.int32)

    for iteration in range(max_iter):
        # Compute pairwise distances between each point and each centroid.
        # distances: shape (N, K)
        distances = distance_func(A_gpu, centroids)
        # Assign each point to the closest centroid.
        new_labels = cp.argmin(distances, axis=1)
        labels = new_labels

        # Update centroids:
        # Create a one-hot encoding of the labels: shape (N, K)
        one_hot = cp.zeros((N, K), dtype=cp.float32)
        one_hot[cp.arange(N), labels] = 1.0

        # Compute the new centroids as the weighted average of points in each cluster.
        new_centroids = one_hot.T @ A_gpu  # shape: (K, D)
        # Count how many points are assigned to each cluster.
        counts = one_hot.sum(axis=0)  # shape: (K,)

        # Avoid division by zero: for clusters with no points, keep the old centroid.
        nonzero_mask = counts > 0
        # Only update centroids that have at least one point.
        new_centroids[nonzero_mask] /= counts[nonzero_mask, cp.newaxis]
        new_centroids[~nonzero_mask] = centroids[~nonzero_mask]

        # Check for convergence: if centroids have not moved much, break.
        centroid_shift = cp.linalg.norm(new_centroids - centroids, axis=1).max()
        centroids = new_centroids
        if centroid_shift < tol:
            break

    # Return labels back to CPU (as NumPy array)
    return cp.asnumpy(labels)

def our_ann_lsh(N, D, A, X, K, L=6, num_hashes=2): #TODO: smarter way for hyperparams maybe...
    """
    Parameters:
       N (int): Number of data points in A.
       D (int): Dimensionality of vectors.
       A (ndarray): Data points array of shape (N, D) (NumPy array; will be transferred to GPU).
       X (ndarray): Query vector of shape (D,) or (1, D).
       K (int): How many closest points to return.
       L (int): Number of hash tables.
       num_hashes (int): Number of hashes per hash table.
    
    Returns:
       np.ndarray: Array of indices of the top-K nearest neighbors (indices correspond to the original A).
    """
    # NOTE: might want to implement this into all implementations
    # if distance_func.__name__ != 'distance_cosine':
    #     print(f"""WARN: you were trying to use a distance that was not cosine with the LSH implementation. 
    #             This LSH implementation is built to be used with cosine distance specifically.
    #             Therefore, instead of using specified {distance_func.__name__} distance, cosine distance is still being used.""")
    #     distance_func = distance_cosine

    t.capture_knn_and_kmeans = False #Turn off capturing knn and kmeans times so it does not mess with the benchmark calculations

    distance_func=process_distance_func(args.dist)
    # Transfer data to GPU if not already in CuPy arrays
    A_gpu = cp.asarray(A).astype(cp.float32)  # shape: (N, D)
    X_gpu = cp.asarray(X).astype(cp.float32)
    if X_gpu.ndim == 1:
        X_gpu = X_gpu[None, :]  # shape becomes (1, D)

    t.start_capture()
    # Precompute bit values for converting boolean hash codes into integers.
    bit_values = cp.array([1 << j for j in range(num_hashes)], dtype=cp.int32)
    
    # Generate L sets of random hyperplanes for the L hash tables.
    hyperplanes = cp.random.randn(L, num_hashes, D).astype(cp.float32)
    
    # --- Vectorized computation across all L hash tables ---
    # Compute dot products for all hash tables in one call.
    # Resulting shape: (N, L, num_hashes)
    dots_all = cp.tensordot(A_gpu, hyperplanes, axes=(1, 2))
    
    # Create binary codes: 1 if dot > 0, else 0.
    codes = (dots_all > 0).astype(cp.int32)
    
    # Convert binary codes to integer bucket IDs along the last axis.
    # Resulting shape: (N, L)
    codes_int = cp.sum(codes * bit_values, axis=2)
    
    # Compute query hash codes for all tables.
    # Resulting shape after tensordot: (1, L, num_hashes)
    t.start_query()
    query_dots_all = cp.tensordot(X_gpu, hyperplanes, axes=(1, 2))
    query_codes = cp.sum((query_dots_all > 0).astype(cp.int32) * bit_values, axis=2)  # shape: (1, L)
    query_codes = query_codes[0]  # shape: (L,)

    # Create a candidate mask: select points that match the query in at least one table.
    candidate_mask = cp.any(codes_int == query_codes, axis=1)
    candidates = cp.where(candidate_mask)[0]
    if candidates.size == 0:
        candidates = cp.arange(N)

    # Extract candidate data points.
    A_candidates = A_gpu[candidates]  # shape: (num_candidates, D)

    # Compute cosine distances between candidates and the query.
    # distance_func should accept (data_points, query) with shapes (num_candidates, D) and (1, D)
    dists = distance_func(A_candidates, X_gpu)  # Expected shape: (num_candidates, 1)
    dists = dists.flatten()  # shape: (num_candidates,)

    # Efficient top-K retrieval using cp.argpartition:
    topk_vals, topk_indices = top_k_efficient_quicksellect(dists, K, block_size=128)

    # Map the top-K candidate indices back to the original indices in A.
    final_indices = candidates[topk_indices]

    t.capture_knn_and_kmeans = True 
    
    return cp.asnumpy(final_indices)
    

def our_ann(N, D, A, X, K): #TODO: smarter way for hyperparams maybe...
    """
    Improved ANN implementation with better recall rate
     The algorithm proceeds as follows:
      1) Run k-means (our_kmeans) on A to split data into K clusters.
      2) Compute cluster centroids and use our_knn to find the K1 centroids nearest to query X.
      3) For each of these clusters, use our_knn to find K2 nearest points (candidates).
      4) Merge all candidate points and then select the top K closest points to X among them.
    
    Parameters:
        N (int): Number of data points in A.
        D (int): Dimensionality of vectors.
        A (ndarray): Data points array of shape (N, D) (NumPy array; will be transferred to GPU).
        X (ndarray): Query vector of shape (D,) or (1, D).
        K (int): Number of clusters for k-means (and also used as the number of final neighbors).
        distance_func: Distance function (e.g., distance_l2) that computes pairwise distances.
        K1 (int): Number of nearest centroids (clusters) to consider.
        K2 (int): Number of nearest points to extract from each selected cluster.
    
    Returns:
        np.ndarray: Array of indices of the top-K nearest neighbors (indices correspond to the original A).
    """
    t.capture_knn_and_kmeans = False #Turn off capturing knn and kmeans times so it does not mess with the benchmark calculations

    distance_func = process_distance_func(args.dist)

    # Ensure that A and X are on the GPU as float32.
    A_gpu = cp.asarray(A, dtype=cp.float32)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    if X_gpu.ndim == 1:
        X_gpu = X_gpu.reshape(1, D)

    t.start_capture()
    
    # Dynamic hyperparams
    K1 = math.ceil(K*0.8)
    K2 = math.ceil(K * 2)


    # 1) Run k-means to assign each point in A to one of K clusters.
    labels = our_kmeans(N, D, A_gpu, K)  # our_kmeans returns a NumPy array of labels.
    labels_gpu = cp.asarray(labels)

    # 2) Compute centroids for each cluster.
    centroids = cp.empty((K, D), dtype=cp.float32)
    for k in range(K):
        cluster_mask = (labels_gpu == k)
        if cp.any(cluster_mask):
            centroids[k] = A_gpu[cluster_mask].mean(axis=0)
        else:
            centroids[k] = 0.0  # If no points in cluster, set centroid to zeros.

    t.start_query()
    # 3) Find the K1 closest centroids to the query X.
    nearest_cluster_indices = our_knn(K, D, centroids, X_gpu, min(K1, K))
    
    candidate_indices_list = []
    candidate_vectors_list = []

    # 4) For each selected cluster, find the K2 nearest points to X.
    for i in nearest_cluster_indices:
        # Get indices of all points in cluster i.
        indices_in_cluster = cp.nonzero(labels_gpu == i)[0]
        if indices_in_cluster.size == 0:
            continue
        candidate_vectors = A_gpu[indices_in_cluster]
        # Choose up to K2 nearest points from this cluster.
        k2 = int(min(candidate_vectors.shape[0], K2))
        candidate_local_indices = our_knn(candidate_vectors.shape[0], D, candidate_vectors, X_gpu, k2)
        candidate_indices_list.append(indices_in_cluster[candidate_local_indices])
        candidate_vectors_list.append(candidate_vectors[candidate_local_indices])
    
    if len(candidate_indices_list) == 0:
        return cp.asnumpy(cp.array([], dtype=cp.int32))
    
    # Concatenate candidate indices and vectors.
    candidate_indices = cp.concatenate(candidate_indices_list, axis=0)
    candidate_vectors = cp.concatenate(candidate_vectors_list, axis=0)
    
    # 5) From all candidates (total up to K1*K2), select the top K nearest neighbors to X.
    final_k = int(min(candidate_vectors.shape[0], K))
    top_k_local = our_knn(candidate_vectors.shape[0], D, candidate_vectors, X_gpu, final_k)
    top_k_indices = candidate_indices[top_k_local]
    
    t.capture_knn_and_kmeans = True
    return cp.asnumpy(top_k_indices)

def test_kmeans():
    N, D, A, K = testdata_kmeans(args.testfile)
    kmeans_result = our_kmeans(N, D, A, K)
    print("K-Means (task 1.1) results are:")
    print(kmeans_result)
    
def test_kmeans_detailed():
    # test data
    N, D, A, K = testdata_kmeans(args.testfile)
    
    # initial setup
    print("\nK-Means Clustering Test:")
    print(f"Number of points (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Number of clusters (K): {K}")
    print(f"Data shape: {A.shape}")
    print(f"Distance metric: {args.dist}")
    
    # start time
    start_time = time.time()
    
    # run teh function
    kmeans_result = our_kmeans(N, D, A, K)
    
    # net execution time
    execution_time = time.time() - start_time
    
    print("\nResults:")
    print(f"Centroids shape: {kmeans_result.shape}")
    print(f"Execution time is: {execution_time:.4f} seconds with {N} points, {D} dimensions, and {K} clusters")
    print(kmeans_result)
 

def test_knn():
    # test data
    N, D, A, X, K = testdata_knn(args.testfile)  # Testing for 1 query points
    knn_result = our_knn(N, D, A, X, K)
    
    print("KNN (task 1) results are:")
    print(knn_result)

def test_knn_detailed():
    N, D, A, X, K = testdata_knn(args.testfile)
    
    print(f"\nTesting KNN with:")
    print(f"Database size (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Neighbors (K): {K}")
    
    # Run multiple times for timing statistics
    num_runs = 5
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        knn_result = our_knn(N, D, A, X, K)
        times.append(time.time() - start_time)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"KNN: {mean_time:.4f} ± {std_time:.4f} seconds")
    
    return knn_result
    
def test_ann():
    # Test with different configurations
    configs = ['small', 'large', 'highdim', 'moderate']
    
    for config in configs:
        print(f"\nTesting with {config} configuration:")
        N, D, A, X, K = testdata_ann(config=config)
        ann_result = our_ann(N, D, A, X, K)
        print(f"ANN results: {ann_result}")

def test_ann_detailed():
    # test data
    N, D, A, X, K = testdata_ann(args.testfile)  # Test with 1 query points
    
    print("\n=== Testing Approximate Nearest Neighbors ===")
    print(f"Database size (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Number of neighbors (K): {K}")
    print(f"Distance metric: {args.dist}")
    
    T = 5  # T is the number of trials
    times = []
    results = []
    
    print("\nRunning trials...")
    for t in range(T):
        start_time = time.time()
        ann_result = our_ann(N, D, A, X, K)
        trial_time = time.time() - start_time
        times.append(trial_time)
        results.append(ann_result)
        print(f"Trial {t+1}: {trial_time:.4f} seconds")
    
    # analyse results
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print("\n=== Results ===")
    print(f"Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Fastest trial: {min(times):.4f} seconds")
    print(f"Slowest trial: {max(times):.4f} seconds")
    
    # checking for consistency across trials
    if T > 1:
        print("\n=== Consistency Analysis ===")
        base_result = results[0]
        for t in range(1, T):
            match_rate = len(set(base_result) & set(results[t])) / K
            print(f"Trial {t+1} match rate with Trial 1: {match_rate:.4f}")
        
    return results[0]

def test_ann_lsh():
    N, D, A, X, K = testdata_ann(args.testfile)
    ann_result = our_ann_lsh(N, D, A, X, K)
    print("ANN (task 2.2) results are:")
    print(ann_result)

def test_ann_lsh_detailed(): 
    # test data
    N, D, A, X, K = testdata_ann(args.testfile)  # Test with 1 query points
    
    print("\n=== Testing LSH Approximate Nearest Neighbors ===")
    print(f"Database size (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Number of queries (M): {X.shape[0]}")
    print(f"Number of neighbors (K): {K}")
    print(f"Distance metric: {args.dist}")
    
    T = 5  # T is the number of trials
    times = []
    results = []
    
    print("\nRunning trials...")
    for t in range(T):
        start_time = time.time()
        ann_result = our_ann_lsh(N, D, A, X, K)
        trial_time = time.time() - start_time
        times.append(trial_time)
        results.append(ann_result)
        print(f"Trial {t+1}: {trial_time:.4f} seconds")
    
    # analyse results
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print("\n=== Results ===")
    print(f"Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Fastest trial: {min(times):.4f} seconds")
    print(f"Slowest trial: {max(times):.4f} seconds")
    
    # checking for consistency across trials
    if T > 1:
        print("\n=== Consistency Analysis ===")
        base_result = results[0]
        for t in range(1, T):
            match_rate = len(set(base_result) & set(results[t])) / K
            print(f"Trial {t+1} match rate with Trial 1: {match_rate:.4f}")
    
    return results[0]

def recall_rate(exact_neighbors, approx_neighbors):
    # convert to numpy arrays from cupy arrays if needed
    if isinstance(exact_neighbors, cp.ndarray):
        exact_neighbors = cp.asnumpy(exact_neighbors)
    if isinstance(approx_neighbors, cp.ndarray):
        approx_neighbors = cp.asnumpy(approx_neighbors)
    
    # convret to flat lists
    exact_list = exact_neighbors.flatten().tolist()
    approx_list = approx_neighbors.flatten().tolist()
    
    # calculate recall rate
    correct_matches = len(set(exact_list) & set(approx_list))
    return correct_matches / len(exact_list)

def recall_test(knn_function, ann_function, T=10):
    N, D, A, X, K = testdata_ann(args.testfile)
    
    print("\nRunning recall test...")
    print(f"N={N}, D={D}, K={K}")
    
    # knn
    print("Computing exact KNN...")
    knn_results = knn_function(N, D, A, X, K)
    
    # Run ANN multiple times
    print(f"Computing ANN {T} times...")
    total_recall = 0.0
    recalls = []
    
    for t in range(T):
        ann_results = ann_function(N, D, A, X, K)
        recall = recall_rate(knn_results, ann_results)
        recalls.append(recall)
        total_recall += recall
        print(f"Trial {t+1}: Recall = {recall:.4f}")
    
    avg_recall = total_recall / T
    std_recall = np.std(recalls)
    
    print(f"\nFinal Results:")
    print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Best Recall:    {max(recalls):.4f}")
    print(f"Worst Recall:   {min(recalls):.4f}")


def init_gpu():
    try:
        device = cp.cuda.runtime.getDeviceProperties(0)
        print(f"\nGPU Device: {device['name'].decode()}")
        print(f"Total Memory: {device['totalGlobalMem'] / 1e9:.2f} GB")
        
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        
        x = cp.arange(1000)
        x = cp.sum(x)
        cp.cuda.Stream.null.synchronize()
        
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        print("CUDA path variables might not have been set correctly!")
        return False
    
if __name__ == "__main__":
    for dist_fn, [N, D, A, X, K] in t:
        knn_result = t.run_cpu(fn=our_knn, fn_args=[N, D, A, X, K])
        # t.run_gpu(fn=our_kmeans,
        #           fn_args=[N, D, A, K])
        t.run_gpu(fn=our_ann,
                  fn_args=[N, D, A, X, K],
                  recall_rate_comparator=knn_result)
        t.run_gpu(fn=our_ann_lsh,
                  fn_args=[N, D, A, X, K],
                  recall_rate_comparator=knn_result)



