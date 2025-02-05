# Authors: 
#
# License: 
#
# Adapts source code from: 
# Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause


from __future__ import print_function

import locale
from warnings import warn
import time
import warnings

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree
# from sklearn.manifold import MDS, TSNE 
# from sklearn.decomposition import PCA


from umap.umap_ import raise_disconnected_warning, smooth_knn_dist, nearest_neighbors, compute_membership_strengths
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample, init_graph_transform, find_ab_params
import umap.distances as dist
import umap.sparse as sparse

from umap.utils import (
    submatrix,
    ts,
    csr_unique,
    fast_knn_indices,
)
from umap.spectral import spectral_layout


try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph
import numba



# from featuremap.core_transition_state import kernel_density_estimate

from pynndescent import NNDescent
from pynndescent.distances import named_distances as pynn_named_distances
from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

DISCONNECTION_DISTANCES = {
    "correlation": 2,
    "cosine": 2,
    "hellinger": 1,
    "jaccard": 1,
    "dice": 1,
}


# Preprocess the data by singular value decomposition
def _preprocess_data(X): 
    T1 = time.time()
    if X.shape[1] > 100 and X.shape[0] > 100:
        print("Performing SVD decomposition on the data")
        u, s, vh = scipy.sparse.linalg.svds(X, k=100, which='LM', random_state=42)
        X = np.matmul(u, np.diag(s))
    elif X.shape[1] > 100 and X.shape[0] < 100:
        print(int(X.shape[0]-1))
        u, s, vh = scipy.sparse.linalg.svds(X, k=int(X.shape[0]-1), which='LM', random_state=42)
        X = np.matmul(u, np.diag(s))
    else:    
        vh = np.eye(X.shape[1])
    T2 = time.time()
    # print(f'SVD decomposition time is {T2-T1}')
    return X, vh

from sklearn.neighbors import NearestNeighbors
def kernel_density_estimate(data, X, bw=0.5, min_radius=5, output_onlylogp=False, ):
        """
        Density estimation for data points specified by X with kernel density estimation.

        Parameters
        ----------
        data : array of shape (n_samples, n_features)
            2D array including data points. Input to density estimation.
       
        X : array
            2D array including multiple data points. Input to density estimation.
        output_onlylogp : bool
            If true, returns logp, else returns p, g, h, msu.

        Returns
        -------
        p : array
            1D array.  Unnormalized probability density. The probability density
            is not normalized due to numerical stability. Exact log probability
        """
        nbrs = NearestNeighbors(n_neighbors=min_radius + 1).fit(data)
        adaptive_bw = np.maximum(nbrs.kneighbors(data)[0][:, -1], bw)

        # the number of data points and the dimensionality
        n, d = data.shape

        from scipy.spatial.distance import cdist
        # compare euclidean distances between each pair of data and X
        D = cdist(data, X)
        

        # and evaluate the kernel at each distance
        # prevent numerical overflow due to large exponentials
        logc = -d * np.log(np.min(adaptive_bw)) - d / 2 * np.log(2 * np.pi)
        C = (adaptive_bw[:, np.newaxis] / np.min(adaptive_bw)) ** (-d) * \
            np.exp(-1 / 2. * (D / adaptive_bw[:, np.newaxis]) ** 2)

        if output_onlylogp:
            # return the kernel density estimate
            return np.log(np.mean(C, axis=0).T) + logc
        else:
            return np.mean(C, axis=0).T


@numba.njit()
def local_svd(
        data,
        knn_index,
        weight,
        n_neighbors=15,
        n_neighbors_in_guage=30,
        ):
    """
    Local singular value decomposition (SVD) for each node in the data

    Parameters
    -----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by FeatureMAP.
    knn_index: array of shape (n_samples, n_neighbors)
        The index of k-nearest neighbors for each node
    weight: array of shape (n_samples, n_samples)
        The weight matrix of the graph
    n_neighbors: int
        The number of nearest neighbors
    n_neighbors_in_guage: int
        The number of nearest neighbors for local SVD
    
    Returns
    --------
    gauge_u: list of shape (n_neighbors_in_guage, n_neighbors_in_guage); store u
    singular_values: list of shape (n_neighbors_in_guage,); store single values for each frame
    gauge_vh: list of shape (n_neighbors_in_guage, d); store v

    """
    
    gauge_u = []
    singular_values = []
    gauge_vh = []
    
    for row_i in numba.prange(data.shape[0]):
        if n_neighbors_in_guage < n_neighbors:
            indices = knn_index[row_i, 1:n_neighbors_in_guage+1].astype(np.int64)
        else:
            row_weight = weight[row_i]
            indices = np.argsort(-row_weight)[1:n_neighbors_in_guage+1].astype(np.int64)

        data_around_i = data[indices] - data[row_i]
        
        weights_around_i = weight[row_i, indices]
        weight_diag = np.diag(weights_around_i / np.sum(weights_around_i))
        weight_sqrt = np.sqrt(weight_diag)
        
        u, s, vh = np.linalg.svd(np.dot(weight_sqrt, data_around_i), full_matrices=False)
        
        gauge_u.append(u)
        singular_values.append(s)
        gauge_vh.append(vh)

    return gauge_u, singular_values, gauge_vh



import time
def graph_convolution(features, knn_index, num_iterations=20, verbose=False):   
    """
    Perform iterative neighbor averaging as a simple graph convolution.

    Args:
        features (np.ndarray): Input feature matrix of shape (num_nodes, num_features, num_channels).
        knn_index (np.ndarray): Nearest neighbor indices of shape (num_nodes, num_neighbors).
        num_iterations (int, optional): Number of smoothing iterations. Default is 20.

    Returns:
        np.ndarray: Smoothed feature matrix after `num_iterations` iterations.
        dict: Dictionary storing the first averaged result under the key "VH".
    """
    if verbose:
        print(f'Applying graph convolution for {num_iterations} iterations...')
    start_time = time.time()

    # Dictionary to store intermediate results
    featuremap_results = {}

    # Iterative neighbor averaging
    for iteration in range(num_iterations):
        smoothed_features = np.zeros_like(features)

        for i in range(features.shape[0]):
            # Compute the mean over nearest neighbors
            neighbor_features = features[knn_index[i], :, :]
            smoothed_features[i] = np.mean(neighbor_features, axis=0)

            # Normalize the mean feature vectors
            smoothed_features[i] /= np.linalg.norm(smoothed_features[i], axis=1, keepdims=True)

        # Update features for the next iteration
        features = smoothed_features

        # Store the first iteration result
        if iteration == 0:
            featuremap_results["VH"] = features.astype(np.float32, copy=True)

    end_time = time.time()
    if verbose:
        print(f'Graph convolution completed in {end_time - start_time:.2f} seconds')

    return features, featuremap_results

                 

# @numba.njit()
def tangent_space_approximation(
        data,
        graph,
        featuremap_kwds,
        ):
    """
    Compute the origial gauge in high-dimensional data by local SVD

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by FeatureMAP.
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    featuremap_kwds: dict
        Key word arguments to be used by the FeatureMAP optimization.
            
    Returns
    -------
    Local tangent space U, S, V
    """
    
    # check if the data is sparse
    if scipy.sparse.issparse(data):
        data = data.toarray()
    
    
    weight = graph.toarray()

    n_neighbors = featuremap_kwds["n_neighbors"]
    n_neighbors_in_guage = int(featuremap_kwds["gauge_coefficient"] * n_neighbors)
    knn_index = featuremap_kwds["_knn_indices"]
    
    gauge_u = [] # list of shape (n_neighbors_in_guage, n_neighbors_in_guage); store u
    singular_values = [] # list of shape (n_neighbors_in_guage,); store single values for each frame
    gauge_vh = [] # list of shape (n_neighbors_in_guage, d); store v
    
    T1 = time.time()
    gauge_u,singular_values, gauge_vh = local_svd(
            data,
            knn_index,
            weight,
            n_neighbors,
            n_neighbors_in_guage,
            )
    
    featuremap_kwds["U"] = np.array(gauge_u).astype(np.float32, copy=True) # For visualization of local 
    featuremap_kwds["Singular_value"] = np.array(singular_values).astype(np.float32, copy=True) # For visualization of local 
    featuremap_kwds["VH"] = np.array(gauge_vh).astype(np.float32, copy=True)
    T2 = time.time()
    if featuremap_kwds['verbose']:
        print(ts() + f' Local SVD time is {T2-T1}')

     # Before average 
    gauge_vh = featuremap_kwds["VH"].copy()
    ###########################################
    # # Average over NNs in original space. (Message passing accross knn network)
    #########################################
    # knn_index = featuremap_kwds["_knn_indices"]
    knn_index = knn_index
    gauge_vh = np.array(gauge_vh)

    if data.shape[0] < 5000:
        num_iterations = int(np.log2(data.shape[0])/2)
    elif data.shape[0] < 10000:
        num_iterations = int(np.log2(data.shape[0])) 
    else:
        num_iterations = int(np.log2(data.shape[0])*1.5)
    # num_iterations  = 42

    if featuremap_kwds['verbose']:
        print(ts() + f' Average over {num_iterations} times')

    gauge_vh, featuremap_kwds["VH"] = graph_convolution(gauge_vh, knn_index, num_iterations=num_iterations, verbose=featuremap_kwds['verbose'])

    featuremap_kwds["vh_smoothed"] = np.array(gauge_vh).astype(np.float32, copy=True)




def project_gauge_to_knn_graph( 
        data,
        graph,
        featuremap_kwds,
        scale=10,
        use_negative_cosines=False,):
    """
    Project the gauge (rotation matrix) to the KNN graph and compute the transition probability matrix
    Modified from the scVelo: https://scvelo.readthedocs.io/en/stable/scvelo.utils.get_transition_matrix.html#scvelo.utils.get_transition_matrix

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by FeatureMAP.
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    featuremap_kwds: dict
        Key word arguments to be used by the FeatureMAP optimization.
    scale: float
        The scale factor for the transition probability matrix
    use_negative_cosines: bool
        Whether to use negative cosine similarity in the transition probability matrix

    Returns
    -------
    transition_matrix: array of shape (n_samples, n_samples)
        The transition probability matrix
    """
    # for each edge, compute the transition probability by the (normalized) cosine similarity between edge and the gauge
    gauge_v1 = featuremap_kwds["VH"][:,0,:]
    gauge_v2 = featuremap_kwds["VH"][:,1,:]

    # gauge_v1 = featuremap_kwds["vh_smoothed"][:,0,:]
    # gauge_v2 = featuremap_kwds["vh_smoothed"][:,1,:]

    head = graph.row
    tail = graph.col

    # Probability transition matrix for gauge v1 and v2
    T_v1 = np.zeros((data.shape[0], data.shape[0]))
    T_v2 = np.zeros((data.shape[0], data.shape[0]))

    for i in range(len(head)):
        j = head[i]
        k = tail[i]
        # edge vector
        edge_vector = data[k] - data[j]
        edge_vector = edge_vector.astype(np.float32)

        # gauge vector
        gauge_vector_v1 = gauge_v1[j].astype(np.float32)
        gauge_vector_v2 = gauge_v2[j].astype(np.float32)

        # cosine sim
        T_v1[j,k] = np.dot(edge_vector, gauge_vector_v1) / (np.linalg.norm(edge_vector) * np.linalg.norm(gauge_vector_v1))
        T_v2[j,k] = np.dot(edge_vector, gauge_vector_v2) / (np.linalg.norm(edge_vector) * np.linalg.norm(gauge_vector_v2))
    
    # # cosine distance   
    # T_v1 = 1 - T_v1
    # T_v2 = 1 - T_v2
    
    # # clip the negative values in the transition probability matrix
    # T_v1 = np.clip(T_v1, 0, None)
    # T_v2 = np.clip(T_v2, 0, None)
        
    T_v1_pos = T_v1.copy()
    T_v1_neg = T_v1.copy()

    T_v2_pos = T_v2.copy()
    T_v2_neg = T_v2.copy()

    T_v1_pos = np.clip(T_v1_pos, 0, 1)
    T_v1_neg= np.clip( T_v1_neg, -1, 0)

    T_v2_pos = np.clip(T_v2_pos, 0, 1)
    T_v2_neg = np.clip(T_v2_neg, -1, 0)

    # T_v1.eliminate_zeros()
    # T_v1_neg.eliminate_zeros()

    # Apply the exponential transform to the transition probability matrix
    T_v1 = np.expm1(T_v1_pos * scale) # equivalent to np.exp(graph.A * scale) - 1
    T_v2 = np.expm1(T_v2_pos * scale)

    
    if use_negative_cosines:
        T_v1 -= np.expm1(-T_v1_neg * scale)
        T_v2 -= np.expm1(-T_v2_neg * scale)
    else:
        T_v1 += np.expm1(T_v1_neg * scale)
        T_v1 += 1

        T_v2 += np.expm1(T_v1_neg * scale)
        T_v2 += 1

    # Normalize the transition probability matrix by knn neighbors
    T_v1 = T_v1 / np.sum(T_v1, axis=1)[:, np.newaxis]
    T_v2 = T_v2 / np.sum(T_v2, axis=1)[:, np.newaxis]

    featuremap_kwds["T_v1"] = T_v1
    featuremap_kwds["T_v2"] = T_v2


   
def recover_gauge_from_embedding(
        data_embedding,
        featuremap_kwds,
        ):
    """
    Recover the gauge from the low-dimensional embedding

    Parameters
    ----------
    data_embedding: array of shape (n_samples, n_components)
        The low-dimensional embedding of the data
    featuremap_kwds: dict

    Returns
    -------
    gauge_v1_emb: array of shape (n_samples, n_components)
        The recovered gauge v1 in low dimesional space
    gauge_v2_emb: array of shape (n_samples, n_components)
        The recovered gauge v2 in low dimensional space
    """
    # Example input data
    knn_indices = featuremap_kwds["_knn_indices"]
    n_neighbors = featuremap_kwds["n_neighbors"]

    T_v1 = featuremap_kwds["T_v1"]
    T_v2 = featuremap_kwds["T_v2"]

    # Modify the transition probability matrix by subtracting 1/n
    n_samples = data_embedding.shape[0]
    T_v1 = T_v1 - 1/n_neighbors
    T_v2 = T_v2 - 1/n_neighbors

    # Compute displacement using knn neighbors
    displacement = data_embedding[knn_indices] - data_embedding[:, np.newaxis, :]  # shape (n_samples, k_neighbors, n_components)
    displacement_v1 = displacement
    # rotate the displacement by pi/2 
    displacement_v2 = np.array([displacement[:,:,1], -displacement[:,:,0]]).transpose(1,2,0)

    # Select items from T_v1 and T_v2 rows using knn_indices
    T_v1_knn = np.take_along_axis(T_v1, knn_indices, axis=1)  # shape (n_samples, k_neighbors)
    T_v2_knn = np.take_along_axis(T_v2, knn_indices, axis=1)  # shape (n_samples, k_neighbors)

    # Compute the gauge recovery
    gauge_v1_emb = np.einsum('ij,ijk->ik', T_v1_knn, displacement_v1)  # shape (n_samples, n_components)
    gauge_v2_emb = np.einsum('ij,ijk->ik', T_v2_knn, displacement_v2)  # shape (n_samples, n_components)

    featuremap_kwds["gauge_v1_emb"] = gauge_v1_emb
    featuremap_kwds["gauge_v2_emb"] = gauge_v2_emb

    # stack along the second axis
    featuremap_kwds["VH_embedding"] = np.stack((gauge_v1_emb, gauge_v2_emb), axis=1)



def tangent_space_embedding(
        featuremap_kwds,
        ):    
    """
    Embedding the gauge (rotation matrix) to low dim space

    Parameters
    ----------
    featuremap_kwds: dict
        Key word arguments to be used by the FeatureMAP optimization.  
    """
    
    ################################################################################
    # Embedding the gauge (rotation matrix) to low dim space
    # Consider the distance within topological structure of KNN network
    ###############################################################################
    random_state = featuremap_kwds["random_state"]
    n_neighbors = featuremap_kwds["n_neighbors"]
    metric = featuremap_kwds["metric"]
    min_dist = featuremap_kwds["min_dist"]
    n_epochs = featuremap_kwds["n_epochs"]

    gauge_vh = featuremap_kwds["vh_smoothed"].copy()


    # First largest PC
    # T1 = time.time()
    gauge_vh_copy = gauge_vh.copy()
    rotation_matrix = np.array(gauge_vh_copy)[:,0,:]
    
    # from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(rotation_matrix.shape[1], 50))
    X_pca = pca.fit_transform(rotation_matrix)
    
    import umap
    # print('random_state', random_state)
    umap_embedding = umap.UMAP(n_neighbors=n_neighbors,random_state=random_state, metric=metric,min_dist=min_dist, n_epochs=n_epochs).fit_transform(X=X_pca)
    # umap_embedding = umap.UMAP(n_neighbors=30, metric='euclidean',min_dist=0.5).fit_transform(X=X_pca)
    featuremap_kwds['gauge_v1_emb'] = umap_embedding
    # T2 = time.time()
    # print(f'UMAP time is {T2-T1}')
    
    umap_embedding_norm = np.linalg.norm(umap_embedding, axis=1)
        
    # gauge_vh_embedding is embedding of VH
    gauge_vh_mean_embedding = np.zeros([umap_embedding.shape[0], umap_embedding.shape[1],2])
    gauge_vh_mean_embedding[:,0,0] = umap_embedding[:, 0]/umap_embedding_norm
    gauge_vh_mean_embedding[:,0,1] = umap_embedding[:, 1]/umap_embedding_norm
    
    
    # Second largest PC
    rotation_matrix = np.array(gauge_vh_copy)[:,1,:]
    pca = PCA(n_components=min(rotation_matrix.shape[1], 50))
    X_pca = pca.fit_transform(rotation_matrix)
    
    
    umap_embedding = umap.UMAP(n_neighbors=n_neighbors,random_state=random_state, metric=metric,min_dist=min_dist, n_epochs=n_epochs).fit_transform(X=X_pca)
    # umap_embedding = umap.UMAP(n_neighbors=30,  metric='euclidean',min_dist=0.5).fit_transform(X=X_pca)
    featuremap_kwds['gauge_v2_emb'] = umap_embedding
    
    umap_embedding_norm = np.linalg.norm(umap_embedding, axis=1)
        
    # gauge_vh_embedding is embedding of VH
    gauge_vh_mean_embedding[:,1,0] = umap_embedding[:, 0]/umap_embedding_norm
    gauge_vh_mean_embedding[:,1,1] = umap_embedding[:, 1]/umap_embedding_norm

    featuremap_kwds["VH_embedding"] = np.array(gauge_vh_mean_embedding).astype(np.float32, copy=True) # VH_embedding after average
  
      
    # # First largest PC
    # # T1 = time.time()
    # gauge_vh_copy = gauge_vh.copy()
    # rotation_matrix = np.array(gauge_vh_copy)[:,0,:]
    
    # # from sklearn.decomposition import TruncatedSVD
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=min(rotation_matrix.shape[1], 50))
    # X_pca = pca.fit_transform(rotation_matrix)

    # # Spectral embedding
    # graph, _, _ = fuzzy_simplicial_set(
    #                 X_pca,
    #                 n_neighbors,
    #                 random_state,
    #                 metric,
    #                 )

    # spectral_embedding = spectral_layout(
    #         X_pca,
    #         graph,
    #         dim=2,
    #         random_state=random_state,
    #         metric=metric,
    #     )
    
    # featuremap_kwds['gauge_v1_emb'] = spectral_embedding
    # # T2 = time.time()
    # # print(f'UMAP time is {T2-T1}')
    
    # spectral_embedding_norm = np.linalg.norm(spectral_embedding, axis=1)
        
    # # gauge_vh_embedding is embedding of VH
    # gauge_vh_mean_embedding = np.zeros([spectral_embedding.shape[0], spectral_embedding.shape[1],2])
    # gauge_vh_mean_embedding[:,0,0] = spectral_embedding[:, 0]/spectral_embedding_norm
    # gauge_vh_mean_embedding[:,0,1] = spectral_embedding[:, 1]/spectral_embedding_norm
    
    
    # # Second largest PC
    # rotation_matrix = np.array(gauge_vh_copy)[:,1,:]
    # pca = PCA(n_components=min(rotation_matrix.shape[1], 50))
    # X_pca = pca.fit_transform(rotation_matrix)
    
    #  # Spectral embedding
    # graph, _, _ = fuzzy_simplicial_set(
    #                 X_pca,
    #                 n_neighbors,
    #                 random_state,
    #                 metric,
    #                 )

    # spectral_embedding = spectral_layout(
    #         X_pca,
    #         graph,
    #         dim=2,
    #         random_state=random_state,
    #         metric=metric,
    #     )
    # featuremap_kwds['gauge_v2_emb'] = spectral_embedding
    
    # spectral_embedding_norm = np.linalg.norm(spectral_embedding, axis=1)
    
    # # gauge_vh_embedding is embedding of VH
    # gauge_vh_mean_embedding[:,1,0] = spectral_embedding[:, 0]/spectral_embedding_norm
    # gauge_vh_mean_embedding[:,1,1] = spectral_embedding[:, 1]/spectral_embedding_norm

    # featuremap_kwds["VH_embedding"] = np.array(gauge_vh_mean_embedding).astype(np.float32, copy=True) # VH_embedding after average
  
   

    
def variation_embedding(
        data,
        featuremap_kwds,
        ):        
    """
    Compute the variation embedding of the data

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by FeatureMAP.
    featuremap_kwds: dict
        Key word arguments to be used by the FeatureMAP optimization.
    
    Returns
    -------
    emb_variation: array of shape (n_samples, n_components)
        The optimized of ``graph`` of variation into an ``n_components`` dimensional
        euclidean space.

    """

    n_components=featuremap_kwds["n_components"]
    random_state=featuremap_kwds["random_state"]
    n_neighbors=featuremap_kwds["n_neighbors"]
    min_dist=featuremap_kwds["min_dist"]
    spread=featuremap_kwds["spread"]
    metric=featuremap_kwds["metric"]
    threshold=featuremap_kwds["threshold"]
      
    ##########################################
    # Compute feature variation
    ############################################
    gauge_vh = featuremap_kwds["vh_smoothed"].copy()
  
    singular_values_collection = featuremap_kwds["Singular_value"].copy()
    
    # Compute intrinsic dimensionality locally
    def pc_accumulation(arr, threshold):
        arr_sum = np.sum(np.square(arr))
        temp_sum = 0
        for i in range(arr.shape[0]):
            temp_sum += arr[i] * arr[i]
            if temp_sum > arr_sum * threshold:
                return i
    
    # threshold = 0.5
    intrinsic_dim = np.zeros(data.shape[0]).astype(int)
    for i in range(data.shape[0]):            
        intrinsic_dim[i] = pc_accumulation(singular_values_collection[i], threshold)
    # Compute the gene norm in top k PCs (norm of the arrow in biplot)
    k = int(np.median(intrinsic_dim))
    if featuremap_kwds['verbose']:
        print(f'k is {k}')
    
    # gene_var_norm = np.linalg.norm(gauge_vh[:, :k, :], axis=1)
    gene_var_norm = np.sqrt(np.einsum('ijk, ijk->ik', gauge_vh[:, :k, :], gauge_vh[:, :k, :]))

    featuremap_kwds["variation_pc"] = np.array(gene_var_norm).astype(np.float32, copy=True)
    
    
    ######################################
    # Variation embedding
    ######################################## 
    import umap
    emb_variation = umap.UMAP(n_components=int(n_components), random_state=random_state, 
                              n_neighbors=n_neighbors,metric=metric,min_dist=min_dist, spread=spread).fit(gene_var_norm)
    featuremap_kwds["variation_embedding"] = emb_variation.embedding_
    featuremap_kwds["graph_v"] = emb_variation.graph_
    
    featuremap_kwds['gauge_v1_emb'] = emb_variation.embedding_[:, :2]
    # print(featuremap_kwds['gauge_v1_emb'].shape)
    featuremap_kwds['gauge_v2_emb'] = emb_variation.embedding_[:,[1,0]]
    featuremap_kwds['gauge_v2_emb'][:,0] = -featuremap_kwds['gauge_v2_emb'][:,0]

    # normalze the gauge_v1_emb and gauge_v2_emb
    gauge_v1_emb_norm = np.linalg.norm(featuremap_kwds['gauge_v1_emb'], axis=1)
    gauge_v2_emb_norm = np.linalg.norm(featuremap_kwds['gauge_v2_emb'], axis=1)

    featuremap_kwds['VH_embedding'] = np.stack((featuremap_kwds['gauge_v1_emb']/gauge_v1_emb_norm[:,np.newaxis], featuremap_kwds['gauge_v2_emb']/gauge_v2_emb_norm[:, np.newaxis]), axis=1)

    return emb_variation.embedding_


"""
Add gauge to 1-simplex;
embed it with gauge in low-dimensional space
"""
def simplicial_set_embedding_with_tangent_space_embedding(
    data,
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    featuremap_kwds,
    output_feat,
    output_metric=dist.named_distances_with_gradients["euclidean"],
    output_metric_kwds={},
    euclidean_output=True,
    parallel=False,
    verbose=False,
    tqdm_kwds=None,
    output_variation=False,
    # original_data_flag=True,
    # pca_vh=None
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by FeatureMAP.

    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.

    n_components: int
        The dimensionality of the euclidean space into which to embed the data.

    initial_alpha: float
        Initial learning rate for the SGD.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    gamma: float
        Weight to apply to negative samples.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or callable
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.

    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.

    featuremap_kwds: dict
        Key word arguments to be used by the FeatureMAP optimization.


    output_metric: function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.

    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.

    euclidean_output: bool
        Whether to use the faster code specialised for euclidean output metrics

    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    tqdm_kwds: dict
        Key word arguments to be used by the tqdm progress bar.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    graph = graph.tocoo() #TODO: tocoo or toarray?
    head = graph.row
    tail = graph.col
    # weight = graph.toarray() 
    # weight = graph.data
    # print(np.count_nonzero(weight, axis=1))

    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    # For smaller datasets we can use more epochs
    if graph.shape[0] <= 10000:
        default_epochs = 400
    else:
        default_epochs = 200

    # Use more epochs for FeatureMAP
    # if featuremap:
    default_epochs += 200
    

    if n_epochs is None:
        n_epochs = default_epochs

    if n_epochs > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()

    """
    Initialization
    """
    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral_layout(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

   
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data = {}
    
    # Compute the gauge in high-dimensional data
    if verbose:
        print(ts() + " Computing tangent space")
    T1 = time.time()
    n_neighbors=featuremap_kwds["n_neighbors"]
    n_neighbors_in_guage=int(featuremap_kwds["gauge_coefficient"] * n_neighbors)
    tangent_space_approximation(data, graph, featuremap_kwds=featuremap_kwds)
    T2 = time.time()
    if verbose:
        print(ts() + f' Tangent_space_approximation time is {T2-T1}')
    
    # if verbose:
    #     print(ts() + " Project gauge to knn_graph")
    # # Project the gauge to KNN graph to compute the transition probability matrix
    # project_gauge_to_knn_graph(data, graph, featuremap_kwds=featuremap_kwds)
   
    '''
    Variation embedding only
    '''
    
    T1 = time.time()
    embedding = variation_embedding(data=data,featuremap_kwds=featuremap_kwds)  
    T2 = time.time()

    if output_variation == True:
        if verbose:
            print(ts() + f' Variation_embedding time is {T2-T1}')

        return embedding
    else:    

        # Embedding the gauge (rotation matrix) to low dim space
        if verbose:
            print(ts() + " Tangent space embedding")
        # T1 = time.time()
        # tangent_space_embedding(featuremap_kwds)
        # T2 = time.time()
        # if verbose:
        #     print(ts() + f' Tangent_space_embedding time is {T2-T1}')


        head = graph.row
        tail = graph.col
        
        mu_sum = np.zeros(n_vertices, dtype=np.float32) # For each node, sum of edge existing probabilty of incident edges
        for i in range(len(head)):
            j = head[i]
            k = tail[i]
            mu = graph.data[i] # edge existing probability

            mu_sum[j] += mu
            mu_sum[k] += mu
        
        epsilon = 1e-8
        # print('singular_values, ' + str(singular_values))
        
        # Get the variance from singular values, corresponding to squared Euclidean distance
        singular_values = featuremap_kwds["Singular_value"].copy()
        ro_var = np.square(np.array(singular_values))
        ro = np.log(epsilon + ro_var) # radius in each directions of the hyper-ellipsoid
        # print('ro, ' + str(ro))
        
        R = (ro - np.mean(ro, axis=0)) / np.std(ro, axis=0) # normalization by column
        featuremap_kwds["R"] = R.astype(np.float32, copy=True)
        featuremap_kwds["mu"] = graph.data
        featuremap_kwds["mu_sum"] = mu_sum


        embedding = (
            10.0
            * (embedding - np.min(embedding, 0))
            / (np.max(embedding, 0) - np.min(embedding, 0))
        ).astype(np.float32, order="C")
        
        if verbose:
            print(ts() + ' Start optimizing layout')
        T1 = time.time()
        embedding = optimize_layout_euclidean_anisotropic_projection(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
            # featuremap=featuremap,
            featuremap_kwds=featuremap_kwds,
            tqdm_kwds=tqdm_kwds,
            move_other=True,
        )
        T2 = time.time()
        if verbose:
            print(ts() + f' Optimize layout time is {T2-T1}')


        
    """
    Compute the radius of two principal directions in embedding directions 
    """
    if output_feat:
        T1 = time.time()
        # if output_feat:
        if verbose:
            print(ts() + " Computing expression embedding densities")
        
        # scale the embedding coordinates
        # embedding = embedding * 100        
        # Compute graph in embedding
        (knn_indices, knn_dists, _,) = nearest_neighbors(
            embedding,
            featuremap_kwds["n_neighbors"],
            "euclidean",
            {},
            False,
            random_state,
            verbose=False,
        )
    
        emb_graph, _, _, emb_dists = fuzzy_simplicial_set(
            embedding,
            featuremap_kwds["n_neighbors"],
            random_state,
            "euclidean",
            {},
            knn_indices,
            knn_dists,
            verbose=False,
            return_dists=True,
        )
    
        
        emb_graph = emb_graph.tocoo()
        emb_graph.sum_duplicates()
        emb_graph.eliminate_zeros()
    
        n_vertices = emb_graph.shape[1]
        dim = embedding.shape[1]
    
        phi_sum = np.zeros(n_vertices, dtype=np.float32)
        re_sum = np.zeros([n_vertices, dim], dtype=np.float32)
        re_sum_without_log = np.zeros([n_vertices, dim], dtype=np.float32)
    
        
        VH_embedding = featuremap_kwds["VH_embedding"]
    
        head = emb_graph.row
        tail = emb_graph.col
        
        for i in range(len(head)):
            j = head[i]
            k = tail[i]
            
            current = embedding[j]
            other = embedding[k]
            
            current_VH = VH_embedding[j] # array shape of (dim, dim)
            other_VH = VH_embedding[k]
            
            vec_diff = other - current
    
            # D = emb_dists[j, k]
            phi = emb_graph.data[i]
            phi_sum[j] += phi
            phi_sum[k] += phi
            
            for d in range(dim):
                vec_proj_vh_j = np.dot(vec_diff, current_VH[d]) # project to d-th rotation direction
                vec_proj_vh_k = np.dot(vec_diff, other_VH[d])        
         
                re_sum[j,d] += phi * vec_proj_vh_j * vec_proj_vh_j
                re_sum[k,d] += phi * vec_proj_vh_k * vec_proj_vh_k
                
        for i in range(re_sum.shape[0]):
            re_sum_without_log[i] = epsilon + (re_sum[i] / phi_sum[i])
    
        for i in range(re_sum.shape[0]):
            re_sum[i] = np.log(epsilon + (re_sum[i] / phi_sum[i]))
            
        featuremap_kwds["re_sum"] = np.sqrt(re_sum) # Variance of Gaussian distribution
        featuremap_kwds["re_sum_without_log"] = np.sqrt(re_sum_without_log) # Variance of Gaussian distribution
        featuremap_kwds["emb_graph"] = emb_graph
        featuremap_kwds["emb_dists"] = emb_dists
        featuremap_kwds["emb_knn_indices"] = knn_indices
         
        T2 = time.time()
        if verbose:
            print(ts() + f' Embedding radii computation time is {T2-T1}')
        
    return embedding
        





class FeatureMAP(BaseEstimator):
    """Feature-preserving Manifold Approximation and Projection

    Approximate the high dimensional manifold of the input data by low dimensional embedding,
    with feature-preserving property. The algorithm is based on the paper:
    Feature-preserving manifold approximation and projection to visulize single-cell data, Yang et al. 2023.

    The algorithm is based on the UMAP algorithm, and the main difference is that FeatureMAP 
    uses the tangent space of the input data to compute the (embedding) gauge for low dimensional embedding,
    where the features are locally illustrated by the embedding tangent space (i.e., gauge) and data points are
    embedded to the low dimensional space with the gauge.


    Parameters
    ----------
    n_neighbors: float (optional, default 30)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

    metric: string  (currently, only 'euclidean' is supported)
        The metric to use to compute distances in high dimensional space.

    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.

    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    min_dist: float (optional, default 0.5)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.

    low_memory: bool (optional, default True)
        For some datasets the nearest neighbor computation can consume a lot of
        memory. If you find that FeatureMAP is failing due to memory constraints
        consider setting this option to True. This approach is more
        computationally expensive, but avoids excessive memory use.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    metric_kwds: dict (optional, default None)
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. If None then no arguments are passed on.

    angular_rp_forest: bool (optional, default False)
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.

    transform_seed: int (optional, default 42)
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.

    verbose: bool (optional, default False)
        Controls verbosity of logging.

    tqdm_kwds: dict (optional, defaul None)
        Key word arguments to be used by the tqdm progress bar.

    unique: bool (optional, default False)
        Controls if the rows of your data should be uniqued before being
        embedded.  If you have more duplicates than you have n_neighbour
        you can have the identical data points lying in different regions of
        your space.  It also violates the definition of a metric.
        For to map from internal structures back to your data use the variable
        _unique_inverse_.

    featuremap: bool (optional, default False)
        Specifies whether the featity-augmented objective of FeatureMAP
        should be used for optimization. Turning on this option generates
        an embedding where the local featities are encouraged to be correlated
        with those in the original space. Parameters below with the prefix 'feat'
        further control the behavior of this extension.

    feat_lambda: float (optional, default 2.0)
        Controls the regularization weight of the anisotropic density correlation term
        in FeatureMAP. Higher values prioritize density preservation over the
        FeatureMAP objective, and vice versa for values closer to zero. Setting this
        parameter to zero is equivalent to running the original UMAP algorithm.

    feat_frac: float (optional, default 0.3)
        Controls the fraction of epochs (between 0 and 1) where the
        feature-augmented objective is used in FeatureMAP. The first
        (1 - feat_frac) fraction of epochs optimize the original UMAP objective
        before introducing the anisotropic density correlation term.

    feat_gauge_coefficient: float (default 2.0)
        A coefficient multiplication of n_neighbors as number of neighbors in 
        gauge construction. Larger coefficient means long distance information
    
    feat_var_shift: float (optional, default 0.1)
        A small constant added to the variance of local radii in the
        embedding when calculating the anisotropic density correlation objective to
        prevent numerical instability from dividing by a small number
    
    output_feat: bool (optional, default False)
        Whether to compute the embedding radius. 
        If set to True, the algorithm will compute the embedding radius
        and return the embedding radius.

    disconnection_distance: float (optional, default np.inf or maximal value for bounded distances)
        Disconnect any vertices of distance greater than or equal to disconnection_distance when approximating the
        manifold via our k-nn graph. This is particularly useful in the case that you have a bounded metric.  The
        UMAP assumption that we have a connected manifold can be problematic when you have points that are maximally
        different from all the rest of your data.  The connected manifold assumption will make such points have perfect
        similarity to a random set of other points.  Too many such points will artificially connect your space.
    
    output_variation: bool (optional, default True)
        Whether to compute the variation embedding only. If set to True, the algorithm will compute the variation embedding
        only and return the embedding. If set to False, the algorithm will compute the Feature-augmented embedding.
    
    threshold: float (optional, default 0.9)
        The threshold to compute the intrinsic dimensionality of the data. The intrinsic dimensionality is computed
        by the number of principal components that accumulates 90% of the variance of the data.
       
    """

    def __init__(
        self,
        n_neighbors=30,
        n_components=2,
        metric="euclidean",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.5,
        spread=1.0,
        low_memory=True,
        n_jobs=-1,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        transform_seed=42,
        transform_mode="embedding",
        force_approximation_algorithm=False,
        verbose=False,
        tqdm_kwds=None,
        unique=False,
        # featuremap=False,
        feat_lambda=0.5,
        feat_frac=0.3,
        feat_gauge_coefficient = 2.0,
        feat_var_shift=0.1,
        output_feat=False,
        disconnection_distance=None,
        output_variation=False,
        threshold=0.9
        # original_data_flag=True,
        # pca_vh=None
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.output_metric = output_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_seed = transform_seed
        self.transform_mode = transform_mode
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.tqdm_kwds = tqdm_kwds
        self.unique = unique

        # self.featuremap = featuremap
        self.feat_lambda = feat_lambda
        self.feat_frac = feat_frac
        self.feat_gauge_coefficient = feat_gauge_coefficient
        self.feat_var_shift = feat_var_shift
        self.output_feat = output_feat
        self.disconnection_distance = disconnection_distance
        
        self.output_variation = output_variation
        self.threshold = threshold
        # self.original_data_flag = original_data_flag,
        # self.pca_vh = pca_vh
        
        
        self.n_jobs = n_jobs

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in (
            "spectral",
            "random",
        ):
            raise ValueError('string init values must be "spectral" or "random"')
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        # if self.n_components larger than the number of features, it will raise an error
        if self.n_components >= self._raw_data.shape[1]:
            raise ValueError(
                "n_components must be less than or equal to the number of features"
            )
        if self.n_epochs is not None and (
            self.n_epochs < 0 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a nonnegative integer")
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds
        # save repeated checks later on
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False
        # set input distance metric & inverse_transform distance metric
        if callable(self.metric):
            in_returns_grad = self._check_custom_metric(
                self.metric, self._metric_kwds, self._raw_data
            )
            if in_returns_grad:
                _m = self.metric

                @numba.njit(fastmath=True)
                def _dist_only(x, y, *kwds):
                    return _m(x, y, *kwds)[0]

                self._input_distance_func = _dist_only
                self._inverse_distance_func = self.metric
            else:
                self._input_distance_func = self.metric
                self._inverse_distance_func = None
                warn(
                    "custom distance metric does not return gradient; inverse_transform will be unavailable. "
                    "To enable using inverse_transform method, define a distance function that returns a tuple "
                    "of (distance [float], gradient [np.array])"
                )
        elif self.metric == "precomputed":
            if self.unique:
                raise ValueError("unique is poorly defined on a precomputed metric")
            warn("using precomputed metric; inverse_transform will be unavailable")
            self._input_distance_func = self.metric
            self._inverse_distance_func = None
        elif self.metric == "hellinger" and self._raw_data.min() < 0:
            raise ValueError("Metric 'hellinger' does not support negative values")
        elif self.metric in dist.named_distances:
            if self._sparse_data:
                if self.metric in sparse.sparse_named_distances:
                    self._input_distance_func = sparse.sparse_named_distances[
                        self.metric
                    ]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = dist.named_distances[self.metric]
            try:
                self._inverse_distance_func = dist.named_distances_with_gradients[
                    self.metric
                ]
            except KeyError:
                warn(
                    "gradient function is not yet implemented for {} distance metric; "
                    "inverse_transform will be unavailable".format(self.metric)
                )
                self._inverse_distance_func = None
        elif self.metric in pynn_named_distances:
            if self._sparse_data:
                if self.metric in pynn_sparse_named_distances:
                    self._input_distance_func = pynn_sparse_named_distances[self.metric]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = pynn_named_distances[self.metric]

            warn(
                "gradient function is not yet implemented for {} distance metric; "
                "inverse_transform will be unavailable".format(self.metric)
            )
            self._inverse_distance_func = None
        else:
            raise ValueError("metric is neither callable nor a recognised string")
        # set output distance metric
        if callable(self.output_metric):
            out_returns_grad = self._check_custom_metric(
                self.output_metric, self._output_metric_kwds
            )
            if out_returns_grad:
                self._output_distance_func = self.output_metric
            else:
                raise ValueError(
                    "custom output_metric must return a tuple of (distance [float], gradient [np.array])"
                )
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in dist.named_distances_with_gradients:
            self._output_distance_func = dist.named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric in dist.named_distances:
            raise ValueError(
                "gradient function is not yet implemented for {}.".format(
                    self.output_metric
                )
            )
        else:
            raise ValueError(
                "output_metric is neither callable nor a recognised string"
            )
        # set angularity for NN search based on metric
        if self.metric in (
            "cosine",
            "correlation",
            "dice",
            "jaccard",
            "ll_dirichlet",
            "hellinger",
        ):
            self.angular_rp_forest = True

        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError("n_jobs must be a postive integer, or -1 (for all cores)")

        if self.feat_lambda < 0.0:
            raise ValueError("feat_lambda cannot be negative")
        if self.feat_frac < 0.0 or self.feat_frac > 1.0:
            raise ValueError("feat_frac must be between 0.0 and 1.0")
        if self.feat_gauge_coefficient < 0.0:
            raise ValueError("feat_gauge_coefficient cannot be negative")
        if self.feat_var_shift < 0.0:
            raise ValueError("feat_var_shift cannot be negative")

        self._featuremap_kwds = {
            "lambda": self.feat_lambda, #if self.featuremap else 0.0,
            "frac": self.feat_frac, #if self.featuremap else 0.0,
            "gauge_coefficient": self.feat_gauge_coefficient, #if self.featuremap else 0.0,
            "var_shift": self.feat_var_shift,
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state,
            "metric": self.metric,
            "min_dist": self.min_dist,
            "spread": self.spread,
            "n_components": self.n_components,
            "verbose": self.verbose,
            "n_epochs": self.n_epochs,
            "threshold": self.threshold,   
            
        }
        # if self.output_variation:
        #     self._featuremap_kwds['threshold'] = self.threshold

        # if self.featuremap:
        if self.output_metric not in ("euclidean", "l2"):
            raise ValueError(
                "Non-Euclidean output metric not supported for FeatureMAP."
            )

        # This will be used to prune all edges of greater than a fixed value from our knn graph.
        # We have preset defaults described in DISCONNECTION_DISTANCES for our bounded measures.
        # Otherwise a user can pass in their own value.
        if self.disconnection_distance is None:
            self._disconnection_distance = DISCONNECTION_DISTANCES.get(
                self.metric, np.inf
            )
        elif isinstance(self.disconnection_distance, int) or isinstance(
            self.disconnection_distance, float
        ):
            self._disconnection_distance = self.disconnection_distance
        else:
            raise ValueError("disconnection_distance must either be None or a numeric.")

        if self.tqdm_kwds is None:
            self.tqdm_kwds = {}
        else:
            if isinstance(self.tqdm_kwds, dict) is False:
                raise ValueError(
                    "tqdm_kwds must be a dictionary. Please provide valid tqdm "
                    "parameters as key value pairs. Valid tqdm parameters can be "
                    "found here: https://github.com/tqdm/tqdm#parameters"
                )
        if "desc" not in self.tqdm_kwds:
            self.tqdm_kwds["desc"] = "Epochs completed"
        if "bar_format" not in self.tqdm_kwds:
            bar_f = "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]"
            self.tqdm_kwds["bar_format"] = bar_f

    def _check_custom_metric(self, metric, kwds, data=None):
        # quickly check to determine whether user-defined
        # self.metric/self.output_metric returns both distance and gradient
        if data is not None:
            # if checking the high-dimensional distance metric, test directly on
            # input data so we don't risk violating any assumptions potentially
            # hard-coded in the metric (e.g., bounded; non-negative)
            x, y = data[np.random.randint(0, data.shape[0], 2)]
        else:
            # if checking the manifold distance metric, simulate some data on a
            # reasonable interval with output dimensionality
            x, y = np.random.uniform(low=-10, high=10, size=(2, self.n_components))

        if scipy.sparse.issparse(data):
            metric_out = metric(x.indices, x.data, y.indices, y.data, **kwds)
        else:
            metric_out = metric(x, y, **kwds)
        # True if metric returns iterable of length 2, False otherwise
        return hasattr(metric_out, "__iter__") and len(metric_out) == 2
    


    def fit(self, X):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """
  
        # # SVD decompostion of data 
        # if X.shape[1] > 100:
        #     if self.verbose:
        #         print("Performing SVD decomposition on the data")
        #     u, s, vh = scipy.sparse.linalg.svds(X, k=100, which='LM', random_state=42)
        #     X = np.matmul(u, np.diag(s))
        # else:
        #     vh = np.eye(X.shape[1])

        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        # SVD decomposition of data
        # self._featuremap_kwds['svd_vh'] = vh.T

        if self.verbose:
            print(str(self))

        self._original_n_threads = numba.get_num_threads()
        if self.n_jobs > 0 and self.n_jobs is not None:
            numba.set_num_threads(self.n_jobs)

        # Check if we should unique the data
        # We've already ensured that we aren't in the precomputed case
        if self.unique:
            # check if the matrix is feate
            if self._sparse_data:
                # Call a sparse unique function
                index, inverse, counts = csr_unique(X)
            else:
                index, inverse, counts = np.unique(
                    X,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                    axis=0,
                )[1:4]
            if self.verbose:
                print(
                    "Unique=True -> Number of data points reduced from ",
                    X.shape[0],
                    " to ",
                    X[index].shape[0],
                )
                most_common = np.argmax(counts)
                print(
                    "Most common duplicate is",
                    index[most_common],
                    " with a count of ",
                    counts[most_common],
                )
            # We'll expose an inverse map when unique=True for users to map from our internal structures to their data
            self._unique_inverse_ = inverse
        # If we aren't asking for unique use the full index.
        # This will save special cases later.
        else:
            index = list(range(X.shape[0]))
            inverse = list(range(X.shape[0]))
        # print('inverse, ' +  str(inverse))
        # Error check n_neighbors based on data size
        if X[index].shape[0] <= self.n_neighbors:
            if X[index].shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X[index].shape[0] - 1
            # if self.featuremap:
            self._featuremap_kwds["n_neighbors"] = self._n_neighbors
        else:
            self._n_neighbors = self.n_neighbors

        # Note: unless it causes issues for setting 'index', could move this to
        # initial sparsity check above
        if self._sparse_data and not X.has_sorted_indices:
            X.sort_indices()

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print(ts(), "Construct fuzzy simplicial set")

        """
        Construct the fuzzy_simplicial_set
        """    
        if self.metric == "precomputed" and self._sparse_data:
            # For sparse precomputed distance matrices, we just argsort the rows to find
            # nearest neighbors. To make this easier, we expect matrices that are
            # symmetrical (so we can find neighbors by looking at rows in isolation,
            # rather than also having to consider that sample's column too).
            # print("Computing KNNs for sparse precomputed distances...")
            if sparse_tril(X).getnnz() != sparse_triu(X).getnnz():
                raise ValueError(
                    "Sparse precomputed distance matrices should be symmetrical!"
                )
            if not np.all(X.diagonal() == 0):
                raise ValueError("Non-zero distances from samples to themselves!")
            self._knn_indices = np.zeros((X.shape[0], self.n_neighbors), dtype=np.int)
            self._knn_dists = np.zeros(self._knn_indices.shape, dtype=np.float)
            for row_id in range(X.shape[0]):
                # Find KNNs row-by-row
                row_data = X[row_id].data
                row_indices = X[row_id].indices
                if len(row_data) < self._n_neighbors:
                    raise ValueError(
                        "Some rows contain fewer than n_neighbors distances!"
                    )
                row_nn_data_indices = np.argsort(row_data)[: self._n_neighbors]
                self._knn_indices[row_id] = row_indices[row_nn_data_indices]
                self._knn_dists[row_id] = row_data[row_nn_data_indices]

            # Disconnect any vertices farther apart than _disconnection_distance
            disconnected_index = self._knn_dists >= self._disconnection_distance
            self._knn_indices[disconnected_index] = -1
            self._knn_dists[disconnected_index] = np.inf
            edges_removed = disconnected_index.sum()

            (
                self.graph_,
                self._sigmas,
                self._rhos,
                # self.graph_dists_,
            ) = fuzzy_simplicial_set(
                X[index],
                self.n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
                # self.output_feat,

            )
            # Report the number of vertices with degree 0 in our our self.graph_
            # This ensures that they were properly disconnected.
            vertices_disconnected = np.sum(
                np.array(self.graph_.sum(axis=1)).flatten() == 0
            )
            raise_disconnected_warning(
                edges_removed,
                vertices_disconnected,
                self._disconnection_distance,
                self._raw_data.shape[0],
                verbose=self.verbose,
            )
        # Handle small cases efficiently by computing all distances
        # TODO: get _knn_indices when less than 4096 
        elif X[index].shape[0] < 1000 and not self.force_approximation_algorithm:
            self._small_data = True
            try:
                # sklearn pairwise_distances fails for callable metric on sparse data
                _m = self.metric if self._sparse_data else self._input_distance_func
                dmat = pairwise_distances(X[index], metric=_m, **self._metric_kwds)
            except (ValueError, TypeError) as e:
                # metric is numba.jit'd or not supported by sklearn,
                # fallback to pairwise special

                if self._sparse_data:
                    # Get a fresh metric since we are casting to feate
                    if not callable(self.metric):
                        _m = dist.named_distances[self.metric]
                        dmat = dist.pairwise_special_metric(
                            X[index].toarray(),
                            metric=_m,
                            kwds=self._metric_kwds,
                        )
                    else:
                        dmat = dist.pairwise_special_metric(
                            X[index],
                            metric=self._input_distance_func,
                            kwds=self._metric_kwds,
                        )
                else:
                    dmat = dist.pairwise_special_metric(
                        X[index],
                        metric=self._input_distance_func,
                        kwds=self._metric_kwds,
                    )
            # set any values greater than disconnection_distance to be np.inf.
            # This will have no effect when _disconnection_distance is not set since it defaults to np.inf.
            edges_removed = np.sum(dmat >= self._disconnection_distance)
            dmat[dmat >= self._disconnection_distance] = np.inf
            # print(dmat.shape)
            # print(dmat[:2, :])
            self._knn_dists = np.sort(dmat, axis=1)
            self._knn_indices = np.argsort(dmat, axis=1)
            # print( self._knn_dists[:2, :])
            # print( self._knn_indices[:2, :])
            
            (
                self.graph_,
                self._sigmas,
                self._rhos,
                # self.graph_dists_,
            ) = fuzzy_simplicial_set(
                dmat,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
                # self.output_feat,

            )
            # Report the number of vertices with degree 0 in our our self.graph_
            # This ensures that they were properly disconnected.
            vertices_disconnected = np.sum(
                np.array(self.graph_.sum(axis=1)).flatten() == 0
            )
            raise_disconnected_warning(
                edges_removed,
                vertices_disconnected,
                self._disconnection_distance,
                self._raw_data.shape[0],
                verbose=self.verbose,
            )
        else:
            # Standard case
            self._small_data = False
            # Standard case
            if self._sparse_data and self.metric in pynn_sparse_named_distances:
                nn_metric = self.metric
            elif not self._sparse_data and self.metric in pynn_named_distances:
                nn_metric = self.metric
            else:
                nn_metric = self._input_distance_func

            (
                self._knn_indices,
                self._knn_dists,
                self._knn_search_index,
            ) = nearest_neighbors(
                X[index],
                self._n_neighbors,
                nn_metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.low_memory,
                use_pynndescent=True,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )

            # Disconnect any vertices farther apart than _disconnection_distance
            disconnected_index = self._knn_dists >= self._disconnection_distance
            self._knn_indices[disconnected_index] = -1
            self._knn_dists[disconnected_index] = np.inf
            edges_removed = disconnected_index.sum()

            (
                self.graph_,
                self._sigmas,
                self._rhos,
                # self.graph_dists_,
            ) = fuzzy_simplicial_set(
                X[index],
                self.n_neighbors,
                random_state,
                nn_metric,
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
                # self.output_feat,

            )
            # Report the number of vertices with degree 0 in our our self.graph_
            # This ensures that they were properly disconnected.
            vertices_disconnected = np.sum(
                np.array(self.graph_.sum(axis=1)).flatten() == 0
            )
            raise_disconnected_warning(
                edges_removed,
                vertices_disconnected,
                self._disconnection_distance,
                self._raw_data.shape[0],
                verbose=self.verbose,
            )

        # if self.featuremap or self.output_feat:
        # self._featuremap_kwds["graph_dists"] = self.graph_dists_
        self._featuremap_kwds["n_neighbors"] = self.n_neighbors
        # self._featuremap_kwds["n_neighbors_in_guage"] = self.n_neighbors # TODO: n_neighbors_in_guage same as n_neighbors or not
        self._featuremap_kwds["_knn_indices"] = self._knn_indices

        if self.verbose:
            print(ts(), "Construct embedding")

        if self.transform_mode == "embedding":
            self.embedding_ = self._fit_embed_data(
                self._raw_data[index],
                self.n_epochs,
                # self.featuremap,
                init,
                random_state,  
            )
            # print('aux_data, ' + str(aux_data))
            # Assign any points that are fully disconnected from our manifold(s) to have embedding
            # coordinates of np.nan.  These will be filtered by our plotting functions automatically.
            # They also prevent users from being deceived a distance query to one of these points.
            # Might be worth moving this into simplicial_set_embedding or _fit_embed_data
            disconnected_vertices = np.array(self.graph_.sum(axis=1)).flatten() == 0
            if len(disconnected_vertices) > 0:
                self.embedding_[disconnected_vertices] = np.full(
                    self.n_components, np.nan
                )

            self.embedding_ = self.embedding_[inverse]
            # if self.output_feat:
            #     self.rad_orig_ = aux_data["rad_orig"][inverse]
            #     self.rad_emb_ = aux_data["rad_emb"][inverse]

        self._featuremap_kwds["X_embedding"] = self.embedding_
        if self.verbose:
            print(ts() + " Finished embedding")

        numba.set_num_threads(self._original_n_threads)
        self._input_hash = joblib.hash(self._raw_data)

        return self

    def _fit_embed_data(self, X, n_epochs, init, random_state):
        """A method wrapper for simplicial_set_embedding that can be
        replaced by subclasses.
        """
        # if featuremap:
        return simplicial_set_embedding_with_tangent_space_embedding(
            X,
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            # self.featuremap,
            self._featuremap_kwds,
            self.output_feat,
            self._output_distance_func,
            self._output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None,
            self.verbose,
            tqdm_kwds=self.tqdm_kwds,
            output_variation=self.output_variation,
        )
        
            

    def fit_transform(self, X):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.

        """
        self.fit(X)
        if self.transform_mode == "embedding":
            if self.output_variation:
                return self.embedding_
            # elif self.output_feat:
            #     return self.embedding_, self.rad_orig_,  self.rad_emb_
            else:
                return self.embedding_
        elif self.transform_mode == "graph":
            return self.graph_
        else:
            raise ValueError(
                "Unrecognized transform mode {}; should be one of 'embedding' or 'graph'".format(
                    self.transform_mode
                )
            )

    


    def __repr__(self):
        from sklearn.utils._pprint import _EstimatorPrettyPrinter
        import re
        pp = _EstimatorPrettyPrinter(
            compact=True,
            indent=1,
            indent_at_name=True,
            n_max_elements_to_show=50,
        )
        pp._changed_only = True
        repr_ = pp.pformat(self)
        repr_ = re.sub("tqdm_kwds={.*},", "", repr_, flags=re.S)
        # remove empty lines
        repr_ = re.sub("\n *\n", "\n", repr_, flags=re.S)
        # remove extra whitespaces after a comma
        repr_ = re.sub(", +", ", ", repr_)
        return repr_



import numpy as np
import numba
import umap.distances as dist
from umap.utils import tau_rand_int
from tqdm.auto import tqdm

@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val




@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
    },
)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

# def vdiff(x, y):
#     """    
#     Vector difference.
    
#     Parameters
#     ----------
#     x : array of shape (embedding_dim,)
#     y : array of shape (embedding_dim,)
    
#     Returns
#     -------
#     The vector difference between x and y 

#     """
#     result = y - x
#     return result



def _optimize_layout_euclidean_single_epoch_grad(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    featuremap_flag,
    feat_phi_sum,
    feat_re_sum,
    feat_re_cov,
    feat_re_std,
    feat_re_mean,
    feat_lambda,
    feat_R,
    feat_VH_embedding,
    feat_mu,
    feat_mu_tot,
    
):  
    # print('epochs_per_sample.shape[0]' + str(epochs_per_sample.shape[0]))
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]
            
            # vec_diff = vdiff(current, other)
            vec_diff = other - current

            inner_product = np.dot(vec_diff, vec_diff)
            outer_product = np.outer(vec_diff, vec_diff)
            
            grad_d = np.zeros(dim, dtype=np.float32)
            
            #dim = 1
            # random select a dimension from dim
            # d = tau_rand_int(rng_state) % dim
            # print('featuremap_flag' + str(featuremap_flag))
            if featuremap_flag:
                current_VH = feat_VH_embedding[j] # rotation matrix embedding;  
                other_VH = feat_VH_embedding[k]
                
                grad_cor_coeff = np.zeros(dim, dtype=np.float32)
                # TODO: focus on each d of dim
                for d in numba.prange(dim):
                    phi = 1.0 / (1.0 + a * pow(inner_product, b))
                    dphi_term = (
                        2 * a * b * pow(inner_product, b - 1) * vec_diff / 
                        (1.0 + a * pow(inner_product, b))
                    )

                    v_j = current_VH[d]
                    v_k = other_VH[d]
                    project_vec_j = np.dot(v_j, vec_diff)
                    project_vec_k = np.dot(v_k, vec_diff)

                    #TODO: check feat_phi_sum, feat_re_sum
                    # Have changed the order of j, k
                    q_jk = phi / feat_phi_sum[j]
                    q_kj = phi / feat_phi_sum[k]
                    
                   
                    # drj = q_jk * (
                    #     project_vec_j * (2 * v_j -  project_vec_j * dphi_term ) / np.exp(feat_re_sum[j,d]) + dphi_term
                    # )
                    # drk = q_kj * (
                    #     project_vec_k * (2 * v_k -  project_vec_k * dphi_term ) / np.exp(feat_re_sum[k,d]) + dphi_term
                    # )
                    drj = np.zeros(dim)
                    drk = np.zeros(dim)
                    for s in numba.prange(dim):
                        drj[s] = q_jk * (
                            project_vec_j * (2 * v_j[s] -  project_vec_j * dphi_term[s] ) / np.exp(feat_re_sum[j,d]) + dphi_term[s]
                        )
                        drk[s] = q_kj * (
                            project_vec_k * (2 * v_k[s] -  project_vec_k * dphi_term[s] ) / np.exp(feat_re_sum[k,d]) + dphi_term[s]
                        )
                    
                    # check feat_re_std: array shape (dim,)
                    re_std_sq = feat_re_std[d] * feat_re_std[d]
                    
         
                    weight_j = (
                        feat_R[j,d]
                        - feat_re_cov[d] * (feat_re_sum[j,d] - feat_re_mean[d]) / re_std_sq
                    )
                    weight_k = (
                        feat_R[k,d]
                        - feat_re_cov[d] * (feat_re_sum[k,d] - feat_re_mean[d]) / re_std_sq
                    )
                    for s in numba.prange(dim):
                        grad_cor_coeff[s] += (weight_j * drj[s] + weight_k * drk[s]) / feat_re_std[d]
    
                for s in numba.prange(dim):
                    grad_cor_coeff[s] = (
                        grad_cor_coeff[s]
                        * feat_lambda
                        * feat_mu_tot
                        / feat_mu[i]
                        / n_vertices
                    )

            # grad_coeff = np.zeros(dim, dtype=np.float32)
            if inner_product > 0.0:
                # gradient of log Q_jk 
                grad_coeff_term = (-2.0) * a * b * pow(inner_product, b - 1.0) 
                grad_coeff_term = grad_coeff_term / (a * pow(inner_product, b) + 1.0)
            else:
                grad_coeff_term = 0
                  
            # gradient w.r.t y_j; sampling edge (j,k), Z_jk = y_k - y_j 
            # grad_d = clip_arr(* vec_diff[d]) * vec_diff[d] * (-1.0)
            for d in numba.prange(dim):
                grad_d = clip(grad_coeff_term * vec_diff[d] * (-1.0))
                # grad_d = grad_coeff[d] * (-1.0) * vec_diff[d]

                if featuremap_flag:
                    # FIXME: grad_cor_coeff might be referenced before assignment
                    grad_d += clip(grad_cor_coeff[d] * (-1.0))
                
                # 
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in numba.prange(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                # vec_diff = vdiff(current, other)
                vec_diff = other - current
                inner_product = np.dot(vec_diff, vec_diff)
                
                if inner_product > 0.0:
                    grad_coeff_term = 2.0 * gamma * b
                    #divisor = np.repeat((0.001 + inner_product) * (a * pow(inner_product, b) + 1), dim)
                    grad_coeff_term = grad_coeff_term / ((0.001 + inner_product) * (a * pow(inner_product, b) + 1))
                elif j == k:
                    continue
                else:
                    grad_coeff_term = 0.0
                

                for d in numba.prange(dim):
                    if grad_coeff_term > 0.0:
                        grad_d = clip(grad_coeff_term * vec_diff[d] * (-1.0))
                    else:
                        grad_d = 2.0 
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


# Compute the variance in each direction of embedding under rotation VH 
def _optimize_layout_euclidean_featuremap_epoch_init_grad(
    head_embedding,
    tail_embedding,
    head,
    tail,
    # random_state,
    # gauge_vh,
    VH_embedding,
    # rotation_angle,
    a,
    b,
    re_sum,
    phi_sum,
):
    """
    Compute the principal radius in the dim-dimensional embedding space.
    
    Parameter
    ---------
    re_sum: array of shape (n_vertices, dim) 
        The principal radius in the embedding space
    phi_sum: array of shape (n_vertices,)
        For node i, the sum of edge existing probability incident to this node
    """
    
    # VH_embedding.fill(0)
    # random_initial = random_state.randint(0, head_embedding.shape[0], 1).astype(np.int64)
    # vh_initial = gauge_vh[random_initial[0]][0]
    # for i in range(head_embedding.shape[0]):
    #     vh_vector = gauge_vh[i][0]
    #     # angle = angle_between(vh_initial, vh_vector)
    #     angle = random_state.random() * 3.14
    #     # angle = random_state.random() * 3.14
    #     # angle = random_state.normal(0,1)

    #     # rotation_angle[i] = angle
    #     vh_embedding = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    #     # vh_embedding = np.identity(2)
    #     VH_embedding[i] = vh_embedding
 
    re_sum.fill(0)
    phi_sum.fill(0)
    
    dim = head_embedding.shape[1]

    for i in numba.prange(head.size):
        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]
        
        # current_VH = VH[j][:,:dim] # projection matrix; keep first dim dimensions 
        # other_VH = VH[k][:,:dim]
        
        # Restrict dim to 2
        # theta_j = rotation_angle[j]
        # theta_k = rotation_angle[k]
        
        current_VH = VH_embedding[j] # array shape of (dim, dim)
        other_VH = VH_embedding[k]
        # current_VH = np.identity(2)
        # other_VH = np.identity(2)
        
        # vec_diff = vdiff(current, other)
        vec_diff = other - current
        # inner_product = np.dot(vec_diff, vec_diff)
        dist_squared = rdist(current, other)
        
        phi = 1.0 / (1.0 + a * pow(dist_squared, b))
        phi_sum[j] += phi
        phi_sum[k] += phi
        
        for d in numba.prange(dim):
            vec_proj_vh_j = np.dot(vec_diff, current_VH[d]) # project to d-th rotation direction
            vec_proj_vh_k = np.dot(vec_diff, other_VH[d])        
     
            re_sum[j,d] += phi * vec_proj_vh_j * vec_proj_vh_j
            re_sum[k,d] += phi * vec_proj_vh_k * vec_proj_vh_k
        
        # vec_proj_vh_j = np.dot(current_VH, vec_diff) # project to rotation direction
        # vec_proj_vh_k = np.dot(other_VH, vec_diff)     
        
        # re_sum[j] += phi * vec_proj_vh_j * vec_proj_vh_j
        # re_sum[k] += phi * vec_proj_vh_k * vec_proj_vh_k
         
    epsilon = 1e-8
    for i in numba.prange(re_sum.shape[0]):
        re_sum[i] = np.log(epsilon + (re_sum[i] / phi_sum[i]))
    
    # for i in range(re_sum.shape[0]):
    #     re_sum[i] = np.log(epsilon + np.sqrt((re_sum[i] / phi_sum[i])))
   
        



def optimize_layout_euclidean_anisotropic_projection(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    # featuremap=False,
    featuremap_kwds=None,
    tqdm_kwds=None,
    move_other=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    featuremap: bool (optional, default False)
        Whether to use the feature-augmented featuremap objective
    featuremap_kwds: dict (optional, default None)
        Auxiliary data for featuremap
    tqdm_kwds: dict (optional, default None)
        Keyword arguments for tqdm progress bar.
    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
           _optimize_layout_euclidean_single_epoch_grad, 
           fastmath=True, 
           parallel=parallel,
         # nopython=True,
           cache=True
    )
    # optimize_fn = _optimize_layout_euclidean_single_epoch_grad
    
    if featuremap_kwds is None:
        featuremap_kwds = {}
    if tqdm_kwds is None:
        tqdm_kwds = {}
    
    # if featuremap:
    feat_init_fn = numba.njit(
         _optimize_layout_euclidean_featuremap_epoch_init_grad,
         fastmath=True,
         parallel=parallel,
        #  nopython=True,
         cache=True
    )
    
    
    # feat_init_fn = _optimize_layout_euclidean_featuremap_epoch_init_grad

    feat_mu_tot = np.sum(featuremap_kwds["mu_sum"]) / 2  # sum of all edges' existing probability, float, shape of (1,)
    # should we modify lambda? Yes, we should
    #TODO: modify lambda
    feat_lambda = featuremap_kwds["lambda"] 
    feat_R = featuremap_kwds["R"] # array shape of (n_vertices d)
    feat_VH = featuremap_kwds["VH"]
    # feat_VH_embedding = featuremap_kwds["VH_embedding"] # array of shape (n_vertices, dim)
    feat_VH_embedding = np.repeat(np.eye(2)[np.newaxis, :, :], head_embedding.shape[0], axis=0) # initialize vh_embedding by identity matrix
    feat_VH_embedding = feat_VH_embedding.astype(np.float32)

    # feat_rotation_angle = featuremap_kwds["rotation_angle"]
    feat_mu = featuremap_kwds["mu"] # edge probability
    feat_phi_sum = np.zeros(n_vertices, dtype=np.float32) # For each node i in embedding space, sum of edge existing probality incident to this node
    feat_re_sum = np.zeros([n_vertices, dim], dtype=np.float32) # Embedding radius in principal directions
    feat_var_shift = featuremap_kwds["var_shift"]
    # else: 
    #     feat_mu_tot = 0
    #     feat_lambda = 0
    #     feat_R = np.zeros(1, dtype=np.float32)
    #     feat_VH = np.zeros(1, dtype=np.float32)
    #     feat_VH_embedding = np.zeros(1, dtype=np.float32)
    #     # feat_rotation_angle = np.zeros(1, dtype=np.float32)
    #     feat_mu = np.zeros(1, dtype=np.float32)
    #     feat_phi_sum = np.zeros(1, dtype=np.float32)
    #     feat_re_sum = np.zeros(1, dtype=np.float32)
        
    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose
    
    for n in tqdm(range(n_epochs), **tqdm_kwds):
        featuremap_flag = (
            # featuremap and 
            (featuremap_kwds["lambda"] > 0)
            and (((n + 1) / float(n_epochs)) > (1 - featuremap_kwds["frac"]))
        )

        if featuremap_flag:
            # Compute the initial embedding under rotation VH
            # T1 = time.time()
            feat_init_fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                # random_state,
                # feat_VH,
                feat_VH_embedding,
                a,
                b,
                feat_re_sum,
                feat_phi_sum,
            )
            # T2 = time.time()
            # print(f'featuremap initialization time is {T2-T1}')
              
            # feat_init_fn.inspect_types()
            
            # FIXME: feat_var_shift might be referenced before assignment
            feat_re_std = np.sqrt(np.var(feat_re_sum, axis=0) + feat_var_shift)
            feat_re_mean = np.mean(feat_re_sum, axis=0)
            feat_re_sum_centered = np.subtract(feat_re_sum, feat_re_mean)

            product = np.diag(np.dot(feat_re_sum_centered.T, feat_R[:,:dim]))
            feat_re_cov = product / (n_vertices - 1)
        else:
            feat_re_std = np.zeros(dim, dtype=np.float32)
            feat_re_mean = np.zeros(dim, dtype=np.float32)
            feat_re_cov = np.zeros(dim, dtype=np.float32)

        # # recover the gauge from the low dimensional embedding
        # if featuremap_flag and n % 10 == 0:
        #     # print('recover gauge from embedding')
        #     recover_gauge_from_embedding(data_embedding=head_embedding, featuremap_kwds=featuremap_kwds)
        #     feat_VH_embedding = featuremap_kwds["VH_embedding"].astype(np.float32)

        # T1 = time.time()
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
            featuremap_flag,
            feat_phi_sum,
            feat_re_sum,
            feat_re_cov,
            feat_re_std,
            feat_re_mean,
            feat_lambda,
            feat_R,
            feat_VH_embedding,
            feat_mu,
            feat_mu_tot,
        )

        # T2 = time.time()
        # print(f'Optimize_fn time is {T2-T1}')
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
        
    return head_embedding

