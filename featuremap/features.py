#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:52:14 2023

"""


import anndata as ad
from anndata import AnnData
# from quasildr.structdr import Scms
import numpy as np
import time
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

from umap.umap_ import nearest_neighbors

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal

from featuremap.core_transition_state import kernel_density_estimate



# Create adata object for plotting
def create_adata(X, emb_featuremap, obs=None, var=None):
    adata = ad.AnnData(X=X)
    if obs is not None:
        adata.obs = obs
    if var is not None:
        adata.var = var
    adata.obsm['X_featmap'] = emb_featuremap.embedding_

    # find the shape of every item in emb_featuremap._featuremap_kwds
    for key, item in emb_featuremap._featuremap_kwds.items():
        # check if the item is an array
        if isinstance(item, np.ndarray): 
            if item.shape[0] == X.shape[0]:
                adata.obsm[key] = item
            elif item.shape[0] == X.shape[1]:
                adata.varm[key] = item
            else:
                print(f'{key} is not added to adata')

    return adata


def pseudotime_mst(adata, random_state, start_point_index):
    """
    Given a staring point, compute the distance to all other points on the mst
    """
    from featuremap import featuremap_
    from umap.umap_ import fuzzy_simplicial_set
    import numpy as np

    # Generate 2 embeddings for computing MST, and average the pseudotime
    pseudotime_mst = np.zeros(adata.shape[0])
    for i in range(2):
        # pairwise distances of knn graph
        # check if 'X_featmap_v_3d' exists in obsm
        if 'X_featmap_v_3d' not in adata.obsm.keys():
            adata.obsm['X_featmap_v_3d'] = featuremap_.FeatureMAP(n_components=3, output_variation=True,).fit_transform(adata.X)

        _, _,_,dists = fuzzy_simplicial_set(adata.obsm['X_featmap_v_3d'], n_neighbors=60, random_state=random_state,  
                                            metric='euclidean', metric_kwds={}, verbose=False, return_dists=True)
        # Minimum Spanning Tree on kgraph
        from scipy.sparse.csgraph import minimum_spanning_tree
        # Compute the minimum spanning tree
        # including the graphs with disconnected components
        mst = minimum_spanning_tree(dists)
        # mst = mst + mst.T 
        # transform to coo matrix
        mst = mst.tocoo()

        # Given a staring point, compute the distance to all other points on the mst
        import numpy as np
        import networkx as nx

        # Create a weighted graph from the MST
        G = nx.from_scipy_sparse_array(mst, edge_attribute='weight')

        # Compute the weighted shortest path from the starting point to all other points on the MST
        shortest_paths = nx.shortest_path_length(G, source=start_point_index, weight='weight')
        
        # Convert the shortest paths to an array
        distances = np.zeros(len(shortest_paths))
        for i, d in shortest_paths.items():
            distances[i] = d
        pseudotime_mst += distances

        if adata.shape[0] > 5000:
            from sklearn.preprocessing import MinMaxScaler
            pseudotime_mst = MinMaxScaler().fit_transform(pseudotime_mst.reshape(-1,1)).reshape(-1)
            adata.obs['feat_pseudotime'] = pseudotime_mst
            return pseudotime_mst

    pseudotime_mst /= 2
    from sklearn.preprocessing import MinMaxScaler
    pseudotime_mst = MinMaxScaler().fit_transform(pseudotime_mst.reshape(-1,1)).reshape(-1)
    adata.obs['feat_pseudotime'] = pseudotime_mst





def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor * 5



def plot_gauge(
        adata:AnnData,
        embedding='X_featmap',
        vkey='gauge_v1_emb',
        density=1,
        smooth=0.5,
        n_neighbors=None,
        min_mass=1,
        autoscale=True,
        ):
    # Set grid as the support
    X_emb=adata.obsm[embedding]  # Exclude one leiden cluster;
    # rotational_matrix = adata.uns['emb_umap']._densmap_kwds['VH_embedding']
    # rotational_matrix = adata.obsm['VH_embedding']
    # r_emb = adata.obsm['rad_emb_no_log']
    # s = Scms(X_emb, 0.5, min_radius=5)

    # X_emb=adata.obsm[embedding]
    V_emb=adata.obsm[vkey] 
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T
    
    # p1, _, _, _, C = s._kernel_density_estimate_anisotropic(
    #   X_grid, rotational_matrix, r_emb)
    # p1, _, _, _ = s._kernel_density_estimate(X_grid)
    p1 = kernel_density_estimate(X_emb, X_grid)
    
    
    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)
    
    # p_mass = p1
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    # V_grid = V_emb[neighs] 
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1    
    
    min_mass *= np.percentile(p_mass, 99) / 100
    # min_mass = 0.01
    X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]
    
    if autoscale:
          V_grid /= quiver_autoscale(X_grid, V_grid)

    plt.contourf(meshes_tuple[0], meshes_tuple[1], p1.reshape(int(50 * density),int(50 * density)),
                  levels=20, cmap='Blues')
    emb = adata.obsm[embedding]
    # color = np.array(adata.obs['leiden']).astype(int)
    # plt.scatter(emb[:,0],emb[:,1], s=1, c=color, cmap='Set2', alpha=0.1)
    plt.scatter(emb[:,0],emb[:,1], s=1, alpha=0.1)
    plt.title('Eigengene')
    plt.xticks([])
    plt.yticks([])
    plt.quiver(X_grid[:,0], X_grid[:,1],V_grid[:,0],V_grid[:,1],color='black',alpha=1,scale=3)
    # plt.show()
    # plt.clf()
    



def plot_gauge_both(
        adata:AnnData,
        embedding='X_featmap',
        # vkey='gauge_v1_emb',
        density=1,
        smooth=0.5,
        n_neighbors=None,
        min_mass=1,
        autoscale=True,
        ):
    # Set grid as the support
    X_emb=adata.obsm[embedding]  # Exclude one leiden cluster;
    # rotational_matrix = adata.uns['emb_umap']._densmap_kwds['VH_embedding']
    # rotational_matrix = adata.obsm['VH_embedding']
    # r_emb = adata.obsm['rad_emb_no_log']
    # s = Scms(X_emb, 0.5, min_radius=5)

    # X_emb=adata.obsm[embedding]
    vkey='gauge_v1_emb'
    V_emb=adata.obsm[vkey] 
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T
    
    # p1, _, _, _, C = s._kernel_density_estimate_anisotropic(
    #   X_grid, rotational_matrix, r_emb)
    # p1, _, _, _ = s._kernel_density_estimate(X_grid)
    p1 = kernel_density_estimate(X_emb, X_grid)
    
    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)
    
    # p_mass = p1
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    # V_grid = V_emb[neighs] 
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1    
    
    min_mass *= np.percentile(p_mass, 99) / 100
    # min_mass = 0.01
    X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]
    
    if autoscale:
          V_grid /= quiver_autoscale(X_grid, V_grid)

    plt.contourf(meshes_tuple[0], meshes_tuple[1], p1.reshape(int(50 * density),int(50 * density)),
                  levels=20, cmap='Greys')
    emb = adata.obsm[embedding]
    # color = np.array(adata.obs['leiden']).astype(int)
    # plt.scatter(emb[:,0],emb[:,1], s=1, c=color, cmap='Set2', alpha=0.1)
    plt.scatter(emb[:,0],emb[:,1], s=1, alpha=0.1, c='gray')    
    plt.title('Eigengene')
    plt.xticks([])
    plt.yticks([])
    qv1 = plt.quiver(X_grid[:,0], X_grid[:,1],V_grid[:,0],V_grid[:,1],color='red',alpha=1,scale=3)
    



    # X_emb=adata.obsm[embedding]
    vkey='gauge_v2_emb'
    V_emb=adata.obsm[vkey] 
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T
    
    # p1, _, _, _, C = s._kernel_density_estimate_anisotropic(
    #   X_grid, rotational_matrix, r_emb)
    # p1, _, _, _ = s._kernel_density_estimate(X_grid)
    p1 = kernel_density_estimate(X_emb, X_grid)

    
    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)
    
    # p_mass = p1
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    # V_grid = V_emb[neighs] 
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1    
    
    min_mass *= np.percentile(p_mass, 99) / 100
    # min_mass = 0.01
    X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]
    
    if autoscale:
          V_grid /= quiver_autoscale(X_grid, V_grid)

    # plt.contourf(meshes_tuple[0], meshes_tuple[1], p1.reshape(int(50 * density),int(50 * density)),
    #               levels=20, cmap='Blues')
    emb = adata.obsm[embedding]
    # color = np.array(adata.obs['leiden']).astype(int)
    # plt.scatter(emb[:,0],emb[:,1], s=1, c=color, cmap='Set2', alpha=0.1)
    # plt.scatter(emb[:,0],emb[:,1], s=1, alpha=0.1)
    plt.title('Eigengene')
    plt.xticks([])
    plt.yticks([])
    qv2 =  plt.quiver(X_grid[:,0], X_grid[:,1],V_grid[:,0],V_grid[:,1],color='blue',alpha=1,scale=3)
    
    # create a legend by arrows, red for v1, blue for v2
    legend = plt.legend([qv1, qv2], ['v1', 'v2'])
    legend.set_bbox_to_anchor((1, 1))  # Set the shape of the legend
    # plt.show()
    # plt.clf()
    
    
def matrix_multiply(X, Y):
   # X shape: (11951, 60, 100)
   # Y shape: (100, 14577)
   # The goal is to multiply each 60x100 matrix in X with Y, resulting in 11951 matrices of size 60x14577

   # Reshape X to a 2D array for matrix multiplication
   X_reshaped = X.reshape(-1, Y.shape[0])  # Shape becomes (11951*60, 100)
   
   # Perform matrix multiplication
   result = np.dot(X_reshaped, Y)  # Resulting shape is (11951*60, 14577)
   
   # Reshape the result back to 3D
   result_reshaped = result.reshape(X.shape[0], X.shape[1], Y.shape[1])  # Shape becomes (11951, 60, 14577)

   return result_reshaped


from multiprocessing import Pool
# import itertools
def compute_norm_chunk(array, start, end):
    # Slice the actual array
    chunk = array[start:end]
    return np.linalg.norm(chunk, axis=1)

def compute_norm_parallel(array, chunk_size):
    # Split the first dimension into chunks
    ranges = [(i, min(i + chunk_size, array.shape[0])) for i in range(0, array.shape[0], chunk_size)]

    with Pool() as pool:
        # Map the compute_norm_chunk function to each chunk
        results = pool.starmap(compute_norm_chunk, [(array, r[0], r[1]) for r in ranges])
    # Concatenate the results
    return np.concatenate(results)


def local_intrinsic_dim(
        adata: AnnData,
        threshold=0.9,
):
    singular_values_collection = adata.obsm['Singular_value'].copy()
    
    # Compute intrinsic dimensionality locally
    def pc_accumulation(arr, threshold):
        arr_sum = np.sum(np.square(arr))
        temp_sum = 0
        for i in range(arr.shape[0]):
            temp_sum += arr[i] * arr[i]
            if temp_sum > arr_sum * threshold:
                return i
    
    intrinsic_dim = np.zeros(adata.shape[0]).astype(int)
    
    for i in range(adata.shape[0]):            
        intrinsic_dim[i] = pc_accumulation(singular_values_collection[i], threshold)
    plt.hist(intrinsic_dim)
    plt.title('Local_intrinsic_dim')
    plt.show()
    plt.clf()
    
    adata.obs['intrinsic_dim'] = intrinsic_dim
    return intrinsic_dim

# @numba.njit()
def feature_variation(
        adata: AnnData,
        threshold=0.5,
            ):
    """
    Compute the feature variation and feature loadings based on local SVD.
    
    Parameters
    ----------
    adata : AnnData
        An annotated data matrix.
    """
    import numpy as np
    vh_smoothed = adata.obsm['vh_smoothed'].copy()
    svd_vh = adata.varm['svd_vh'].copy().T

    # gauge_vh_original = adata.obsm['gauge_vh_original'].copy()
    # vh_smoothed = gauge_vh_original
    
    intrinsic_dim = local_intrinsic_dim(adata, threshold)
    
    # sc.pl.embedding(adata,'featmap',color=['instrinsic_dim'],cmap='plasma')
    # sc.pl.embedding(adata,'umap',color=['instrinsic_dim'],cmap='plasma')
    
    # Compute the gene norm in top k PCs (norm of the arrow in biplot)
    k = int(np.median(intrinsic_dim))
    print(f'k is {k}')
    
    print("Start matrix multiplication")
    T1 = time.time()
    # pcVals_project_back = np.matmul(vh_smoothed, svd_vh[np.newaxis, :])
    # pcVals_project_back = np.matmul(vh_smoothed, svd_vh)
    pcVals_project_back = np.matmul(vh_smoothed[:,:k,:], svd_vh)
    # pcVals_project_back = np.matmul(np.squeeze(vh_smoothed[:,:k,:]), svd_vh)

    # pcVals_project_back =  matrix_multiply(vh_smoothed, svd_vh)
    T2 = time.time()
    print(f'Finish matrix multiplication in {T2-T1}')
    
    T1 = time.time()
    # gene_val_norm = np.linalg.norm(pcVals_project_back[:, :k, :], axis=1)
    gene_val_norm = np.linalg.norm(pcVals_project_back, axis=1)
    # gene_val_norm = np.abs(pcVals_project_back)
    adata.layers['variation_feature'] = gene_val_norm
    T2 = time.time()
    print(f'Finish norm calculation in {T2-T1}')
    
    # T1 = time.time()        
    # gene_norm_first_two = np.linalg.norm(pcVals_project_back[:, :2, :], axis=1)
    # pc_loadings_scale = pcVals_project_back[:, :2, :] /\
    #     gene_norm_first_two[:,np.newaxis,:] *\
    #         gene_val_norm[:,np.newaxis,:]
    
    # # pc_loadings_scale = pcVals_project_back[:, :2, :] /\
    # #     np.linalg.norm(pcVals_project_back[:, :2, :], axis=1)[:,np.newaxis,:] 
    
    # adata.obsm['feature_loading_scale'] = pc_loadings_scale
    # T2 = time.time()
    # print(f'Finish feature loading in {T2-T1}')
    
    # # Feature loadings on each local gauge
    # gauge_vh_emb = adata.obsm['VH_embedding']
    # feature_loading_emb = adata.obsm['feature_loading_scale'] 
    # feature_loadings_embedding = np.matmul(feature_loading_emb.transpose(0,2,1), gauge_vh_emb.transpose(0,2,1)) # Project to gauge_embedding
    # adata.obsm['feature_loading_embedding'] = feature_loadings_embedding.transpose(0,2,1)


# @numba.njit()
def feature_projection(
        adata: AnnData,
        feature='',
        # parallel=False
            ):
    """
    Compute the feature variation and feature loadings based on local SVD.
    
    Parameters
    ----------
    adata : AnnData
        An annotated data matrix.
    """
    
    vh_smoothed = adata.obsm['vh_smoothed'].copy()
    # gauge_vh_original = adata.obsm['gauge_vh_original'].copy()
    # vh_smoothed = gauge_vh_original

    # gauge_u = adata.obsm['gauge_u'].copy()
    # singular_values_collection = adata.obsm['Singular_value'].copy()
    svd_vh = adata.varm['svd_vh'].copy().T

    # Compute the gene norm in top k PCs (norm of the arrow in biplot)
    print("Start matrix multiplication")
    T1 = time.time()
    feature_index = np.where(adata.var.index == feature)[0][0]
    pcVals_project_back_feature = np.matmul(np.array(vh_smoothed)[:,:2,:], svd_vh[:,feature_index].reshape(-1,1))
    T2 = time.time()
    print(f'Finish matrix multiplication in {T2-T1}')
    # feature_index = np.where(adata.var.index == feature)[0][0]
    # pcVals_project_back_feature = np.matmul(np.array(vh_smoothed)[:,:2,:], svd_vh[:,feature_index].reshape(-1,1))

    # Feature loadings on each local gauge
    gauge_vh_emb = adata.obsm['VH_embedding']
    # feature_loading_emb = adata.obsm['feature_loading_scale'] 
    feature_loadings_embedding = np.matmul(pcVals_project_back_feature.transpose(0,2,1), gauge_vh_emb.transpose(0,2,1)) # Project to gauge_embedding
    adata.obsm[f'feature_{feature}_loading'] = np.squeeze(feature_loadings_embedding)




 
def plot_feature(
        adata:AnnData,
        feature='',
        feature_loading_emb='feature_loading_embedding',
        embedding='X_featmap',
        cluster_key='clusters',
        plot_within_cluster=[],
        pseudotime_adjusted=False,
        pseudotime='dpt_pseudotime',
        trend='positive',
        ratio=0.2,
        density=1,
        smooth=0.5,
        n_neighbors=None,
        min_mass=1,
        autoscale=True,):
    """
    Plot a given feature (e.g., gene) in two dimensional visualization

    Parameters
    ----------
    adata : AnnData
        An annotated data matrix.
    feature : string
        Feature name to be plotted.
    embedding : string
        Embedding background for feature plot. The default is 'X_featmap'.
    cluster_key : string
        Cluster name indicator. The default is 'clusters'.
    plot_within_cluster : list
        A list of clusters in which the feaure is to plot. The default is [].
    pseudotime_adjusted : bool
        Whether to adjust the feature direction by pseudotime. The default is False.
    pseudotime : string
        Pseudotime indicator. The default is 'dpt_pseudotime'.
    trend : string of {'positive','negative'}
        The direction along pseudotime. The default is 'positive'.
    ratio : float
        Filtering ratio by expression to filter varition by low expression. The default is 0.5.
    density : float
        Grid desity for plot. The default is 1.
    smooth : float
        For kde estimation. The default is 0.5.
    n_neighbors : int
        Number of neighbours for kde. The default is None.
    min_mass : float
        Minumum denstiy to show the grid plot. The default is 1.
    autoscale : bool
        Scale the arrow plot. The default is True.

   """
    # Compute the feature loading embedding
    # feature_loading(adata)
   
    vkey=f'feature_{feature}_loading'

    feature_id = np.where(adata.var_names == feature)[0][0]
    adata.obsm[vkey] = adata.obsm[feature_loading_emb][:,:,feature_id]

    # Set grid as the support
    X_emb=adata.obsm[embedding]
    # rotational_matrix = adata.uns['emb_umap']._densmap_kwds['VH_embedding']
    # rotational_matrix = adata.obsm['VH_embedding']
    # r_emb = adata.obsm['rad_emb_no_log']
    # s = Scms(X_emb, 0.5, min_radius=5)
    
    V_emb=adata.obsm[vkey] 
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # p1, _, _, _, C = s._kernel_density_estimate_anisotropic(
    #   X_grid, rotational_matrix, r_emb)
    # p1, _, _, _ = s._kernel_density_estimate(X_grid)
    p1 = kernel_density_estimate(X_emb, X_grid)

     
    # estimate grid variation
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)
    
    # p_mass = p1
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    # V_grid = V_emb[neighs] 
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1    
    
    # Restrict the plot within given clusters
    def grid_within_cluster(X_grid):
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(X_emb)
        _, neighs = nn.kneighbors(X_grid)
        
        # plot_within_cluster = ['Beta']
        if len(plot_within_cluster) > 0:
            grid_in_cluster = []
            for cluster in plot_within_cluster:
                idx_in_cluster = np.where(np.array(adata.obs[cluster_key] == cluster))[0]
                for i in range(neighs.shape[0]):
                    if neighs[i,0] in idx_in_cluster:
                        grid_in_cluster.append(i)
        return grid_in_cluster

    # start ploting feature 
    feature_id = np.where(adata.var_names == feature)[0][0]
    # average expression in grid points over NNs
    expr_grid = []
    
    if isinstance(adata.X, np.ndarray):
        expr_count = adata.X.copy()[:,feature_id]
    else:
        expr_count = adata.X.toarray().copy()[:,feature_id]

    
    expr_grid = (expr_count[neighs] * weight).sum(1)
    expr_grid /= np.maximum(1, p_mass)
    
    # Filter the expr_velo by low expression 
    threshold = max(expr_grid) * ratio
    # feature_velo_loading = pc_loadings_grid[:,:,feature_id]
    V_grid[expr_grid<threshold]=np.nan
    
    min_mass *= np.percentile(p_mass, 99) / 100
    # min_mass = 0.01
    X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]
    if autoscale:
          V_grid /=  quiver_autoscale(X_grid, V_grid)
          
    # Adjust the v direction by the sign of local expression change
    # V_grid = V_grid * 10
    displace_grid = X_grid + V_grid
    grid_idx = np.unique(np.where(np.isnan(displace_grid) == False)[0])
    _, displace_grid_neighs = nn.kneighbors(displace_grid[grid_idx])
    _, start_grid_neighs = nn.kneighbors(X_grid[grid_idx])
    displace_expr = np.mean(expr_count[displace_grid_neighs[:,:100]], axis=1) - np.mean(expr_count[start_grid_neighs[:,:100]],axis=1)
    displace_expr_sign = np.sign(displace_expr)
    # displace_expr_sign[displace_expr_sign == 0] = 1
    V_grid[grid_idx] = np.multiply(V_grid[grid_idx], displace_expr_sign[:, np.newaxis])
    
    
    # Keep arrows along the positive (negative) trend of time flow 
    if pseudotime_adjusted:
        time_ = np.array(adata.obs[pseudotime])
        
        displace_grid_adjusted = X_grid + V_grid
        grid_idx_adjusted = np.unique(np.where(np.isnan(displace_grid_adjusted) == False)[0])
        _, displace_grid_neighs = nn.kneighbors(displace_grid_adjusted[grid_idx_adjusted])
        _, start_grid_neighs = nn.kneighbors(X_grid[grid_idx_adjusted])
        displace_time = np.mean(time_[displace_grid_neighs[:,:100]], axis=1) - np.mean(time_[start_grid_neighs[:,:100]],axis=1)
        displace_time_sign = np.sign(displace_time)
        
        if trend == 'positive':
            displace_time_sign[displace_time_sign < 0] = 0
        else:
            displace_time_sign[displace_time_sign > 0] = 0
            displace_time_sign[displace_time_sign < 0] = 1
    
        V_grid[grid_idx_adjusted] = np.multiply(V_grid[grid_idx_adjusted], displace_time_sign[:, np.newaxis])

   
    # plt.contourf(meshes_tuple[0], meshes_tuple[1], p1.reshape(int(50 * density),int(50 * density)),
                #   levels=20, cmap='Blues')
    emb = adata.obsm[embedding]
    # color = np.array(adata.obs['leiden']).astype(int)
    # plt.scatter(emb[:,0],emb[:,1], s=1, c=color, cmap='Set2', alpha=0.1)
    plt.scatter(emb[:,0],emb[:,1], s=1, alpha=0.1)
    plt.title(feature)
    plt.xticks([])
    plt.yticks([])
    if len(plot_within_cluster) > 0:
        grid_in_cluster = grid_within_cluster(X_grid)
        plt.quiver(X_grid[grid_in_cluster,0], X_grid[grid_in_cluster,1],V_grid[grid_in_cluster,0],V_grid[grid_in_cluster,1],color='black',alpha=1)
    else:
        plt.quiver(X_grid[:,0], X_grid[:,1],V_grid[:,0],V_grid[:,1],color='black',alpha=1,scale=2)
    plt.xlabel(feature_loading_emb)
    # save the plot
    plt.savefig(f'./figures/cd8/gene_{feature}_emb_{embedding}.png')
    plt.show()
    plt.clf()

    del adata.obsm[vkey]
    # plt.savefig(f'./data/flow/gene_{feature}.pdf')

def plot_one_feature(
        adata:AnnData,
        feature='',
        embedding='X_featmap',
        cluster_key='clusters',
        plot_within_cluster=[],
        pseudotime_adjusted=False,
        pseudotime='dpt_pseudotime',
        trend='positive',
        ratio=0.2,
        density=1,
        smooth=0.5,
        n_neighbors=None,
        min_mass=1,
        autoscale=True,):
    """
    Plot a given feature (e.g., gene) in two dimensional visualization

    Parameters
    ----------
    adata : AnnData
        An annotated data matrix.
    feature : string
        Feature name to be plotted.
    embedding : string
        Embedding background for feature plot. The default is 'X_featmap'.
    cluster_key : string
        Cluster name indicator. The default is 'clusters'.
    plot_within_cluster : list
        A list of clusters in which the feaure is to plot. The default is [].
    pseudotime_adjusted : bool
        Whether to adjust the feature direction by pseudotime. The default is False.
    pseudotime : string
        Pseudotime indicator. The default is 'dpt_pseudotime'.
    trend : string of {'positive','negative'}
        The direction along pseudotime. The default is 'positive'.
    ratio : float
        Filtering ratio by expression to filter varition by low expression. The default is 0.5.
    density : float
        Grid desity for plot. The default is 1.
    smooth : float
        For kde estimation. The default is 0.5.
    n_neighbors : int
        Number of neighbours for kde. The default is None.
    min_mass : float
        Minumum denstiy to show the grid plot. The default is 1.
    autoscale : bool
        Scale the arrow plot. The default is True.

   """
    # Compute the feature loading embedding
    # feature_loading(adata)
   
    vkey=f'feature_{feature}_loading'

    feature_id = np.where(adata.var_names == feature)[0][0]
    # adata.obsm[vkey] = adata.obsm[feature_loading_emb][:,:,feature_id]

    # Set grid as the support
    X_emb=adata.obsm[embedding]
    # rotational_matrix = adata.uns['emb_umap']._densmap_kwds['VH_embedding']
    # rotational_matrix = adata.obsm['VH_embedding']
    # r_emb = adata.obsm['rad_emb_no_log']
    # s = Scms(X_emb, 0.5, min_radius=5)
    
    V_emb=adata.obsm[vkey] 
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # p1, _, _, _, C = s._kernel_density_estimate_anisotropic(
    #   X_grid, rotational_matrix, r_emb)
    # p1, _, _, _ = s._kernel_density_estimate(X_grid)
    p1 = kernel_density_estimate(X_emb, X_grid)

     
    # estimate grid variation
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)
    
    # p_mass = p1
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    # V_grid = V_emb[neighs] 
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1    
    
    # V_grid /= np.max(np.linalg.norm(V_grid, axis=1))

    
    # Restrict the plot within given clusters
    def grid_within_cluster(X_grid):
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(X_emb)
        _, neighs = nn.kneighbors(X_grid)
        
        # plot_within_cluster = ['Beta']
        if len(plot_within_cluster) > 0:
            grid_in_cluster = []
            for cluster in plot_within_cluster:
                idx_in_cluster = np.where(np.array(adata.obs[cluster_key] == cluster))[0]
                for i in range(neighs.shape[0]):
                    if neighs[i,0] in idx_in_cluster:
                        grid_in_cluster.append(i)
        return grid_in_cluster

    # start ploting feature 
    feature_id = np.where(adata.var_names == feature)[0][0]
    # average expression in grid points over NNs
    expr_grid = []
    
    if isinstance(adata.X, np.ndarray):
        expr_count = adata.X.copy()[:,feature_id]
    else:
        expr_count = adata.X.toarray().copy()[:,feature_id]

    
    expr_grid = (expr_count[neighs] * weight).sum(1)
    expr_grid /= np.maximum(1, p_mass)
    
    # Filter the expr_velo by low expression 
    threshold = max(expr_grid) * ratio
    # feature_velo_loading = pc_loadings_grid[:,:,feature_id]
    V_grid[expr_grid<threshold]=np.nan
    
    min_mass *= np.percentile(p_mass, 99) / 100
    # min_mass = 0.01
    X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]
    if autoscale:
          V_grid /=  1* quiver_autoscale(X_grid, V_grid)
          
    # Adjust the v direction by the sign of local expression change
    # V_grid = V_grid * 10
    displace_grid = X_grid + V_grid
    grid_idx = np.unique(np.where(np.isnan(displace_grid) == False)[0])
    _, displace_grid_neighs = nn.kneighbors(displace_grid[grid_idx])
    _, start_grid_neighs = nn.kneighbors(X_grid[grid_idx])
    displace_expr = np.mean(expr_count[displace_grid_neighs[:,:100]], axis=1) - np.mean(expr_count[start_grid_neighs[:,:100]],axis=1)
    displace_expr_sign = np.sign(displace_expr)
    # displace_expr_sign[displace_expr_sign == 0] = 1
    V_grid[grid_idx] = np.multiply(V_grid[grid_idx], displace_expr_sign[:, np.newaxis])
    
    
    # Keep arrows along the positive (negative) trend of time flow 
    if pseudotime_adjusted:
        time_ = np.array(adata.obs[pseudotime])
        
        displace_grid_adjusted = X_grid + V_grid
        grid_idx_adjusted = np.unique(np.where(np.isnan(displace_grid_adjusted) == False)[0])
        _, displace_grid_neighs = nn.kneighbors(displace_grid_adjusted[grid_idx_adjusted])
        _, start_grid_neighs = nn.kneighbors(X_grid[grid_idx_adjusted])
        displace_time = np.mean(time_[displace_grid_neighs[:,:100]], axis=1) - np.mean(time_[start_grid_neighs[:,:100]],axis=1)
        displace_time_sign = np.sign(displace_time)
        
        if trend == 'positive':
            displace_time_sign[displace_time_sign < 0] = 0
        else:
            displace_time_sign[displace_time_sign > 0] = 0
            displace_time_sign[displace_time_sign < 0] = 1
    
        V_grid[grid_idx_adjusted] = np.multiply(V_grid[grid_idx_adjusted], displace_time_sign[:, np.newaxis])

   
    # plt.contourf(meshes_tuple[0], meshes_tuple[1], p1.reshape(int(50 * density),int(50 * density)),
                #   levels=20, cmap='Blues')
    ax = plt.subplot()
    emb = adata.obsm[embedding]
    # color = np.array(adata.obs['clusters'])
    import seaborn as sns
    # sns.scatterplot(data=adata.obs, x=emb[:,0], y=emb[:,1], s=20, hue='clusters', palette='tab10', alpha=0.5, legend=False)
    # plt.scatter(emb[:,0],emb[:,1], s=1, c=color, cmap='tab10', alpha=0.1)
    plt.scatter(emb[:,0],emb[:,1], s=1, alpha=0.1)
    plt.title(feature)
    plt.xticks([])
    plt.yticks([])
    # check if 'clusters' is in adata.obs
    if 'clusters' in adata.obs:
        sc.pl.embedding(adata, embedding, color='clusters', projection='2d', size=20, ax=ax, show=False)

    if len(plot_within_cluster) > 0:
        grid_in_cluster = grid_within_cluster(X_grid)
        plt.quiver(X_grid[grid_in_cluster,0], X_grid[grid_in_cluster,1],V_grid[grid_in_cluster,0],V_grid[grid_in_cluster,1],color='black',alpha=1)
    else:
        plt.quiver(X_grid[:,0], X_grid[:,1],V_grid[:,0],V_grid[:,1],color='black',alpha=1,scale=2)
    plt.xlabel(vkey)
    # save the plot
    plt.show()
    plt.clf()

    del adata.obsm[vkey]
    # plt.savefig(f'./data/flow/gene_{feature}.pdf')



def variation_feature_pp(adata):
    import anndata as ad
    layer = 'variation_feature'
    adata_var = ad.AnnData(X=adata.layers[layer].copy(), )
    adata_var.obs = adata.obs.copy()
    adata_var.var = adata.var.copy()
    adata_var.layers['counts'] = adata.X.copy() 
    
    adata_var.X[np.isnan(adata_var.X)]=0

    adata_var.obs_names = adata.obs_names
    adata_var.var_names = adata.var_names
    adata_var.obs['clusters'] = adata.obs['clusters'].copy()
    adata_var.layers['counts'] = adata.X.copy()

    # Normalization
    sc.pp.normalize_total(adata_var, target_sum=1e4 )
    sc.pp.log1p(adata_var, )

    # Filtering variation for DGV 
    adata_var.layers['var_filter'] = adata_var.X.copy()
    # Filter low variation
    idx = adata_var.layers['var_filter'] < np.max(adata_var.layers['var_filter']) * 0.2
    # idx = adata_var.layers['var_filter'] < np.quantile(adata_var.layers['var_filter'], 0.2)
    # print(f'Low var ratio is {np.sum(idx) / (idx.shape[0]*idx.shape[1])}')
    adata_var.layers['var_filter'][idx] = 0

    # Filter variation by low count
    if isinstance(adata.X, np.ndarray):
        idx = adata.X < np.max(adata.X) * 0.2
    else:
        idx = adata.X.toarray() < np.max(adata.X.toarray()) * 0.2

    # idx = adata.X.toarray() < np.quantile(adata.X.toarray()[np.nonzero(adata.X.toarray())], 0.2)
    # idx = adata.X < np.max(adata.X) * 0.2
    # print(f'Low var ratio by expression is {np.sum(idx) / (idx.shape[0]*idx.shape[1])}')
    adata_var.layers['var_filter'][idx] = 0
    # Normalization
    sc.pp.normalize_total(adata_var, target_sum=1e4, layer='var_filter' )
    sc.pp.log1p(adata_var, layer='var_filter')

    return adata_var


def feature_variation_embedding(
        adata,
        n_components=2,
        layer = 'variation_feature',
        variation_preprocess_flag=False,
        random_state=42
        ):
    
    adata_var = ad.AnnData(X=adata.layers[layer].copy(), )
    adata_var.X[np.isnan(adata_var.X)]=0
    
    adata_var.obs_names = adata.obs_names
    adata_var.var_names = adata.var_names
    adata_var.obs['clusters'] = adata.obs['clusters'].copy()
    adata_var.layers['counts'] = adata.X.copy()
    
    # Normalization
    # sc.pl.highest_expr_genes(adata_var, n_top=20,)
    sc.pp.normalize_total(adata_var, target_sum=1e4 )
    sc.pp.log1p(adata_var, )
    
    if variation_preprocess_flag:
        # Filtering variation for DGV 
        adata_var.layers['var_filter'] = adata_var.X.copy()
        # Filter low variation
        idx = adata_var.layers['var_filter'] < np.max(adata_var.layers['var_filter']) * 0.2
        # idx = adata_var.layers['var_filter'] < np.quantile(adata_var.layers['var_filter'], 0.2)
        # print(f'Low var ratio is {np.sum(idx) / (idx.shape[0]*idx.shape[1])}')
        adata_var.layers['var_filter'][idx] = 0
        
        # Filter variation by low count
        if isinstance(adata.X, np.ndarray):
            idx = adata.X < np.max(adata.X) * 0.2
        else:
            idx = adata.X.toarray() < np.max(adata.X.toarray()) * 0.2

        # idx = adata.X.toarray() < np.quantile(adata.X.toarray()[np.nonzero(adata.X.toarray())], 0.2)

        # idx = adata.X < np.max(adata.X) * 0.2
        # print(f'Low var ratio by expression is {np.sum(idx) / (idx.shape[0]*idx.shape[1])}')
        adata_var.layers['var_filter'][idx] = 0
        # Normalization
        sc.pp.normalize_total(adata_var, target_sum=1e4, layer='var_filter' )
        sc.pp.log1p(adata_var, layer='var_filter')
    
    # Variation embedding
    data_original = adata_var.X.copy()
    data_original[np.isnan(data_original)] = 0
    
    print('Start PCA')
    # PCA by svd
    import scipy
    u, s, vh = scipy.sparse.linalg.svds(
        data_original, k= min(data_original.shape[1]-1, 100), which='LM', random_state=42)
    # u, s, vh = scipy.linalg.svd(gene_val_norm, full_matrices=False)
    # PCA coordinates in first 100 dims
    emb_svd = np.matmul(u, np.diag(s))
    
    print('Start embedding')
    import umap
    emb_umap = umap.UMAP(random_state=random_state, n_neighbors=30,min_dist=0.5, spread=1, n_components=n_components).fit(emb_svd)
    adata_var.obsm['X_featmap_v'] = emb_umap.embedding_
    sc.pl.embedding(adata_var, 'featmap_v', legend_fontsize=10,color=['clusters'], projection='2d', size=20, )
    
    # sc.pl.embedding(adata_var, 'umap_v', legend_fontsize=10,color=['clusters_original'], projection='2d', size=20, )
    adata.obsm['X_featmap_v'] = adata_var.obsm['X_featmap_v']
    
    return adata_var
    
def featuremap_var_3d(emb_var_3d, color=None, symbol=None, marker_size=3):
    import plotly   
    # importlib.reload(nbformat)
    import plotly.express as px
    fig_3d = px.scatter_3d(emb_var_3d, x=0, y=1, z=2, color=color, symbol=symbol)
    if marker_size is None:
        marker_size = 120000 / emb_var_3d.shape[0]
    fig_3d.update_traces(marker_size=marker_size) # Modify the point size
    fig_3d.update_layout(autosize=False, width=500,height=500,)
    fig_3d.show()

