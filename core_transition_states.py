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
import networkx as nx

import seaborn as sns

from featuremap.featuremap_ import nearest_neighbors

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal

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

def plot_density(
        adata: AnnData,
        emb = 'featmap',
            ):
    """
    Plot the density of the embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    emb : str
        The embedding space to plot the density.

    
    """
    data = adata.obsm[f'X_{emb}'].copy()  # Exclude one leiden cluster;

    min_x = min(data[:, 0])
    max_x = max(data[:, 0])
    min_y = min(data[:, 1])
    max_y = max(data[:, 1])
    # part = 200
    if data.shape[0] < 5000:
        num_grid_point = data.shape[0] * 0.5
    else:
        num_grid_point = 2000
    x_range = max_x - min_x
    y_range = max_y - min_y
    # x_range = 1 - 0.618
    # y_range = 0.618
    part_y = np.sqrt(num_grid_point / x_range * y_range)
    part_x = x_range / y_range * part_y
    # part_y = 60
    # part_x = 60
    # Assign num of grid points mort to vertical direction ??
    xv, yv = np.meshgrid(np.linspace(min_x, max_x, round(part_x)), np.linspace(min_y, max_y, round(part_y)),
                         sparse=False, indexing='ij')
    # xv, yv = np.meshgrid(np.linspace(-10, 10, part), np.linspace(-10, 15, part),
    #                       sparse=False, indexing='ij')
    grid_contour = np.column_stack([np.concatenate(xv), np.concatenate(yv)])
    # T1 = time.time()
    # p1, g1, h1, msu,_ = s._kernel_density_estimate_anisotropic(grid_contour, rotational_matrix, r_emb)
    p1 = kernel_density_estimate(data=data, X=grid_contour, output_onlylogp=False, )

    # T2 = time.time()
    # print('Finish kernel_density_estimate_anisotropic in ' + str(T2-T1))
    # ifilter_1 = np.where(p1 >= (np.max(p1)*0.05))[0]  # sampling
    # fig, ax = plt.subplots(figsize=(15, 15))
    plt.contourf(xv, yv, p1.reshape(round(part_x), round(part_y)),
                 levels=20, cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.clf()

def compute_density(
        adata:AnnData,
        emb='featmap',
        cluster_key='leiden',
        density_key='density',
        quantile_core = 0.5,
        quantile_trans = 0.2,
        
        ):
    """
    Identify the core state and transition state in the embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    emb : str
        The embedding space to plot the density.    
    cluster_key : str
        The key of clusters in adata.obs.
    top_quantile : float 
        The top quantile of the density.            
    """
    
    import scanpy as sc
    # adata.obs['clusters'] = adata.obs['clusters_fine']

    # if there is no clusters in obs, then use leiden
    if cluster_key not in adata.obs.keys():
        cluster_key = 'leiden'
        # Clusters by leiden
        import scanpy as sc
        sc.pp.neighbors(adata, n_neighbors=30,)
        sc.tl.leiden(adata, resolution=0.5)

    partition_label = adata.obs[cluster_key].copy()
    partition_label.value_counts()
    # print(partition_label.value_counts())
    data = adata.obsm[f'X_{emb}'].copy()
    p= kernel_density_estimate(data, data)

    # normalize the density by min-max scaling
    p = (p - np.min(p)) / (np.max(p) - np.min(p))

    adata.obs['density'] = p
    
    # Density in each cluster
    adata.obs[density_key] = np.nan
    leiden_clusters = adata.obs[cluster_key].copy()
    leiden_clusters.value_counts()
    
    for cluster in leiden_clusters.cat.categories.values:
        cluster_in_cluster_label = (leiden_clusters == cluster)
        data_cluster = data[cluster_in_cluster_label, :]
    
        p_1 = kernel_density_estimate(data_cluster, data_cluster)
        # adata.obs['density_separate_cluster'][cluster_in_cluster_label] = p_1
        # normalize the density by min-max scaling
        p_1 = (p_1 - np.min(p_1)) / (np.max(p_1) - np.min(p_1))

        adata.obs.loc[cluster_in_cluster_label, density_key] = p_1
        density = adata.obs[density_key][cluster_in_cluster_label]
    
    # Select top ratio(%) in each cluster as core state
    leiden_clusters = adata.obs[cluster_key].copy()
    
    adata.obs['density_core'] = np.nan
    adata.obs['density_largest'] = np.nan
    adata.obs['density_transition'] = np.nan

    for cluster in leiden_clusters.cat.categories.values:
        cluster_in_cluster_label = (leiden_clusters == cluster)
        density = adata.obs[density_key][cluster_in_cluster_label].copy()
        # density = adata.obs['density'][cluster_in_cluster_label]
        cluster_index = leiden_clusters.index[leiden_clusters == cluster]
        density_sort = density[cluster_index].sort_values(ascending=False)
        if int(len(cluster_index) * (1-quantile_core)) < 50:
            density_sort_topper_index = density_sort.index[:50]
            density_sort_topper_value = density_sort[:50]
        else:
            density_sort_topper_index = density_sort.index[:int(len(cluster_index) * (1-quantile_core))]
            density_sort_topper_value = density_sort[:int(len(cluster_index) * (1-quantile_core))]
        adata.obs.loc[density_sort_topper_index, 'density_core_id'] = cluster
        adata.obs.loc[density_sort_topper_index, 'density_core'] = density_sort_topper_value

        if int(len(cluster_index) * quantile_trans) < 50:
            density_sort_bottom_index = density_sort.index[-50:]
            density_sort_bottom_value = density_sort[-50:]
        else:
            density_sort_bottom_index = density_sort.index[-int(len(cluster_index) * (quantile_trans)):]
            density_sort_bottom_value = density_sort[-int(len(cluster_index) * (quantile_trans)):]
        adata.obs.loc[density_sort_bottom_index, 'density_transition_id'] = cluster
        adata.obs.loc[density_sort_bottom_index, 'density_transition'] = density_sort_bottom_value

        # non-corestate
        # density_sort_rest_index = density_sort.index[int(len(cluster_index) * 0.2):]
        # adata.obs['corestates'][density_sort_rest_index] = f'{cluster} Rest'
        
        density_sort_largest_index = density_sort.index[:1]
        # adata.obs['corestates_largest'][density_sort_largest_index] = cluster
        adata.obs.loc[density_sort_largest_index, 'density_largest'] = cluster
    
    # adata.obs['corestates'] = pd.Series(adata.obs['corestates'].copy(), dtype='category').values
    
    # # Expand the core state by NNs
    # from featuremap.featuremap_ import nearest_neighbors
    # n_neighbors = 15
    # knn_indices, _, _ = nearest_neighbors(adata.obsm[f'X_{emb}'].copy(), n_neighbors=n_neighbors,
    #                                               metric="euclidean", metric_kwds={}, angular=False, random_state=42)

    # # corestates_nn_points coresponding to clusters
    # # initialize an empty dataframe
    # df_temp = pd.DataFrame(index=adata.obs_names, columns=['density_filter'])
    # df_temp['density_filter'] = np.nan
    # for cluster in leiden_clusters.cat.categories.values:
    #     corestates_points = np.where(adata.obs['density_filter'] == cluster)[0]
    #     corestates_nn_points = np.unique(knn_indices[corestates_points].reshape(-1))
    #     df_temp.loc[adata.obs_names[corestates_nn_points], 'density_filter'] = '1'
    #     # adata.obs.loc[adata.obs_names[corestates_nn_points], 'corestates_nn_points'] = cluster
    
    # adata.obs['density_filter'] = df_temp['density_filter']
    # adata.obs['density_filter'] = pd.Categorical(adata.obs['density_filter'], categories=adata.obs[cluster_key].cat.categories, ordered=True)

    sc.pl.embedding(adata, emb, color=['density_core'],)
    sc.pl.embedding(adata, emb, color=['density_transition'],)
 
    # # corestates_nn_points: binary
    # adata.obs['density_filter_binary'] = 0
    # corestates_points = np.where(adata.obs['density_filter'].notna())[0]
    
    # corestates_points = np.unique(corestates_points.reshape(-1))
    # corestates_binary = np.isin(np.array(range(adata.shape[0])), corestates_points) * 1
    # adata.obs['density_filter_points'] = corestates_binary
    
    # adata.obs['core_trans_states'] = '0'
    # corestate_points = np.where(adata.obs['density_filter_points']==1)[0]
    # adata.obs.loc[adata.obs_names[corestate_points],'core_trans_states'] = '1'
    

    # sc.pl.embedding(adata, emb, color=['core_trans_states'])



def compute_density_0(
        adata:AnnData,
        emb='featmap',
        cluster_key='leiden',
        density_key='density',
        quantile_core = 0.5,
        quantile_trans = 0.2,
        
        ):
    """
    Identify the core state and transition state in the embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    emb : str
        The embedding space to plot the density.    
    cluster_key : str
        The key of clusters in adata.obs.
    top_quantile : float 
        The top quantile of the density.            
    """
    
    import scanpy as sc
    # adata.obs['clusters'] = adata.obs['clusters_fine']

    # if there is no clusters in obs, then use leiden
    if cluster_key not in adata.obs.keys():
        cluster_key = 'leiden'
        # Clusters by leiden
        import scanpy as sc
        sc.pp.neighbors(adata, n_neighbors=30,)
        sc.tl.leiden(adata, resolution=0.5)

    partition_label = adata.obs[cluster_key].copy()
    partition_label.value_counts()
    # print(partition_label.value_counts())
    data = adata.obsm[f'X_{emb}'].copy()
    p= kernel_density_estimate(data, data)

    # normalize the density by min-max scaling
    p = (p - np.min(p)) / (np.max(p) - np.min(p))

    adata.obs['density'] = p
    
    # Density in each cluster
    adata.obs[density_key] = np.nan
    leiden_clusters = adata.obs[cluster_key].copy()
    leiden_clusters.value_counts()
    
    for cluster in leiden_clusters.cat.categories.values:
        cluster_in_cluster_label = (leiden_clusters == cluster)
        data_cluster = data[cluster_in_cluster_label, :]
    
        p_1 = kernel_density_estimate(data_cluster, data_cluster)
        # adata.obs['density_separate_cluster'][cluster_in_cluster_label] = p_1
        # normalize the density by min-max scaling
        p_1 = (p_1 - np.min(p_1)) / (np.max(p_1) - np.min(p_1))

        adata.obs.loc[cluster_in_cluster_label, density_key] = p_1

    sc.pl.embedding(adata, emb, color=[density_key,])
    
    density_filter = adata.obs[density_key].copy()
    # get the 0.8 quantile of curvature
    quantile_08 = np.quantile(density_filter, quantile_core)
    # set curvature to Nan for the nodes with curvature less than quantile_08
    density_filter[density_filter<quantile_08] = np.nan
    adata.obs['density_core'] = density_filter

    density_filter = adata.obs[density_key].copy()
    # get the 0.8 quantile of curvature
    quantile_08 = np.quantile(density_filter, quantile_trans)
    # set curvature to Nan for the nodes with curvature less than quantile_08
    density_filter[density_filter>quantile_08] = np.nan
    adata.obs['density_transition'] = density_filter


    sc.pl.embedding(adata, emb, color=['density_core'],)
    sc.pl.embedding(adata, emb, color=['density_transition'],)


########################################################
# Collect trasition state and core state given clusters
##############################################################

def nodes_of_transition_states(adata, start_state, end_state, clusters):
    """
    Collect the nodes of transition states given the start and end state.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    start_state : str
        The start state of the transition.
    end_state : str 
        The end state of the transition.
    clusters : list
        The list of clusters in the data.

    Returns
    -------
    path_nodes : np.array
        The nodes of the path from start to end state.
    path_points_nn : np.array
        The points of the path from start to end state.
    end_bridge_points : np.array
        The points of the end bridge.
    core_points : np.array
        The points of the core states.
    transition_points : np.array
        The points of the transition states.

    
    """

    node_name_start = adata.obs['corestates_largest'][adata.obs['corestates_largest'] == (start_state)].index[0]
    start = np.where(adata.obs_names == node_name_start)[0][0]
    
    node_name_end = adata.obs['corestates_largest'][adata.obs['corestates_largest'] == (end_state)].index[0]
    end = np.where(adata.obs_names == node_name_end)[0][0]
    
    # Spanning tree on embedding space
    ridge_points = np.where(np.array(adata.obs['trajectory_points'])==1)[0]
    corestate_points = np.where(pd.isna((adata.obs['corestates_largest'])) == False)[0]
    # Points for tree
    tree_points = np.union1d(ridge_points, corestate_points)
    mst_subg = mst_subgraph(adata, tree_points, emb='X_featmap')
    mst_subg.clusters().summary()

    start_id = mst_subg.vs.find(name=start).index
    end_id = mst_subg.vs.find(name=end).index
    
    path_given_start_end = mst_subg.get_shortest_paths(v=start_id, to=end_id)
    path_nodes_name = np.array([mst_subg.vs[i]['name'] for i in path_given_start_end])
    
    # Extend the path to both ends in trajectory
    nodes_start_state = np.where(np.array(adata.obs['clusters'] == str(start_state)) == True)[0]
    nodes_start_ridge = ridge_points[np.where(np.in1d(ridge_points, nodes_start_state))[0]]
    
    nodes_end_state = np.where(np.array(adata.obs['clusters'] == str(end_state)) == True)[0]
    nodes_end_ridge = ridge_points[np.where(np.in1d(ridge_points, nodes_end_state))[0]]
    
    node_corestate_start = adata.obs['corestates'][adata.obs['corestates_largest'] == start_state].index
    corestate_start = np.where(np.in1d(adata.obs_names, node_corestate_start))[0]
    
    node_corestate_end = adata.obs['corestates'][adata.obs['corestates_largest'] == end_state].index
    corestate_end = np.where(np.in1d(adata.obs_names, node_corestate_end))[0]
    
    from functools import reduce
    path_nodes = reduce(np.union1d, (path_nodes_name, corestate_start, corestate_end, nodes_start_ridge, nodes_end_ridge))
    
    path_binary = np.isin(np.array(range(adata.shape[0])), path_nodes)
    adata.obs['path_binary'] = (path_binary * 1).astype(int)

    sc.pl.embedding(adata, 'featmap', legend_loc='on data', s=10, color=['path_binary'],cmap='bwr')
    # sc.pl.embedding(adata_var, 'umap_v', legend_loc='on data', s=10, color=['path_binary'])
    
    from featuremap.featuremap_ import nearest_neighbors
    knn_indices, knn_dists, _ = nearest_neighbors(adata.obsm['X_featmap'].copy(), n_neighbors=60,
                                                  metric="euclidean", metric_kwds={}, angular=False, random_state=42)
    path_nodes_nn = np.unique(knn_indices[path_nodes].reshape(-1))
    
    core_nodes = np.array([]).astype(int)
    for cluster in clusters:
        core_nodes = np.append(core_nodes, np.where(adata.obs['corestates'] == str(cluster))[0])
    
    knn_indices, knn_dists, _ = nearest_neighbors(adata.obsm['X_featmap'].copy(), n_neighbors=60,
                                                  metric="euclidean", metric_kwds={}, angular=False, random_state=42)
    core_points = np.unique(knn_indices[core_nodes].reshape(-1))

    path_points_nn = np.union1d(path_nodes_nn, core_points)

    path_points_binary = np.isin(np.array(range(adata.shape[0])), path_points_nn) * 1
    adata.obs['path_points_nn'] = path_points_binary
    sc.pl.embedding(adata, 'featmap', legend_loc='on data', s=10, color=['path_points_nn'],cmap='bwr')    

    end_bridge_nodes = reduce(np.union1d, (path_nodes_name, corestate_start, corestate_end))
    end_bridge_nodes = np.unique(knn_indices[end_bridge_nodes].reshape(-1))
    transition_points = end_bridge_nodes

    end_bridge_points = np.union1d(end_bridge_nodes, core_points)
    # end_bridge_points_binary = np.isin(np.array(range(adata.shape[0])), end_bridge_points) * 1
    # adata.obs['end_bridge_points'] = end_bridge_points_binary
    # sc.pl.embedding(adata, 'featmap', legend_loc='on data', s=10, color=['end_bridge_points'],cmap=cmp('bwr'))    
    
    adata.obs['core_trans_temp'] = np.nan
    adata.obs['core_trans_temp'][end_bridge_points] = '0'
    adata.obs['core_trans_temp'][core_points] = '1'
    sc.pl.embedding(adata, 'featmap', color=['core_trans_temp'])

    
    return path_nodes, path_points_nn, end_bridge_points, core_points, transition_points



# def ridge_estimation(
#         adata:AnnData
#         ):
    
#     data = adata.obsm['X_featmap'].copy()  # Exclude one leiden cluster;
#     # data = adata_var.obsm['X_umap_v']
#     pos_collection = []
#     # for sample_time in range(20):
#     s = Scms(data, 0.5, min_radius=5)
#     p, _, h, msu = s._kernel_density_estimate(data)
#     ifilter_2 =  np.where(p >= (np.max(p)*0.05))[0] # sampling
#     # shifted = np.append(grid_contour[ifilter_1, :],data[ifilter_2, :], axis=0)
#     shifted = data[ifilter_2,:]
#     inverse_sample_index = s.inverse_density_sampling(shifted, n_samples=200, n_jobs=1, batch_size=16)
#     shifted = shifted[inverse_sample_index]
    
#     n_iterations = 200
#     allshiftedx_grid = np.zeros((shifted.shape[0],n_iterations))
#     allshiftedy_grid = np.zeros((shifted.shape[0],n_iterations))
#     for j in range(n_iterations):
#         allshiftedx_grid[:,j] = shifted[:,0]
#         allshiftedy_grid[:,j] = shifted[:,1]
#         shifted += 1*s.scms_update(shifted,method='GradientLogP',stepsize=0.02, relaxation=0.5)[0]
#     pos = np.column_stack([allshiftedx_grid[:,-1], allshiftedy_grid[:,-1]])
#     pos_collection.append(pos)
#     pos = np.array(pos_collection).reshape(-1,2)
#     p_pos, _, _, _ = s._kernel_density_estimate(pos)
#     pos_filter_idx =  np.where(p_pos >= (np.max(p_pos)*0.1))[0] # sampling
#     pos_filter = pos[pos_filter_idx]
    
#     # Plot the ridge
#     s = Scms(data, 0.5, min_radius=5)
#     min_x = min(data[:, 0])
#     max_x = max(data[:, 0])
#     min_y = min(data[:, 1])
#     max_y = max(data[:, 1])
#     # part = 200
#     num_grid_point = data.shape[0] * 0.5
#     x_range = max_x - min_x
#     y_range = max_y - min_y
#     # x_range = 1 - 0.618
#     # y_range = 0.618
#     part_y = np.sqrt(num_grid_point / x_range * y_range)
#     part_x = x_range / y_range * part_y
#     # Assign num of grid points mort to vertical direction ??
#     xv, yv = np.meshgrid(np.linspace(min_x, max_x, round(part_x)), np.linspace(min_y, max_y, round(part_y)),
#                          sparse=False, indexing='ij')
#     grid_contour = np.column_stack([np.concatenate(xv), np.concatenate(yv)])
#     p1, g1, h1, msu = s._kernel_density_estimate(grid_contour, output_onlylogp=False, )
    
#     plt.contourf(xv, yv, p1.reshape(
#         round(part_x), round(part_y)), levels=20, cmap='Blues')
#     plt.scatter(data[:,0],data[:,1], s=1, c='darkgrey', alpha=0.1)
#     plt.scatter(pos_filter[:,0],pos_filter[:,1],c="red", s=1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#     plt.clf()
    
# from scipy.sparse.csgraph import shortest_path, dijkstra
def mst_subgraph(adata, tree_points, emb='X_featmap'):
    """
    Construct the minimum spanning tree over the tree points.

    Parameters
    ----------
    adata
    tree_points : np.array
        Points included in the induced subgraph

    Returns
    -------
    mst_subg : igraph
        minimum spanning_tree over tree_points (anchors).

    """
    # # M = adata.obsp['emb_dists'].copy().toarray() 
    # M = adata_var.obsm['knn_dists'].copy().toarray()

    # graph = csr_matrix(M) # knn graph
    # dist_matrix, predecessors = dijkstra(
    #     csgraph=graph, directed=False, return_predecessors=True)

    # dist_mat = dist_matrix
    # g = sc._utils.get_igraph_from_adjacency(dist_mat) # Complete graph from pairwise distance
    # g.vs["name"] = range(M.shape[0])  # 'name' to store original point id
    
    # g_induced_subg = g.induced_subgraph(tree_points)
    # mst_subg = g_induced_subg.spanning_tree(weights=g_induced_subg.es["weight"])
    
    n_neighbors = 60
    knn_indices, knn_dists, _ = nearest_neighbors(adata.obsm[emb][tree_points].copy(), n_neighbors=n_neighbors,
                                                  metric="euclidean", metric_kwds={}, angular=False, random_state=42)

    # Pairwise distance by knn indices and knn distances
    dist_mat = np.zeros([tree_points.shape[0], tree_points.shape[0]])
    for i in range(tree_points.shape[0]):
        for j in range(n_neighbors):
            dist_mat[i, knn_indices[i,j]] += knn_dists[i,j]

    # knn graph by iGraph
    g = sc._utils.get_igraph_from_adjacency(dist_mat) # Complete graph from pairwise distance
    g.vs["name"] = tree_points  # 'name' to store original point id
    # g_induced_subg = g.induced_subgraph(tree_points)
    mst_subg = g.spanning_tree(weights=g.es["weight"])
    return mst_subg


def ridge_pseudotime(adata, root, plot='featmap'):
    """
    Compute the pseudotime along the ridge path.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    root : str
        The root of the ridge path.
    plot : str  
        The embedding space to plot the pseudotime.
    Returns
    -------
    adata.obs['ridge_pseudotime'] : np.array
        The pseudotime along the ridge path.
            
    """
    from scipy.special import expit
    from sklearn.preprocessing import scale

    
    # Construct mst subgraph
    ridge_points = np.where(np.array(adata.obs['trajectory_points'])==1)[0]
    corestate_points = np.where(pd.isna((adata.obs['corestates_largest'])) == False)[0]
    tree_points = np.union1d(ridge_points, corestate_points)

    mst_subg = mst_subgraph(adata, tree_points, emb='X_featmap')

    farthest_points = mst_subg.farthest_points() # (34, 174, 140)
    farthest_points = np.array(farthest_points[:2])
    farthest_path = mst_subg.get_shortest_paths(v=farthest_points[0], to=farthest_points[1])
    farthest_path_name = np.array([mst_subg.vs[i]['name'] for i in farthest_path])
    farthest_path_binary = np.isin(np.array(range(adata.shape[0])), farthest_path_name)
    adata.obs['farthest_path'] = (farthest_path_binary * 1).astype(int)
    sc.pl.embedding(adata, plot, legend_loc='on data', s=100, color=['farthest_path','trajectory_points'])
    # sc.pl.embedding(adata, 'featmap', color=['leiden','corestates','farthest_path','trajectory_points'])
    
    # Set the starting point
    if root is None:
        start = farthest_points[0]
    else:
        # root_index = adata.obs['corestates_largest'][adata.obs['corestates_largest'] == root].index[0]
        # root_id = np.where(adata.obs_names == root_index)[0][0]
        start = np.where(mst_subg.vs['name'] == root)[0][0]
    # start = start
    dist_from_start = mst_subg.shortest_paths(start, weights="weight")
    nodes_in_tree = np.array([mst_subg.vs[i]['name'] for i in range(mst_subg.vcount())])
    dist_from_start_dict = dict(zip(nodes_in_tree, dist_from_start[0]))
    

    # Pairwise shortest path of origninal knn graph
    # M = adata.obsp['emb_dists'].toarray()
    # M = adata.obsp['knn_dists'].toarray()
    
    from umap.umap_ import fuzzy_simplicial_set
    _, _, _, knn_dists = fuzzy_simplicial_set(
        adata.obsm['X_featmap'] ,
        n_neighbors=60,
        random_state=42,
        metric="euclidean",
        metric_kwds={},
        # knn_indices,
        # knn_dists,
        verbose=True,
        return_dists=True)
    
    M = knn_dists.toarray()


    graph = csr_matrix(M)
    
    dist_matrix, predecessors = shortest_path(
        csgraph=graph, directed=False, indices=tree_points,return_predecessors=True)
    # For each node, find its nearest node in the tree
    dist_matrix = dist_matrix.T
    
    nearest_in_tree = np.argmin(dist_matrix, axis=1)
    nearest_in_tree_dist = np.min(dist_matrix, axis=1)
    data_dist = {'node_in_tree': tree_points[nearest_in_tree],
                 'dist': nearest_in_tree_dist}
    nearest_node_in_tree = pd.DataFrame.from_dict(data_dist,orient='columns')
    
    # For each node, compute the dist to start by first identifying its nearest node in the tree, then to start point
    emb_pseudotime = np.array([nearest_node_in_tree.at[i,'dist'] + 
              dist_from_start_dict[nearest_node_in_tree.at[i,'node_in_tree']]
              for i in range(dist_matrix.shape[0])
              ])
    
    emb_pseudotime[np.where(emb_pseudotime == np.inf)[0]] = 20
    
    adata.obs['ridge_pseudotime'] = expit(scale(emb_pseudotime))
    # adata.obs['emb_pseudotime'] = emb_pseudotime
    
    # root_idx = mst_s1ubg.vs[start]['name']
    # adata.uns["iroot"] = root_idx
    # sc.tl.dpt(adata)
    # adata.obs['dpt_pseudotime'] = expit(scale(adata.obs['dpt_pseudotime'])+1)
    # expit(scale(emb_pseudotime))
    sc.pl.embedding(adata, plot, legend_loc='on data', color=['ridge_pseudotime',])
    # sc.pl.embedding(adata, 'umap', legend_loc='on data', color=['emb_pseudotime',])

    return adata.obs['ridge_pseudotime']


def bifurcation_plot(adata, core_states, transition_states_1, transition_states_2):
    """
    Plot the bifurcation states in the embedding space.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    core_states : list
        The list of core states.
    transition_states_1 : list  
        The list of transition states 1.
    transition_states_2 : list
        The list of transition states 2.


    """

    core_states_map = {str(i):'core' for i in core_states}
    transition_states_map_1 = {str(i):'transition_1' for i in transition_states_1}
    transition_states_map_2 = {str(i):'transition_2' for i in transition_states_2}

    # merge the core states and transition states
    core_trans_states_bifur = {**core_states_map, **transition_states_map_1, **transition_states_map_2}

    adata.obs['core_trans_states_bifur'] = adata.obs['leiden_v'].map(core_trans_states_bifur)
    adata.obs['core_trans_states_bifur']  = adata.obs['core_trans_states_bifur'].astype('category')

    adata.obs['core_trans_states_bifur']  = adata.obs['core_trans_states_bifur'].cat.set_categories([ 'transition_1','core', 'transition_2'], ordered=True)

    sc.pl.embedding(adata, 'featmap_v',legend_fontsize=10, s=10, color=['core_trans_states_bifur'])

def bifurcation_plot_1(adata, core_states, transition_states_1, transition_states_2, transition_states_3):
    """
    Plot the bifurcation states in the embedding space.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    core_states : list
        The list of core states.
    transition_states_1 : list  
        The list of transition states 1.
    transition_states_2 : list
        The list of transition states 2.


    """

    core_states_map = {str(i):'core' for i in core_states}
    transition_states_map_1 = {str(i):'transition_1' for i in transition_states_1}
    transition_states_map_2 = {str(i):'transition_2' for i in transition_states_2}
    transition_states_map_3 = {str(i):'transition_3' for i in transition_states_3}

    # merge the core states and transition states
    core_trans_states_bifur = {**core_states_map, **transition_states_map_1, **transition_states_map_2, **transition_states_map_3}

    adata.obs['core_trans_states_bifur'] = adata.obs['leiden_v'].map(core_trans_states_bifur)
    adata.obs['core_trans_states_bifur']  = adata.obs['core_trans_states_bifur'].astype('category')

    adata.obs['core_trans_states_bifur']  = adata.obs['core_trans_states_bifur'].cat.set_categories([ 'transition_1','core', 'transition_2', 'transition_3'], ordered=True)

    sc.pl.embedding(adata, 'featmap_v',legend_fontsize=10, s=10, color=['core_trans_states_bifur'])

def path_plot(adata, core_states, transition_states):
    """
    Plot the path states in the embedding space.    

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.  
    core_states : list
        The list of core states.
    transition_states : list
        The list of transition states.

    
    """
    core_states_map = {str(i):'core' for i in core_states}
    transition_states_map = {str(i):'transition' for i in transition_states}

    # merge the core states and transition states
    path_state = {**core_states_map, **transition_states_map}

    adata.obs['path_states'] = adata.obs['leiden_v'].map(path_state)
    adata.obs['path_states']  = adata.obs['path_states'].astype('category')

    adata.obs['path_states']  = adata.obs['path_states'].cat.set_categories(['transition', 'core'], ordered=True)


    sc.pl.embedding(adata, 'featmap_v',legend_fontsize=10, s=30, color=['path_states'], palette='tab10') 


############################################
# Density vs pseudotime
############################################
#%%
def plot_density_pseudotime(filtered_data, pseudotime='feat_pseudotime', clusters='clusters', density='density'):
    """
    Plot the density vs pseudotime.

    Parameters
    ----------  
    filtered_data : pd.DataFrame
        The dataframe including the data.
    pseudotime : str
        The pseudotime in the data.
    clusters : str
        The clusters in the data.
    density : str
        The density in the data.


    """
    from pygam import LinearGAM
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = filtered_data[pseudotime].values
    y = filtered_data[density].values
    gam = LinearGAM(n_splines=20).fit(X, y)
    
    fig, ax = plt.subplots()
    XX = gam.generate_X_grid(term=0, n=100)
   
    for response in gam.sample(X, y, quantity='y', n_draws=50, sample_at_X=XX):
        plt.scatter(XX, response, alpha=.01, color='k')
    plt.plot(XX, gam.predict(XX), 'r--')
    plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')

    ax.plot(XX, gam.predict(XX), 'b--', label='_nolegend_')
    sns.scatterplot(x=pseudotime, y=density, data=filtered_data, hue=clusters, ax=ax)
    

    ax.set_xlabel(pseudotime)
    # ax.set_ylabel('')
    ax.set_ylabel(density)
    ax.legend().remove()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(f'./figures/pancreas/density_pseudotime_{pseudotime}_beta.png', bbox_inches='tight')
    plt.show()


def compute_betweenness_centrality(adata, emb_featuremap, quantile_trans=0.8, quantile_core=0.2):
    """
    Compute the betweenness centrality of the cell network.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    emb_featuremap : object
        The featuremap object.
    top_quantile : float
        The top quantile of the betweenness centrality.
            
    """
    graph = emb_featuremap._featuremap_kwds['graph_v']
    # graph =emb_featuremap.graph_

    # create a graph from the adjacency matrix
    G = nx.from_numpy_matrix(graph)
    # graph size of G
    n = G.number_of_nodes()

    # compute the betweenness centrality 
    betweenness = nx.betweenness_centrality(G, k=int(n*0.2))

    # adata.obs['betweenness'] = [betweenness[i] for i in range(len(betweenness))]

    # extract the degree of each node
    degree = G.degree()
    adata.obs['degree'] = [degree[i] for i in range(len(degree))]

    # normalize the betweenness centrality by node degree
    betweenness_centrality = [betweenness[i]/degree[i] for i in range(len(betweenness))]

    # set nan values to 0
    # betweenness_centrality = np.nan_to_num(betweenness_centrality)

    # log normalization
    betweenness_centrality = np.array(betweenness_centrality) * 10e4
    betweenness_centrality = np.log(betweenness_centrality + 1)

    # normalize the betweenness centrality by min-max scaling
    betweenness_centrality = (betweenness_centrality - np.min(betweenness_centrality)) / (np.max(betweenness_centrality) - np.min(betweenness_centrality))
    adata.obs['betweenness_centrality'] = betweenness_centrality

    betweenness_centrality_filter =  betweenness_centrality.copy()  
    # get the 0.8 quantile of betweenness_normalized
    quantile = np.quantile(betweenness_centrality_filter, quantile_trans)
    # set betweenness_normalized to Nan for the nodes with betweenness_normalized less than quantile
    betweenness_centrality_filter[betweenness_centrality_filter<quantile] = np.nan
    adata.obs['betweenness_centrality_transition'] = betweenness_centrality_filter

    betweenness_centrality_filter =  betweenness_centrality.copy()  
    # get the 0.8 quantile of betweenness_normalized
    quantile = np.quantile(betweenness_centrality_filter, quantile_core)
    # set betweenness_normalized to Nan for the nodes with betweenness_normalized less than quantile
    betweenness_centrality_filter[betweenness_centrality_filter>quantile] = np.nan
    adata.obs['betweenness_centrality_core'] = betweenness_centrality_filter

    sc.pl.embedding(adata, basis='featmap_v', color='betweenness_centrality', cmap='viridis', size=10)
    sc.pl.embedding(adata, basis='featmap_v', color='betweenness_centrality_core', cmap='viridis', size=10)
    sc.pl.embedding(adata, basis='featmap_v', color='betweenness_centrality_transition', cmap='viridis', size=10)

        



def compute_curvature(adata, emb_featuremap, quantile_core=0.2, quantile_trans=0.8):

    """
    Compute the curvature of the embedding space.

    k = ||d T / d s||

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    emb_featuremap : object
        The featuremap object.
    top_quantile : float
        The top quantile of the curvature.

    """

    # data = adata.X.copy()
    data = adata.obsm['variation_pc']

    knn_indices = emb_featuremap._featuremap_kwds['_knn_indices']

    # each data point, get the average distance to its neighbors
    from sklearn.metrics import pairwise_distances
    k = knn_indices.shape[1]
    delta_dist = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        indices = knn_indices[i]
        delta_dist[i] = np.mean(pairwise_distances(data[i].reshape(1, -1), data[indices], metric='euclidean'))

    # curvature = 1 / delta_dist
    curvature = delta_dist

    # normalize the curvature by min-max scaling
    curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))
    adata.obs['curvature'] = curvature
    
    curvature_filter =  curvature.copy()
    # get the 0.8 quantile of curvature
    quantile_core_val = np.quantile(curvature_filter, quantile_core)
    # set curvature to Nan for the nodes with curvature less than quantile_08
    curvature_filter[curvature_filter>quantile_core_val] = np.nan
    adata.obs['curvature_core'] = curvature_filter

    curvature_filter =  curvature.copy()
    # get the 0.8 quantile of curvature
    quantile_trans_val = np.quantile(curvature_filter, quantile_trans)
    # set curvature to Nan for the nodes with curvature less than quantile_08
    curvature_filter[curvature_filter<quantile_trans_val] = np.nan
    adata.obs['curvature_transition'] = curvature_filter

 

   
    # print(adata.obs['curvature'])
    # print(f'curvature_filter: {curvature_filter}')

    sc.pl.embedding(adata, basis='featmap_v', color='curvature', cmap='viridis', size=10)
    sc.pl.embedding(adata, basis='featmap_v', color='curvature_core', cmap='viridis', size=10)
    sc.pl.embedding(adata, basis='featmap_v', color='curvature_transition', cmap='viridis', size=10)




def plot_core_transition_states(adata):
    """
    Given the core and transition states defined by the density, curvature and betweenness centrality, 
    union the core and transition states and
    plot the core and transition states in the embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    """
    density_core = adata.obs['density_core'].values.astype(float)
    curvature_core = adata.obs['curvature_core'].values
    betweenness_centrality_core = adata.obs['betweenness_centrality_core'].values

    density_transition = adata.obs['density_transition'].values
    curvature_transition = adata.obs['curvature_transition'].values
    betweenness_centrality_transition = adata.obs['betweenness_centrality_transition'].values

    # set above to binary of not nan
    density_core = ~np.isnan(density_core)
    curvature_core = ~np.isnan(curvature_core)
    betweenness_centrality_core = ~np.isnan(betweenness_centrality_core)

    density_transition = ~np.isnan(density_transition)
    curvature_transition = ~np.isnan(curvature_transition)
    betweenness_centrality_transition = ~np.isnan(betweenness_centrality_transition)

    # core states
    core_states = density_core | curvature_core | betweenness_centrality_core
    # transition states
    transition_states = density_transition | curvature_transition | betweenness_centrality_transition

    # intersection of core and transition states
    core_transition_overlap = core_states & transition_states

    # Exclude the overlap
    core_states[core_transition_overlap] = False
    transition_states[core_transition_overlap] = False

    core_states = core_states.astype(int).astype(str)
    transition_states = transition_states.astype(int).astype(str)

    # Replace '0' with NaN
    core_states_nan = [np.nan if i == '0' else i for i in core_states]
    transition_states_nan = [np.nan if i == '0' else i for i in transition_states]

    # Extract the first two colours from the tab10 colormap
    tab10_colors = [plt.cm.tab10(1), plt.cm.tab10(0)]
    # Convert to a format suitable for seaborn or other libraries
    color_palette = [tuple(color[:3]) for color in tab10_colors]  # Drop alpha if necessary

    adata.obs['core_states'] = core_states_nan
    sc.pl.embedding(adata, basis='X_featmap_v', color='core_states', palette=color_palette, s=10, frameon=False)

    # Extract the first two colours from the tab10 colormap
    tab10_colors = [plt.cm.tab10(0), plt.cm.tab10(1)]
    # Convert to a format suitable for seaborn or other libraries
    color_palette = [tuple(color[:3]) for color in tab10_colors]  # Drop alpha if necessary

    adata.obs['transition_states'] = transition_states_nan
    sc.pl.embedding(adata, basis='X_featmap_v', color='transition_states', palette=color_palette, s=10, frameon=False)



def compute_cluster_state_labels(adata):
    """
    Compute the cluster state labels based on the percentage of core_states and transition_states for each cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
        
    """
    leiden_v_clusters = adata.obs['leiden_v'].cat.categories

    # For each cluster, compute the percentage of core_states and transition_states, respectively.
    # The cluster is labeled as core_states if the percentage of core_states is larger than 0.5, otherwise, it is labeled as transition_states.
    cluster_state_label = []
    cluster_state_dict = {}

    # Save the percentage of core_states and transition_states for each cluster
    core_states_percentage_dict = {}
    transition_states_percentage_dict = {}

    for cluster in leiden_v_clusters:
        cluster_index = np.where(adata.obs['leiden_v'] == cluster)[0]
        core_states_percentage = np.sum(adata.obs['core_states'].values[cluster_index] == '1') / len(cluster_index)
        transition_states_percentage = np.sum(adata.obs['transition_states'].values[cluster_index] == '1') / len(cluster_index)

        core_states_percentage_dict[cluster] = core_states_percentage
        transition_states_percentage_dict[cluster] = transition_states_percentage

        if core_states_percentage > transition_states_percentage:
            cluster_state_label.append('core_states')
            cluster_state_dict[cluster] = 'core_states'
        else:
            cluster_state_label.append('transition_states')
            cluster_state_dict[cluster] = 'transition_states'

    # assign the cluster label to the original adata
    adata.obs['cluster_state_label'] = adata.obs['leiden_v'].map(dict(zip(leiden_v_clusters, cluster_state_label)))
    adata.obs['cluster_state_label'] = adata.obs['cluster_state_label'].astype('category')
    adata.obs['cluster_state_label'] = adata.obs['cluster_state_label'].cat.reorder_categories(['transition_states', 'core_states'])

    sc.pl.embedding(adata, basis='FeatureMAP_v', color='cluster_state_label', cmap='viridis', )

    adata.uns['cluster_state_dict'] = cluster_state_dict

    # adata_var.obs['cluster_state_label'] = adata.obs['cluster_state_label']

    # plot the percentage of core_states and transition_states for each cluster 
    import pandas as pd
    core_states_percentage_df = pd.DataFrame.from_dict(core_states_percentage_dict, orient='index', columns=['core_states_percentage'])
    transition_states_percentage_df = pd.DataFrame.from_dict(transition_states_percentage_dict, orient='index', columns=['transition_states_percentage'])

    percentage_df = pd.concat([transition_states_percentage_df, core_states_percentage_df], axis=1)
    percentage_df = percentage_df.sort_values(by='transition_states_percentage', ascending=False)

    percentage_df.plot(kind='bar', stacked=True, figsize=(10, 8))
    plt.xlabel('Clusters by Leiden')
    plt.ylabel('Percentage')
    plt.legend().set_visible(False)
    plt.show()

    # return adata


import networkx as nx
import leidenalg as la
import numpy as np
import igraph as ig



def weighted_clustering_coefficient(G, node):
    neighbors = list(G.neighbors(node))
    k_i = len(neighbors)
    
    # If the node has fewer than 2 neighbors, its clustering coefficient is 0
    if k_i < 2:
        return 0.0
    
    # Sum of the weights of edges connected to the node (s_i)
    s_i = sum([G[node][neighbor].get('weight', 1.0) for neighbor in neighbors])
    
    # Initialize numerator for the clustering coefficient
    numerator = 0.0
    
    # Iterate through pairs of neighbors
    for j in neighbors:
        for k in neighbors:
            if j != k and G.has_edge(j, k):
                # Edge weights between node and its neighbors j and k
                w_ij = G[node][j].get('weight', 1.0)
                w_ik = G[node][k].get('weight', 1.0)
                # Weight of the edge between j and k
                a_jk = G[j][k].get('weight', 1.0) if G.has_edge(j, k) else 0.0
                # Apply the formula from Barrat et al.
                numerator += (w_ij + w_ik) / 2 * a_jk
    
    # Compute the denominator
    denominator = s_i * (k_i - 1)
    
    # If the denominator is 0, return 0
    if denominator == 0:
        return 0.0
    
    # Return the clustering coefficient
    return numerator / denominator


def clustering_coefficient(G, node):
    neighbors = list(G.neighbors(node))
    k_i = len(neighbors)
    
    # If the node has fewer than 2 neighbors, its clustering coefficient is 0
    if k_i < 2:
        return 0.0
    
    # Initialize numerator for the clustering coefficient
    numerator = 0
    
    # Iterate through pairs of neighbors
    for j in neighbors:
        for k in neighbors:
            if j != k and G.has_edge(j, k):
                numerator += 1
    
    # Since each edge is counted twice, divide by 2
    numerator /= 2
    
    # Compute the denominator
    denominator = k_i * (k_i - 1) / 2
    
    # If the denominator is 0, return 0
    if denominator == 0:
        return 0.0
    
    # Return the clustering coefficient
    return numerator / denominator


# Function to compute the weighted clustering coefficient for each cluster
def clustering_coefficient_by_cluster(G, clusters):
    cluster_coefficients = {}
    
    for cluster_id in np.unique(clusters):
        nodes_in_cluster = [node for node, cluster in enumerate(clusters) if cluster == cluster_id]
        if len(nodes_in_cluster) == 0:
            continue
        
        # Randomly sample 35% nodes from the cluster for efficiency
        sample_size = int(0.35 * len(nodes_in_cluster))
        sample_nodes = np.random.choice(nodes_in_cluster, size=sample_size, replace=False)
        
        # Calculate the weighted clustering coefficient for each node in the cluster
        cluster_coeffs = [
            clustering_coefficient(G, node) for node in sample_nodes
        ]
        
        # Average clustering coefficient for the cluster
        cluster_coefficients[cluster_id] = np.mean(cluster_coeffs)
    
    return cluster_coefficients



def silhouette_score_one_point(distances, labels, point_index):
    """
    Compute the silhouette score for one node using a shortest path distance matrix.
    
    Parameters:
    - distances: 2D list or ndarray of shape (n_nodes, n_nodes), shortest path distances between nodes
    - labels: list or ndarray of shape (n_nodes,)
    - point_index: int, index of the node for which to compute the silhouette score
    
    Returns:
    - silhouette score for the given node
    """
    point_label = labels[point_index]

    # Compute a(i): Mean intra-cluster distance
    same_cluster_nodes = [idx for idx, label in enumerate(labels) if label == point_label]
    n_same_cluster = len(same_cluster_nodes)

    if n_same_cluster <= 1:
        return 0  # If the node is the only one in its cluster, s(i) = 0
    
    intra_distances = [distances[point_index][other_node] 
                       for other_node in same_cluster_nodes if other_node != point_index]
    a_i = sum(intra_distances) / (n_same_cluster - 1)

    # Compute b(i): Minimum mean distance to any other cluster
    other_cluster_labels = set(labels) - {point_label}
    b_i = float('inf')

    for other_label in other_cluster_labels:
        other_cluster_nodes = [idx for idx, label in enumerate(labels) if label == other_label]
        inter_distances = [distances[point_index][other_node] for other_node in other_cluster_nodes]
        mean_inter_distance = sum(inter_distances) / len(other_cluster_nodes)
        b_i = min(b_i, mean_inter_distance)

    # Compute silhouette score for the given node
    s_i = (b_i - a_i) / max(a_i, b_i)

    return s_i

    

import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def compute_and_plot_clustering_coefficients(adata, emb_featuremap, emb_featuremap_v, phate_graph_nx, tsne_graph, densmap_graph):

    G_expr = nx.from_numpy_matrix(emb_featuremap.graph_)
    G_var = nx.from_numpy_matrix(emb_featuremap_v._featuremap_kwds['graph_v'])
    G_phate = phate_graph_nx
    G_tsne = nx.from_numpy_matrix(tsne_graph)
    G_densmap = nx.from_numpy_matrix(densmap_graph)

    clusters = adata.obs['leiden_v'].values.tolist()
    sc.pl.embedding(adata, basis='X_featmap_v', color='leiden_v', legend_loc='on data')

    print('Computing the weighted clustering coefficients...')

    cluster_coefficients_var = clustering_coefficient_by_cluster(G_var, clusters)
    cluster_coefficients_expr = clustering_coefficient_by_cluster(G_expr, clusters)
    cluster_coefficients_phate = clustering_coefficient_by_cluster(G_phate, clusters)
    cluster_coefficients_tsne = clustering_coefficient_by_cluster(G_tsne, clusters)
    cluster_coefficients_densmap = clustering_coefficient_by_cluster(G_densmap, clusters)

    df_var = pd.DataFrame(
        {"Cluster": list(cluster_coefficients_var.keys()), "Coefficient": list(cluster_coefficients_var.values())}
    )
    df_var['type'] = 'variation_graph'

    df_expr = pd.DataFrame(
        {"Cluster": list(cluster_coefficients_expr.keys()), "Coefficient": list(cluster_coefficients_expr.values())}
    )
    df_expr['type'] = 'expression_graph'

    df_phate = pd.DataFrame(
        {"Cluster": list(cluster_coefficients_phate.keys()), "Coefficient": list(cluster_coefficients_phate.values())}
    )
    df_phate['type'] = 'phate_graph'

    df_tsne = pd.DataFrame(
        {"Cluster": list(cluster_coefficients_tsne.keys()), "Coefficient": list(cluster_coefficients_tsne.values())}
    )
    df_tsne['type'] = 'tsne_graph'

    df_densmap = pd.DataFrame(
        {"Cluster": list(cluster_coefficients_densmap.keys()), "Coefficient": list(cluster_coefficients_densmap.values())}
    )
    df_densmap['type'] = 'densmap_graph'

    df = pd.concat([df_var, df_expr, df_phate, df_tsne, df_densmap], axis=0)

    plt.figure(figsize=(4, 3), dpi=100)
    plt.rcParams.update({'font.size': 12})
    fontsize = 12

    cluster_state_dict = adata.uns['cluster_state_dict']
    transition_states = [cluster for cluster, state in cluster_state_dict.items() if state == 'transition_states']
    core_states = [cluster for cluster, state in cluster_state_dict.items() if state == 'core_states']

    coeff_core_var = []
    coeff_core_expr = []
    coeff_core_phate = []
    coeff_core_tsne = []
    coeff_core_densmap = []

    for cluster in core_states:
        coeff_core_var.append(cluster_coefficients_var[cluster])
        coeff_core_expr.append(cluster_coefficients_expr[cluster])
        coeff_core_phate.append(cluster_coefficients_phate[cluster])
        coeff_core_tsne.append(cluster_coefficients_tsne[cluster])
        coeff_core_densmap.append(cluster_coefficients_densmap[cluster])

    f_stat, p_val = f_oneway(coeff_core_tsne, coeff_core_expr, coeff_core_densmap, coeff_core_phate, coeff_core_var)
    print(f'ANOVA: f_stat: {f_stat}, p_val: {p_val}')

    plt.figure(figsize=(4, 2))
    box = sns.boxplot(data=[coeff_core_tsne, coeff_core_expr, coeff_core_densmap, coeff_core_phate, coeff_core_var], color=plt.get_cmap('tab10')(1), fill=False, width=0.5, showfliers=False)
    hatch_patterns = ['/', '\\', 'x', '/', '/', '/']
    for patch, hatch in zip(box.patches, hatch_patterns):
        patch.set_hatch(hatch)
    sns.stripplot(data=[coeff_core_tsne, coeff_core_expr, coeff_core_densmap, coeff_core_phate, coeff_core_var], color=plt.get_cmap('tab10')(1), jitter=True, dodge=True)
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['t-SNE', 'UMAP', 'densMAP', 'PHATE', 'FeatureMAP'], fontsize=fontsize, rotation=15)
    plt.title("Clustering coefficients in core states", fontsize=fontsize)
    plt.ylabel("Clustering coefficient", fontsize=fontsize)
    plt.text(0.1, 0.9, f"ANOVA F-stat: {f_stat:.2e}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=fontsize)
    plt.text(0.1, 0.8, f"ANOVA p-value: {p_val:.2e}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=fontsize)
    plt.show()

    coeff_trans_var = []
    coeff_trans_expr = []
    coeff_trans_phate = []
    coeff_trans_tsne = []
    coeff_trans_densemap = []

    for cluster in transition_states:
        coeff_trans_expr.append(cluster_coefficients_expr[cluster])
        coeff_trans_phate.append(cluster_coefficients_phate[cluster])
        coeff_trans_var.append(cluster_coefficients_var[cluster])
        coeff_trans_tsne.append(cluster_coefficients_tsne[cluster])
        coeff_trans_densemap.append(cluster_coefficients_densmap[cluster])

    f_stat, p_val = f_oneway(coeff_trans_expr, coeff_trans_phate, coeff_trans_var, coeff_trans_tsne, coeff_trans_densemap)
    print(f'ANOVA: f_stat: {f_stat}, p_val: {p_val}')

    plt.figure(figsize=(4, 2))
    box = sns.boxplot(data=[coeff_trans_tsne, coeff_trans_expr, coeff_trans_densemap, coeff_trans_phate, coeff_trans_var], color=plt.get_cmap('tab10')(0), fill=False, width=0.5, showfliers=False)
    hatch_patterns = ['/', '\\', 'x', '/', '\\', 'x']
    for patch, hatch in zip(box.patches, hatch_patterns):
        patch.set_hatch(hatch)
    sns.stripplot(data=[coeff_trans_tsne, coeff_trans_expr, coeff_trans_densemap, coeff_trans_phate, coeff_trans_var], color=plt.get_cmap('tab10')(0), jitter=True, dodge=True)
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['t-SNE', 'UMAP', 'densMAP', 'PHATE', 'FeatureMAP'], fontsize=fontsize, rotation=15)
    plt.title("Clustering coefficients in transition states", fontsize=fontsize)
    plt.ylabel("Clustering coefficient", fontsize=fontsize)
    plt.text(0.1, 0.95, f"ANOVA F-stat: {f_stat:.2e}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=fontsize)
    plt.text(0.1, 0.85, f"ANOVA p-value: {p_val:.2e}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=fontsize)
    plt.show()




import networkx as nx

from sklearn.metrics import pairwise_distances
from scipy.stats import ttest_ind, f_oneway
import seaborn as sns
import matplotlib.pyplot as plt

def compute_and_plot_silhouette_scores(adata, emb_featuremap, emb_featuremap_v, phate_graph_nx, tsne_graph, densmap_graph):
    # Plot embedding
    sc.pl.embedding(adata, basis='X_featmap_v', color='leiden_v', cmap='Blues_r', s=10, legend_loc='on data', title='Leiden_v clustering')

    # Graph distance matrices   
    G_expr = nx.from_numpy_matrix(emb_featuremap.graph_)
    G_var = nx.from_numpy_matrix(emb_featuremap_v._featuremap_kwds['graph_v'])
    G_phate = phate_graph_nx 
    G_tsne = nx.from_numpy_matrix(tsne_graph)
    G_densmap = nx.from_numpy_matrix(densmap_graph)

    # Get the weighted adjacency matrix
    adjacency_expr = nx.to_numpy_array(G_expr)
    adjacency_var = nx.to_numpy_array(G_var)
    adjacency_phate = nx.to_numpy_array(G_phate)
    adjacency_tsne = nx.to_numpy_array(G_tsne)
    adjacency_densmap = nx.to_numpy_array(G_densmap)

    # Compute the distance matrices
    dist_mat_expr = pairwise_distances(adjacency_expr, metric='euclidean')
    dist_mat_var = pairwise_distances(adjacency_var, metric='euclidean')
    dist_mat_phate = pairwise_distances(adjacency_phate, metric='euclidean')
    dist_mat_tsne = pairwise_distances(adjacency_tsne, metric='euclidean')
    dist_mat_densmap = pairwise_distances(adjacency_densmap, metric='euclidean')

    labels = adata.obs['leiden_v'].values.tolist()
    labels = np.array(labels)
    clusters = np.unique(labels)

    ss_clusters_expr = {}
    ss_clusters_var = {}
    ss_clusters_phate = {}
    ss_clusters_tsne = {}
    ss_clusters_densmap = {}

    for cluster in clusters:
        print(f'Cluster {cluster}')
        cluster_indices = np.where(labels == cluster)[0]
        ss_cluster_expr = []
        ss_cluster_var = []
        ss_cluster_phate = []
        ss_cluster_tsne = []
        ss_cluster_densmap = []

        for idx in cluster_indices:
            ss_cluster_expr.append(silhouette_score_one_point(dist_mat_expr, labels, idx))
            ss_cluster_var.append(silhouette_score_one_point(dist_mat_var, labels, idx))
            ss_cluster_phate.append(silhouette_score_one_point(dist_mat_phate, labels, idx))
            ss_cluster_tsne.append(silhouette_score_one_point(dist_mat_tsne, labels, idx))
            ss_cluster_densmap.append(silhouette_score_one_point(dist_mat_densmap, labels, idx))

        ss_clusters_expr[cluster] = sum(ss_cluster_expr) / len(ss_cluster_expr)
        ss_clusters_var[cluster] = sum(ss_cluster_var) / len(ss_cluster_var)
        ss_clusters_phate[cluster] = sum(ss_cluster_phate) / len(ss_cluster_phate)
        ss_clusters_tsne[cluster] = sum(ss_cluster_tsne) / len(ss_cluster_tsne)
        ss_clusters_densmap[cluster] = sum(ss_cluster_densmap) / len(ss_cluster_densmap)

    ss_all = []

    # Silhouette score in UMAP graph
    cluster_state_dict = adata.uns['cluster_state_dict']
    transition_states = [cluster for cluster, state in cluster_state_dict.items() if state == 'transition_states']
    core_states = [cluster for cluster, state in cluster_state_dict.items() if state == 'core_states']

    ss_tran = []
    ss_core = []

    for cluster in transition_states:
        ss_tran.append(ss_clusters_expr[cluster])

    for cluster in core_states:
        ss_core.append(ss_clusters_expr[cluster])

    ss_all.append(ss_tran)
    ss_all.append(ss_core)

    # Silhouette score in phate graph
    ss_tran = []
    ss_core = []

    for cluster in transition_states:
        ss_tran.append(ss_clusters_phate[cluster])

    for cluster in core_states:
        ss_core.append(ss_clusters_phate[cluster])

    ss_all.append(ss_tran)
    ss_all.append(ss_core)

    # T test between ss_tran and ss_core
    t_stat, p_val = ttest_ind(ss_tran, ss_core)
    print(f'T-test: t_stat: {t_stat}, p_val: {p_val}')

    # Silhouette score in variation graph
    ss_tran = []
    ss_core = []

    for cluster in transition_states:
        ss_tran.append(ss_clusters_var[cluster])

    for cluster in core_states:
        ss_core.append(ss_clusters_var[cluster])

    ss_all.append(ss_tran)
    ss_all.append(ss_core)

    # Silhouette score in tsne graph
    ss_tran = []
    ss_core = []

    for cluster in transition_states:
        ss_tran.append(ss_clusters_tsne[cluster])

    for cluster in core_states:
        ss_core.append(ss_clusters_tsne[cluster])

    ss_all.append(ss_tran)
    ss_all.append(ss_core)

    # Silhouette score in densmap graph
    ss_tran = []
    ss_core = []

    for cluster in transition_states:
        ss_tran.append(ss_clusters_densmap[cluster])

    for cluster in core_states:
        ss_core.append(ss_clusters_densmap[cluster])

    ss_all.append(ss_tran)
    ss_all.append(ss_core)

    # Silhouette score in core states
    ss_core_var = []
    ss_core_expr = []
    ss_core_phate = []
    ss_core_tsne = []
    ss_core_densmap = []

    for cluster in core_states:
        ss_core_expr.append(ss_clusters_expr[cluster])
        ss_core_phate.append(ss_clusters_phate[cluster])
        ss_core_var.append(ss_clusters_var[cluster])
        ss_core_tsne.append(ss_clusters_tsne[cluster])
        ss_core_densmap.append(ss_clusters_densmap[cluster])

    # Anova test between ss_core_tsne, ss_core_expr, ss_core_densmap, ss_core_phate, ss_core_var
    f_stat, p_val = f_oneway(ss_core_tsne, ss_core_expr, ss_core_densmap, ss_core_phate, ss_core_var)
    print(f'ANOVA: f_stat: {f_stat}, p_val: {p_val}')

    # Plot the result
    plt.figure(figsize=(10, 6))
    box = sns.boxplot(data=[ss_core_tsne, ss_core_expr, ss_core_densmap, ss_core_phate, ss_core_var], color=plt.get_cmap('tab10')(1), fill=False, width=0.5, showfliers=False)
    hatch_patterns = ['/', '\\', 'x']
    for patch, hatch in zip(box.patches, hatch_patterns):
        patch.set_hatch(hatch)

    sns.stripplot(data=[ss_core_tsne, ss_core_expr, ss_core_densmap, ss_core_phate, ss_core_var], color=plt.get_cmap('tab10')(1), jitter=True, dodge=True)
    plt.xticks(ticks=[0, 1, 2], labels=['UMAP graph', 'PHATE graph', 'Variation graph'])
    plt.title("Silhouette score in core states")
    plt.ylabel("Silhouette score")
    plt.text(0.2, 0.8, f"ANOVA p-value: {p_val:.2e}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()

    # Silhouette score in transition states
    ss_tran_expr = []
    ss_tran_phate = []
    ss_tran_var = []
    ss_tran_tsne = []
    ss_tran_densmap = []

    for cluster in transition_states:
        ss_tran_expr.append(ss_clusters_expr[cluster])
        ss_tran_phate.append(ss_clusters_phate[cluster])
        ss_tran_var.append(ss_clusters_var[cluster])
        ss_tran_tsne.append(ss_clusters_tsne[cluster])
        ss_tran_densmap.append(ss_clusters_densmap[cluster])

    # Anova test between ss_tran_tsne, ss_tran_expr, ss_tran_densmap, ss_tran_phate, ss_tran_var
    f_stat, p_val = f_oneway(ss_tran_tsne, ss_tran_expr, ss_tran_densmap, ss_tran_phate, ss_tran_var)
    print(f'ANOVA: f_stat: {f_stat}, p_val: {p_val}')

    # Plot the result
    plt.figure(figsize=(10, 6))
    box = sns.boxplot(data=[ss_tran_tsne, ss_tran_expr, ss_tran_densmap, ss_tran_phate, ss_tran_var], color=plt.get_cmap('tab10')(0), fill=False, width=0.5, showfliers=False)
    for patch, hatch in zip(box.patches, hatch_patterns):
        patch.set_hatch(hatch)

    sns.stripplot(data=[ss_tran_tsne, ss_tran_expr, ss_tran_densmap, ss_tran_phate, ss_tran_var], color=plt.get_cmap('tab10')(0), jitter=True, dodge=True)
    plt.xticks(ticks=[0, 1, 2], labels=['UMAP graph', 'PHATE graph', 'Variation graph'])
    plt.title("Silhouette score in transition states")
    plt.ylabel("Silhouette score")
    plt.text(0.2, 0.8, f"ANOVA p-value: {p_val:.2e}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()
