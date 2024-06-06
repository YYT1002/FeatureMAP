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

def core_transition_state(
        adata:AnnData,
        emb='featmap',
        cluster_key='clusters',
        top_percent = 0.2
        
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
    top_percent : float 
        The top percentage of the core state in each cluster.
            
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
    data = adata.obsm[f'X_{emb}'].copy()
    p= kernel_density_estimate(data, data)

    adata.obs['density'] = p
    
    # Density in each cluster
    adata.obs['density_separate_cluster'] = np.nan
    leiden_clusters = adata.obs[cluster_key].copy()
    leiden_clusters.value_counts()
    
    for cluster in leiden_clusters.cat.categories.values:
        cluster_in_cluster_label = (leiden_clusters == cluster)
        data_cluster = data[cluster_in_cluster_label, :]
    
        p_1 = kernel_density_estimate(data_cluster, data_cluster)
        # adata.obs['density_separate_cluster'][cluster_in_cluster_label] = p_1
        adata.obs.loc[cluster_in_cluster_label, 'density_separate_cluster'] = p_1
        density = adata.obs['density_separate_cluster'][cluster_in_cluster_label]
    
    # Select top ratio(%) in each cluster as core state
    leiden_clusters = adata.obs[cluster_key].copy()
    
    adata.obs['corestates'] = np.nan
    adata.obs['corestates_largest'] = np.nan
    for cluster in leiden_clusters.cat.categories.values:
        cluster_in_cluster_label = (leiden_clusters == cluster)
        density = adata.obs['density_separate_cluster'][cluster_in_cluster_label].copy()
        # density = adata.obs['density'][cluster_in_cluster_label]
        cluster_index = leiden_clusters.index[leiden_clusters == cluster]
        density_sort = density[cluster_index].sort_values(ascending=False)
        if int(len(cluster_index) * top_percent) > 50:
            density_sort_top20per_index = density_sort.index[:50]
        else:
            density_sort_top20per_index = density_sort.index[:int(len(cluster_index) * top_percent)]
        adata.obs.loc[density_sort_top20per_index, 'corestates'] = cluster
        # non-corestate
        # density_sort_rest_index = density_sort.index[int(len(cluster_index) * 0.2):]
        # adata.obs['corestates'][density_sort_rest_index] = f'{cluster} Rest'
        
        density_sort_largest_index = density_sort.index[:1]
        # adata.obs['corestates_largest'][density_sort_largest_index] = cluster
        adata.obs.loc[density_sort_largest_index, 'corestates_largest'] = cluster
    
    # adata.obs['corestates'] = pd.Series(adata.obs['corestates'].copy(), dtype='category').values
    
    # Expand the core state by NNs
    from featuremap.featuremap_ import nearest_neighbors
    n_neighbors = 15
    knn_indices, _, _ = nearest_neighbors(adata.obsm[f'X_{emb}'].copy(), n_neighbors=n_neighbors,
                                                  metric="euclidean", metric_kwds={}, angular=False, random_state=42)

    # corestates_nn_points coresponding to clusters
    # initialize an empty dataframe
    df_temp = pd.DataFrame(index=adata.obs_names, columns=['corestates'])
    df_temp['corestates'] = np.nan
    for cluster in leiden_clusters.cat.categories.values:
        corestates_points = np.where(adata.obs['corestates'] == cluster)[0]
        corestates_nn_points = np.unique(knn_indices[corestates_points].reshape(-1))
        df_temp.loc[adata.obs_names[corestates_nn_points], 'corestates'] = cluster
        # adata.obs.loc[adata.obs_names[corestates_nn_points], 'corestates_nn_points'] = cluster
    
    adata.obs['corestates'] = df_temp['corestates']
    adata.obs['corestates'] = pd.Categorical(adata.obs['corestates'], categories=adata.obs[cluster_key].cat.categories, ordered=True)

    sc.pl.embedding(adata, emb, color=['corestates'],)
 
    # corestates_nn_points: binary
    adata.obs['corestates_binary'] = 0
    corestates_points = np.where(adata.obs['corestates'].notna())[0]
    
    corestates_points = np.unique(corestates_points.reshape(-1))
    corestates_binary = np.isin(np.array(range(adata.shape[0])), corestates_points) * 1
    adata.obs['corestates_points'] = corestates_binary
    
    adata.obs['core_trans_states'] = '0'
    corestate_points = np.where(adata.obs['corestates_points']==1)[0]
    adata.obs.loc[adata.obs_names[corestate_points],'core_trans_states'] = '1'
    

    sc.pl.embedding(adata, emb, color=['core_trans_states'])

    

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
    sc.pl.embedding(adata, 'featmap_v',legend_fontsize=10, s=10, color=['path_states'])


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