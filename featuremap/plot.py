#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:52:14 2023

"""


import anndata as ad
from anndata import AnnData
from quasildr.structdr import Scms
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





def plot_density(
        adata: AnnData
            ):
    data = adata.obsm['X_featmap'].copy()  # Exclude one leiden cluster;
    # data = adata[index_remove_outlier].obsm['X_featmap']  # Exclude one leiden cluster;
    # rotational_matrix = adata.uns['emb_umap']._densmap_kwds['VH_embedding']
    # rotational_matrix = adata.obsm['VH_embedding_original'].copy()
    # rotational_matrix = adata.obsm['VH_embedding'].copy()
    
    
    # r_emb = adata.obsm['rad_emb_no_log'].copy()
    
    s = Scms(data, 0.5, min_radius=5)
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
    T1 = time.time()
    # p1, g1, h1, msu,_ = s._kernel_density_estimate_anisotropic(grid_contour, rotational_matrix, r_emb)
    p1, g1, h1, msu = s._kernel_density_estimate(grid_contour, output_onlylogp=False, )
    T2 = time.time()
    print('Finish kernel_density_estimate_anisotropic in ' + str(T2-T1))
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
        cluster_key='clusters',
        top_percent = 0.2
        
        ):
    
    # adata.obs['clusters'] = adata.obs['clusters_fine']

    partition_label = adata.obs[cluster_key].copy()
    partition_label.value_counts()
    data = adata.obsm['X_featmap'].copy()
    # data = embedding.embedding_[data_index_0_2]
    # rotational_matrix = adata.obsm['VH_embedding'].copy()
    # rotational_matrix = adata.obsm['VH_embedding_original']
    
    # r_emb = adata.obsm['rad_emb_no_log'].copy()
    # from quasildr.structdr import Scms
    s = Scms(data, 0.8, min_radius=5)
    # p, _, _, _, _ = s._kernel_density_estimate_anisotropic( data, rotational_matrix, r_emb)
    p, _, _, _= s._kernel_density_estimate(data)
    adata.obs['density'] = p
    
    # Density in each cluster
    adata.obs['density_seperate_cluster'] = np.nan
    leiden_clusters = adata.obs[cluster_key].copy()
    leiden_clusters.value_counts()
    
    for cluster in leiden_clusters.cat.categories.values:
        cluster_in_cluster_label = (leiden_clusters == cluster)
        data_cluster = data[cluster_in_cluster_label, :]
        # rotational_matrix_cluster = rotational_matrix[cluster_in_cluster_label, :]
        # r_emb_cluster = r_emb[cluster_in_cluster_label, :]
    
        s = Scms(data_cluster, 0.8, min_radius=5)
        # p_1, _, _, _, _= s._kernel_density_estimate_anisotropic( data_cluster, rotational_matrix_cluster, r_emb_cluster)
        p_1, _, _, _= s._kernel_density_estimate( data_cluster)
        adata.obs['density_seperate_cluster'][cluster_in_cluster_label] = p_1
        density = adata.obs['density_seperate_cluster'][cluster_in_cluster_label]
    
    # Select top ratio(%) in each cluster as core state
    leiden_clusters = adata.obs[cluster_key].copy()
    
    adata.obs['corestates'] = np.nan
    adata.obs['corestates_largest'] = np.nan
    for cluster in leiden_clusters.cat.categories.values:
        cluster_in_cluster_label = (leiden_clusters == cluster)
        density = adata.obs['density_seperate_cluster'][cluster_in_cluster_label].copy()
        # density = adata.obs['density'][cluster_in_cluster_label]
        cluster_index = leiden_clusters.index[leiden_clusters == cluster]
        density_sort = density[cluster_index].sort_values(ascending=False)
        if int(len(cluster_index) * top_percent) > 50:
            density_sort_top20per_index = density_sort.index[:50]
        else:
            density_sort_top20per_index = density_sort.index[:int(len(cluster_index) * top_percent)]
        # density_sort_top20per_index = density_sort.index[:int(len(cluster_index) * 0.2)]
        # density_sort_top20per_index = density_sort.index[:200]
        adata.obs['corestates'][density_sort_top20per_index] = cluster
        # non-corestate
        # density_sort_rest_index = density_sort.index[int(len(cluster_index) * 0.2):]
        # adata.obs['corestates'][density_sort_rest_index] = f'{cluster} Rest'
        
        density_sort_largest_index = density_sort.index[:1]
        adata.obs['corestates_largest'][density_sort_largest_index] = cluster
    
    adata.obs['corestates'] = pd.Series(
        adata.obs['corestates'].copy(), dtype='category').values
    
    
    # Expand the core state by NNs
    from featuremap.featuremap_ import nearest_neighbors
    n_neighbors = 30
    knn_indices, knn_dists, _ = nearest_neighbors(adata.obsm['X_featmap'].copy(), n_neighbors=n_neighbors,
                                                  metric="euclidean", metric_kwds={}, angular=False, random_state=42)

    # corestates_nn_points coresponding to clusters
    adata.obs['corestates_nn_points'] = np.nan
    for cluster in leiden_clusters.cat.categories.values:
        corestates_points = np.where(adata.obs['corestates'] == cluster)[0]
        corestates_nn_points = np.unique(knn_indices[corestates_points].reshape(-1))
        # corestates_nn_points_binary = np.isin(np.array(range(adata.shape[0])), corestates_nn_points) * 1
        adata.obs['corestates_nn_points'][corestates_nn_points] = cluster
    sc.pl.embedding(adata, 'featmap', color=['corestates_nn_points'],)
 
    # corestates_nn_points: binary
    adata.obs['corestates_nn_points'] = np.nan
    corestates_points = np.where(adata.obs['corestates'].isna() == False)[0]
    
    corestates_nn_points = np.unique(knn_indices[corestates_points].reshape(-1))
    corestates_nn_points_binary = np.isin(np.array(range(adata.shape[0])), corestates_nn_points) * 1
    adata.obs['corestates_nn_points'] = corestates_nn_points_binary
    
    adata.obs['core_trans_states'] = '0'
    corestate_points = np.where(adata.obs['corestates_nn_points']==1)[0]
    # adata.obs['path_core_points'][trajectory_points] = '0'
    adata.obs['core_trans_states'][corestate_points] = '1'
    
    # from pandas.api.types import CategoricalDtype
    # cat_type = CategoricalDtype(categories=['Transition', 'Core'], ordered=True)
    # adata.obs['core_trans_states'] = adata.obs['core_trans_states'].astype(cat_type)
    
    sc.pl.embedding(adata, 'featmap', color=['core_trans_states'])

    

########################################################
# Collect trasition state and core state given clusters
##############################################################

def nodes_of_transition_states(adata, start_state, end_state, clusters):

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



def ridge_estimation(
        adata:AnnData
        ):
    
    data = adata.obsm['X_featmap'].copy()  # Exclude one leiden cluster;
    # data = adata_var.obsm['X_umap_v']
    pos_collection = []
    # for sample_time in range(20):
    s = Scms(data, 0.5, min_radius=5)
    # p, _, h, msu,_ = s._kernel_density_estimate_anisotropic(data, rotational_matrix, r_emb)
    p, _, h, msu = s._kernel_density_estimate(data)
    ifilter_2 =  np.where(p >= (np.max(p)*0.05))[0] # sampling
    # shifted = np.append(grid_contour[ifilter_1, :],data[ifilter_2, :], axis=0)
    shifted = data[ifilter_2,:]
    inverse_sample_index = s.inverse_density_sampling(shifted, n_samples=200, n_jobs=1, batch_size=16)
    # ifilter_3 = np.random.randint(adata.shape[0], size=100)
    # shifted = np.append(shifted[inverse_sample_index], data[ifilter_3,:],axis=0)
    # inverse_sample_index = np.unique(np.array(inverse_sample_index).reshape(-1))
    shifted = shifted[inverse_sample_index]
    
    n_iterations = 500
    allshiftedx_grid = np.zeros((shifted.shape[0],n_iterations))
    allshiftedy_grid = np.zeros((shifted.shape[0],n_iterations))
    for j in range(n_iterations):
        allshiftedx_grid[:,j] = shifted[:,0]
        allshiftedy_grid[:,j] = shifted[:,1]
        shifted += 1*s.scms_update(shifted,method='GradientLogP',stepsize=0.02, relaxation=0.5)[0]
    pos = np.column_stack([allshiftedx_grid[:,-1], allshiftedy_grid[:,-1]])
    pos_collection.append(pos)
    pos = np.array(pos_collection).reshape(-1,2)
    p_pos, _, _, _ = s._kernel_density_estimate(pos)
    pos_filter_idx =  np.where(p_pos >= (np.max(p_pos)*0.1))[0] # sampling
    pos_filter = pos[pos_filter_idx]
    
    # Plot the ridge
    s = Scms(data, 0.5, min_radius=5)
    min_x = min(data[:, 0])
    max_x = max(data[:, 0])
    min_y = min(data[:, 1])
    max_y = max(data[:, 1])
    # part = 200
    num_grid_point = data.shape[0] * 0.5
    x_range = max_x - min_x
    y_range = max_y - min_y
    # x_range = 1 - 0.618
    # y_range = 0.618
    part_y = np.sqrt(num_grid_point / x_range * y_range)
    part_x = x_range / y_range * part_y
    # Assign num of grid points mort to vertical direction ??
    xv, yv = np.meshgrid(np.linspace(min_x, max_x, round(part_x)), np.linspace(min_y, max_y, round(part_y)),
                         sparse=False, indexing='ij')
    grid_contour = np.column_stack([np.concatenate(xv), np.concatenate(yv)])
    p1, g1, h1, msu = s._kernel_density_estimate(grid_contour, output_onlylogp=False, )
    
    plt.contourf(xv, yv, p1.reshape(
        round(part_x), round(part_y)), levels=20, cmap='Blues')
    plt.scatter(data[:,0],data[:,1], s=1, c='darkgrey', alpha=0.1)
    plt.scatter(pos_filter[:,0],pos_filter[:,1],c="red", s=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.clf()
    
    
    # Relate the ridge points under the embedding graph
    trajectory_points = pos_filter
    trajectory_nn_points = np.array([], dtype=int)
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors().fit(adata.obsm['X_featmap'].copy())
    _, indices = nbrs.kneighbors(trajectory_points)
    trajectory_nn_points = np.append(trajectory_nn_points, indices[:, 0])
    # trajectory_nn_points = np.append(trajectory_nn_points, indices[:,1])
    trajectory_nn_points = np.unique(trajectory_nn_points)
    # data_trajectory_nn_points = adata.obsm['X_featmap'][trajectory_nn_points]
    # plt.contourf(xv, yv, p1.reshape(
    #     round(part_x), round(part_y)), levels=20, cmap='Blues')
    
    # # label = np.array(adata.obs['leiden'].values).astype(int)
    # # label = np.array(adata.obs['corestates'].values).astype(int)
    # # label[label<0] = 4
    # # plt.scatter(data[:, 0], data[:, 1], cmap='fire', c=[sns.color_palette(n_colors=200)[x] for x in label],
    # #             s=0.5, alpha=0.1)
    # plt.scatter(data_trajectory_nn_points[:, 0], data_trajectory_nn_points[:,
    #             1], color='red', s=0.5)  # 1-dim density ridge
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # plt.clf()
    
    trajectory = np.isin(np.array(range(adata.shape[0])), trajectory_nn_points)
    adata.obs['trajectory_points'] = (trajectory * 1).astype(int)
    
    # ##########################
    # # Connect the trajectory points
    # ############################
    # from umap.umap_ import fuzzy_simplicial_set
    # _, _, _, knn_dists = fuzzy_simplicial_set(
    #     adata.obsm['X_featmap'] ,
    #     n_neighbors=60,
    #     random_state=42,
    #     metric="euclidean",
    #     metric_kwds={},
    #     # knn_indices,
    #     # knn_dists,
    #     verbose=True,
    #     return_dists=True)

    # # M = adata.obsp['emb_dists'].copy().toarray()
    # M = knn_dists.toarray()

    # # Pairwise shortest path
    # graph = csr_matrix(M)
    # # dist_matrix, predecessors = shortest_path(
    # #     csgraph=graph, directed=False, method='D', return_predecessors=True)
    # dist_matrix, predecessors = dijkstra(
    #     csgraph=graph, directed=False, return_predecessors=True)

    # # Set the points in the tree
    # ridge_points = np.where(np.array(adata.obs['trajectory_points'])==1)[0]

    # # corestae_points: largest density in expression plot
    # corestate_points = np.where(pd.isna((adata.obs['corestates_largest'])) == False)[0]
    # # corestate_points = np.where(pd.isna((adata.obs['corestates'])) == False)[0]

    # # Add largest and smallest pseudotime points to compute the induced subgraph
    # # Points for tree
    # tree_points = np.union1d(ridge_points, corestate_points)
    # # tree_points = ridge_points

    # mst_subg = mst_subgraph(adata, tree_points,)
    # mst_subg.clusters().summary()

    # # import igraph as ig
    # # layout = mst_subg.layout("kamada_kawai")
    # # ig.plot(mst_subg, layout=layout)
    # dfs_trace_collection = dfs_1(adata, mst_subg, predecessors)



    # plt.contourf(xv, yv, p1.reshape(
    #     round(part_x), round(part_y)), levels=20, cmap='Blues')
    # data = adata.obsm['X_featmap'].copy()
    # plt.scatter(data[:,0],data[:,1], s=1, c='darkgrey', alpha=0.1)
    # # plt.scatter(data[trajectory_points,0],data[trajectory_points,1], s=1, c='red')

    # emb= adata.obsm['X_featmap']
    # for i in range(dfs_trace_collection.shape[0]):
    #     # i = 1
    #     for j in range(dfs_trace_collection[i].shape[0]-1):
    #         # j = 0
    #         cur_idx = np.array([dfs_trace_collection[i][j], dfs_trace_collection[i][j+1]])
    #         plt.plot(emb[cur_idx,0], emb[cur_idx,1], 'ro-', linewidth=2, markersize=0.1)

    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # plt.clf()

    

# from scipy.sparse.csgraph import shortest_path, dijkstra
def mst_subgraph(adata, tree_points, emb='X_featmap'):
    """

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
    return Q.scale / scale_factor



def plot_gauge(
        adata:AnnData,
        embedding='X_featmap',
        vkey='X_gauge_v1',
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
    s = Scms(X_emb, 0.5, min_radius=5)

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
    
    p1, _, _, _ = s._kernel_density_estimate(
      X_grid)
    
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
          V_grid /= 3 * quiver_autoscale(X_grid, V_grid)
    
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
    plt.show()
    plt.clf()
    
    
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

# @numba.njit()
def feature_loading(
        adata: AnnData,
        parallel=False
            ):
    """
    Compute the feature variation and feature loadings based on local SVD.
    
    Parameters
    ----------
    adata : AnnData
        An annotated data matrix.
    """
    import numpy as np
    gauge_vh = adata.obsm['gauge_vh'].copy()
    # gauge_vh_original = adata.obsm['gauge_vh_original'].copy()
    # gauge_vh = gauge_vh_original
    
    # gauge_u = adata.obsm['gauge_u'].copy()
    singular_values_collection = adata.obsm['gauge_singular_value'].copy()
    pca_vh = adata.varm['pca_vh'].copy().T
    
    T1 = time.time()
    # Compute intrinsic dimensionality locally
    def pc_accumulation(arr, threshold):
        arr_sum = np.sum(np.square(arr))
        temp_sum = 0
        for i in range(arr.shape[0]):
            temp_sum += arr[i] * arr[i]
            if temp_sum > arr_sum * threshold:
                return i
    
    threshold = 0.9
    
    intrinsic_dim = np.zeros(adata.shape[0]).astype(int)
    
    for i in range(adata.shape[0]):            
        intrinsic_dim[i] = pc_accumulation(singular_values_collection[i], threshold)
    plt.hist(intrinsic_dim)
    plt.title('Local_intrinsic_dim')
    plt.show()
    plt.clf()
    
    adata.obs['instrinsic_dim'] = intrinsic_dim
    T2 = time.time()
    print(f'Local intrinsic dim time is {T2-T1}')
    
    # sc.pl.embedding(adata,'featmap',color=['instrinsic_dim'],cmap='plasma')
    # sc.pl.embedding(adata,'umap',color=['instrinsic_dim'],cmap='plasma')
    
    # Compute the gene norm in top k PCs (norm of the arrow in biplot)
    k = int(np.median(intrinsic_dim))
    
    print("Start matrix multiplication")
    T1 = time.time()
    pcVals_project_back = np.matmul(gauge_vh, pca_vh[np.newaxis, :])
    # pcVals_project_back =  matrix_multiply(gauge_vh, pca_vh)
    T2 = time.time()
    print(f'Finish matrix multiplication in {T2-T1}')
    
    T1 = time.time()
    
    # if parallel:
    # gene_val_norm = compute_norm_parallel(pcVals_project_back[:, :k, :], 500)
    # else:
    gene_val_norm = np.linalg.norm(pcVals_project_back[:, :k, :], axis=1)
    # gene_val_norm = np.sum((pcVals_project_back[:, :k, :])**2, axis=1) # square norm


    # velocity = gene_val_norm
    # adata.obsm['pc_loadings'] = pcVals_project_back[:, :k, :]
    adata.layers['variation_feature'] = gene_val_norm
    T2 = time.time()
    print(f'Finish norm calculation in {T2-T1}')
    
    T1 = time.time()        
    gene_norm_first_two = np.linalg.norm(pcVals_project_back[:, :2, :], axis=1)
    pc_loadings_scale = pcVals_project_back[:, :2, :] /\
        gene_norm_first_two[:,np.newaxis,:] *\
            gene_val_norm[:,np.newaxis,:]
    
    # pc_loadings_scale = pcVals_project_back[:, :2, :] /\
    #     np.linalg.norm(pcVals_project_back[:, :2, :], axis=1)[:,np.newaxis,:] 
    
    adata.obsm['feature_loading_scale'] = pc_loadings_scale
    T2 = time.time()
    print(f'Finish feature loading in {T2-T1}')
    
    # Feature loadings on each local gauge
    gauge_vh_emb = adata.obsm['VH_embedding']
    feature_loading_emb = adata.obsm['feature_loading_scale'] 
    feature_loadings_embedding = np.matmul(feature_loading_emb.transpose(0,2,1), gauge_vh_emb.transpose(0,2,1)) # Project to gauge_embedding
    adata.obsm['feature_loading_embedding'] = feature_loadings_embedding.transpose(0,2,1)
    
    
# @numba.njit()
# def local_svd(
#         data,
#         weight,
#         k
#         ):
#     gene_val_norm = np.empty((data.shape[0], data.shape[1]), dtype=np.float32)
#     pc_loadings_scale = np.empty((data.shape[0],2, data.shape[1]), dtype=np.float32)
    
#     for row_i in range(data.shape[0]):
        
#         # row_i = 0
#         row_weight = weight[row_i]
        
#         # if len(row_weight) < n_neighbors_in_guage:
#         #     raise ValueError(
#         #         "Some rows contain fewer than n_neighbors distances!"
#         #     )
#         #  val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
#         data_around_i_index = np.argsort(-row_weight)[1: 61]
#         # print('data_around_i_index, ' + str(data_around_i_index))
#         data_around_i = data[data_around_i_index] - data[row_i]
        
#         weight_around_i = np.zeros(60, dtype=np.float32)
#         for neighbor_j in range(60):
#             neighbor_j_index = data_around_i_index[neighbor_j]
#             weight_around_i[neighbor_j] = weight[row_i][neighbor_j_index]
#             # idx = np.intersect1d(np.where(head==row_i)[0], np.where(tail==neighbor_j_index)[0])[0]
#             # weight_around_i[neighbor_j] = weight[idx]
            
#         # weighted SVD around i
#         # store first n_components rows of VH
#         weight_around_i = np.diag(weight_around_i) / np.sum(weight_around_i)
#         weight_around_i = np.sqrt(weight_around_i)
#         _, _, vh = np.linalg.svd(np.dot(weight_around_i, data_around_i), full_matrices=False)
        
#         # gene_val_norm[row_i, :] = np.linalg.norm(vh[:k,:], axis=0)
#         gene_val_norm[row_i, :] = np.sum(vh[:k,:]**2, axis=0)

#         # pc_loadings_scale[row_i, :, :] = vh[:2,:]/\
#         #     np.linalg.norm(vh[:2,:], axis=0) *\
#         #         gene_val_norm[row_i, :]
        
#         pc_loadings_scale[row_i, :, :] = vh[:2,:]/\
#             np.sum(vh[:2,:]**2, axis=0) *\
#                 gene_val_norm[row_i, :]
        
#     return gene_val_norm, pc_loadings_scale 


# def feature_loading(
#         adata: AnnData
#             ):
#     """
#     Compute the feature variation and feature loadings based on local SVD.
    
#     Parameters
#     ----------
#     adata : AnnData
#         An annotated data matrix.
#     """
#     import numpy as np
#     # gauge_vh = adata.obsm['gauge_vh'].copy()
#     # gauge_vh_original = adata.obsm['gauge_vh_original'].copy()
#     # gauge_vh = gauge_vh_original
    
#     # gauge_u = adata.obsm['gauge_u'].copy()
#     singular_values_collection = adata.obsm['gauge_singular_value'].copy()
#     # pca_vh = adata.varm['pca_vh'].copy().T
    
#     T1 = time.time()
#     # Compute intrinsic dimensionality locally
#     def pc_accumulation(arr, threshold):
#         arr_sum = np.sum(np.square(arr))
#         temp_sum = 0
#         for i in range(arr.shape[0]):
#             temp_sum += arr[i] * arr[i]
#             if temp_sum > arr_sum * threshold:
#                 return i
    
#     threshold = 0.9
    
#     intrinsic_dim = np.zeros(adata.shape[0]).astype(int)
    
#     for i in range(adata.shape[0]):            
#         intrinsic_dim[i] = pc_accumulation(singular_values_collection[i], threshold)
#     plt.hist(intrinsic_dim)
#     plt.title('Local_intrinsic_dim')
#     plt.show()
#     plt.clf()
    
#     adata.obs['instrinsic_dim'] = intrinsic_dim
#     T2 = time.time()
#     print(f'Local intrinsic dim time is {T2-T1}')
    
     
#     # Compute the gene norm in top k PCs (norm of the arrow in biplot)
#     k = int(np.median(intrinsic_dim))
    
#     weight = adata.obsm['weight_graph'].toarray()
#     # knn_index = adata.obsm['knn_indices'] 
#     data = adata.X.astype(np.float32)
    
#     # gene_val_norm = np.empty((data.shape[0], data.shape[1]), dtype=np.float32)
#     # pc_loadings_scale = np.empty((data.shape[0],2, data.shape[1]), dtype=np.float32)

#     T1 = time.time()
    
#     # for row_i in range(data.shape[0]):
        
#     #     # row_i = 0
#     #     row_weight = weight[row_i]
        
#     #     # if len(row_weight) < n_neighbors_in_guage:
#     #     #     raise ValueError(
#     #     #         "Some rows contain fewer than n_neighbors distances!"
#     #     #     )
#     #     #  val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
#     #     data_around_i_index = np.argsort(-row_weight)[1: 61]
#     #     # print('data_around_i_index, ' + str(data_around_i_index))
#     #     data_around_i = data[data_around_i_index] - data[row_i]
        
#     #     weight_around_i = np.zeros(60, dtype=np.float32)
#     #     for neighbor_j in range(60):
#     #         neighbor_j_index = data_around_i_index[neighbor_j]
#     #         weight_around_i[neighbor_j] = weight[row_i][neighbor_j_index]
#     #         # idx = np.intersect1d(np.where(head==row_i)[0], np.where(tail==neighbor_j_index)[0])[0]
#     #         # weight_around_i[neighbor_j] = weight[idx]
            
#     #     # weighted SVD around i
#     #     # store first n_components rows of VH
#     #     weight_around_i = np.diag(weight_around_i) / np.sum(weight_around_i)
#     #     weight_around_i = np.sqrt(weight_around_i)
#     #     _, _, vh = np.linalg.svd(np.dot(weight_around_i, data_around_i), full_matrices=False)
        
#     #     # gene_val_norm[row_i, :] = np.linalg.norm(vh[:k,:], axis=0)
#     #     gene_val_norm[row_i, :] = np.sum(vh[:k,:]**2, axis=0)

#     #     # pc_loadings_scale[row_i, :, :] = vh[:2,:]/\
#     #     #     np.linalg.norm(vh[:2,:], axis=0) *\
#     #     #         gene_val_norm[row_i, :]
        
#     #     pc_loadings_scale[row_i, :, :] = vh[:2,:]/\
#     #         np.sum(vh[:2,:]**2, axis=0) *\
#     #             gene_val_norm[row_i, :]
    
    
#     gene_val_norm, pc_loadings_scale = local_svd(data, weight, k)
#     T2 = time.time()
#     print(f'Finish norm calculation in {T2-T1}')

       
    
#     # print("Start matrix multiplication")
#     # T1 = time.time()
#     # pcVals_project_back = np.matmul(gauge_vh, pca_vh[np.newaxis, :])
#     # # pcVals_project_back =  matrix_multiply(gauge_vh, pca_vh)
#     # T2 = time.time()
#     # print(f'Finish matrix multiplication in {T2-T1}')
    
#     # T1 = time.time()
#     # gene_val_norm = np.linalg.norm(pcVals_project_back[:, :k, :], axis=1)


#     # velocity = gene_val_norm
#     # adata.obsm['pc_loadings'] = pcVals_project_back[:, :k, :]
#     adata.layers['variation_feature'] = gene_val_norm
#     # T2 = time.time()
#     # print(f'Finish norm calculation in {T2-T1}')
    
    
#     # # pc_loadings_scale = pcVals_project_back[:, :2, :] /\
#     # #     np.linalg.norm(pcVals_project_back[:, :2, :], axis=1)[:,np.newaxis,:] *\
#     # #         np.linalg.norm(pcVals_project_back[:, :k, :], axis=1)[:,np.newaxis,:]
#     # T1 = time.time()        
#     # pc_loadings_scale = pcVals_project_back[:, :2, :] /\
#     #     np.linalg.norm(pcVals_project_back[:, :2, :], axis=1)[:,np.newaxis,:] 
    
#     adata.obsm['feature_loading_scale'] = pc_loadings_scale
#     T2 = time.time()
#     print(f'Finish feature loading in {T2-T1}')
    
#     # Feature loadings on each local gauge
#     gauge_vh_emb = adata.obsm['VH_embedding']
#     feature_loading_emb = adata.obsm['feature_loading_scale'] 
#     feature_loadings_embedding = np.matmul(feature_loading_emb.transpose(0,2,1), gauge_vh_emb.transpose(0,2,1)) # Project to gauge_embedding
#     adata.obsm['feature_loading_embedding'] = feature_loadings_embedding.transpose(0,2,1)
    
    
    

 
def plot_feature(
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
    adata.obsm[vkey] = adata.obsm['feature_loading_embedding'][:,:,feature_id]

    # Set grid as the support
    X_emb=adata.obsm[embedding]
    # rotational_matrix = adata.uns['emb_umap']._densmap_kwds['VH_embedding']
    # rotational_matrix = adata.obsm['VH_embedding']
    # r_emb = adata.obsm['rad_emb_no_log']
    s = Scms(X_emb, 0.5, min_radius=5)
    
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
    p1, _, _, _ = s._kernel_density_estimate(
      X_grid)
     
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
          V_grid /= 2* quiver_autoscale(X_grid, V_grid)
          
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

   
    plt.contourf(meshes_tuple[0], meshes_tuple[1], p1.reshape(int(50 * density),int(50 * density)),
                  levels=20, cmap='Blues')
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
    plt.show()
    plt.clf()
    # plt.savefig(f'./data/flow/gene_{feature}.pdf')


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
    
    # PCA by svd
    import scipy
    u, s, vh = scipy.sparse.linalg.svds(
        data_original, k= min(data_original.shape[1]-1, 100), which='LM', random_state=42)
    # u, s, vh = scipy.linalg.svd(gene_val_norm, full_matrices=False)
    # PCA coordinates in first 100 dims
    emb_svd = np.matmul(u, np.diag(s))
    
    import umap
    emb_umap = umap.UMAP(random_state=random_state, n_neighbors=30,min_dist=0.5, spread=1, n_components=n_components).fit(emb_svd)
    adata_var.obsm['X_featmap_v'] = emb_umap.embedding_
    sc.pl.embedding(adata_var, 'featmap_v', legend_fontsize=10,color=['clusters'], projection='2d', size=20, )
    
    # sc.pl.embedding(adata_var, 'umap_v', legend_fontsize=10,color=['clusters_original'], projection='2d', size=20, )
    adata.obsm['X_featmap_v'] = adata_var.obsm['X_featmap_v']
    
    return adata_var
       