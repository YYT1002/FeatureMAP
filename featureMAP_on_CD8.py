#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:06:30 2022

"""

# from anndata import AnnData

import anndata as ad
import warnings

import numpy as np
import scanpy as sc
import scipy

# from featmap import featmap_
import featuremap

sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

#%%

#############################
# Loading the data
#############################

from scipy.io import mmread
lcmv_lognorm = mmread('./data/CD8_exhaustion/lcmv_lognorm.mtx')

adata = sc.read('./data/CD8_exhaustion/CD8_exhaustion_all.h5ad')
adata.obs['clusters'] = adata.obs['ClusterNames']
adata.obs['sampleID'] = adata.obs['sampleID'].astype(str).astype('category')

adata.layers['lognorm'] = lcmv_lognorm.T.toarray()

adata.obs['time'] = adata.obs['sampleID']
for time_label in adata.obs['sampleID'].cat.categories.values:
    if time_label == '0' or time_label == '1' or time_label == '2' or time_label == '3':  
        cluster_index = adata.obs['sampleID'].index[adata.obs['sampleID'] == time_label]
        adata.obs['time'][cluster_index] = np.nan
        
# CL13 
clusters = [ 'Eff-like','Exh-Pre', 'Prolif I', 'Prolif II', 'Exh-Int','Exh-Prog',
          'Exh-Term',  'Exh-HSP', 'Exh-TermGzma', 'Exh-KLR', 
          'ISG']


adata = ad.concat([adata[adata.obs['clusters'] == cluster] for cluster in clusters], merge="same")
adata.raw = adata

####################
# PCA
##################

import numpy as np
from sklearn.decomposition import PCA
X = adata.X.copy()
pca = PCA(n_components=100)
emb_pca = pca.fit_transform(X)

adata.obsm['X_pca'] = emb_pca

# Plot PCA
sc.pl.embedding(adata, 'pca',legend_fontsize =6, legend_loc='on data', color=['clusters'])




#%%

"""
Run featMAP
"""

data_original = adata.X
# dimensionality reduction by svd
u, s, vh = scipy.sparse.linalg.svds(
    data_original, k= min(data_original.shape[1]-1, 100), which='LM', random_state=42)
# u, s, vh = scipy.linalg.svd(data_original, full_matrices=False)
# PCA coordinates in first 100 dims
emb_svd = np.matmul(u, np.diag(s))

# Check if singular value is sorted
if all(s[i] <= s[i+1] for i in range(len(s) - 1)):
    adata.obsm['X_svd'] = np.append(emb_svd[:,-1].reshape((-1,1)),emb_svd[:,-2].reshape((-1,1)), axis=1)
    sc.pl.embedding(adata, 'svd',legend_fontsize =6, legend_loc='on data', color=['clusters'])


emb_featmap = featuremap.FeatureMAP(
                    n_neighbors=30,
                    min_dist=0.3,
                    random_state=12345,
                    n_epochs=500,
                    feat_lambda=0.5,
                    feat_frac=0.3,
                    feat_gauge_coefficient=2,
                    verbose=True,
                    ).fit(emb_svd)

adata.obsm['X_featmap'] = emb_featmap.embedding_
adata.obsm['X_featmap_v'] = emb_featmap._featuremap_kwds['variation_embedding']
adata.obsm['X_gauge_v1'] = emb_featmap._featuremap_kwds['gauge_v1_emb']
adata.obsm['X_gauge_v2'] = emb_featmap._featuremap_kwds['gauge_v2_emb']



sc.pl.embedding(adata, 'featmap',legend_fontsize =6, legend_loc='on data', color=['clusters'])
sc.pl.embedding(adata, 'featmap_v',legend_fontsize =6,s=10, legend_loc='on data', color=['clusters'])
sc.pl.embedding(adata, 'gauge_v1',legend_fontsize =6, s=10,legend_loc='on data', color=['clusters'])
sc.pl.embedding(adata, 'gauge_v2',legend_fontsize =6, legend_loc='on data', color=['clusters'])



adata.obsm['gauge_vh'] = emb_featmap._featuremap_kwds['VH_smoothing']
adata.obsm['gauge_singular_value'] = emb_featmap._featuremap_kwds['Singular_value']
adata.varm['pca_vh'] = vh.T 
adata.obsm['VH_embedding'] = emb_featmap._featuremap_kwds['VH_embedding']
adata.obsm['weight_graph'] = emb_featmap.graph_



#%%
# Variation embedding
from featuremap.plot import feature_loading, feature_variation_embedding

feature_loading(adata)

adata_var = feature_variation_embedding(adata, variation_preprocess_flag=True,random_state=12345)

#%%


# Leiden clustering
sc.tl.pca(adata_var,)
sc.pp.neighbors(adata_var, n_neighbors=15, n_pcs=30)
# sc.tl.umap(adata_var)
# adata_var.obsm['X_umap_v'] = adata_var.obsm['X_umap']
sc.tl.leiden(adata_var, resolution=0.8, random_state=42)
# Put the leiden label to adata
adata.obs['leiden_v'] = adata_var.obs['leiden']
sc.pl.embedding(adata_var, 'featmap_v', legend_loc='on data' , legend_fontsize=10,color=['leiden'])

#%%
##################################
# Contour plot to show the density
######################################
from featuremap.plot import plot_density
plot_density(adata)

#%%
#######################################################
# Compute core-states based on clusters
#########################################################
from featuremap.plot import core_transition_state
core_transition_state(adata)

#%%
#############################################################
# Ridge estimation on manifold; 
############################################################7
from featuremap.plot import ridge_estimation
ridge_estimation(adata)


#%%
from featuremap.plot import ridge_pseudotime


root = np.flatnonzero(adata.obs['corestates_largest']  == 'Prolif I')[0]
# Pseudotime by original KNN graph
ridge_pseudotime(adata, root)





#%%

# Progenitor
# Exh-KLR paht
path = [0,7,37,35,38,36,34,39]

# path_name = path[0]
# path_cluster_id = path[1]
path_id = np.array([]).astype(int)
for cluster in path:
    # cluster = '7'
    path_id = np.append(path_id, np.where(adata_var.obs['leiden'] == str(cluster))[0])
    
path_id = path_id.reshape(-1)

# Plot the path on expression embedding
adata_var.obs['core_trans_states'] = adata.obs['core_trans_states']
adata.obs['path_by_var_emb'] = np.nan
adata.obs['path_by_var_emb'][path_id] = adata_var.obs['core_trans_states'][path_id]
sc.pl.embedding(adata, 'featmap', color=['path_by_var_emb'], edges=False,)

# Plot the path on variation embedding
adata_var.obs['core_trans_states'] = adata.obs['core_trans_states']
adata_var.obs['path_by_var_emb'] = np.nan
adata_var.obs['path_by_var_emb'][path_id] = adata_var.obs['core_trans_states'][path_id]
sc.pl.embedding(adata_var, 'featmap_v', color=['path_by_var_emb'], edges=False, size=10)


#%%

#########################################
# Differential gene variation analysis
#######################################

adata_var.layers['counts'] = adata.X.copy()

# KLR cell
adata_var.obs['leiden_bridge'] = adata_var.obs['leiden'].map({                         
                          # '14': 'End',
                          '0': 'End',
                          '7': 'Bridge',
                           '37':'Bridge',
                          # '5':'Bridge'
                           })

adata.obs['leiden_v_bridge'] = adata_var.obs['leiden_bridge']
sc.pl.embedding(adata,'featmap_v',color=['leiden_v_bridge'],)


sc.tl.rank_genes_groups(adata_var, 'leiden_bridge', groups=['Bridge'],reference='End', use_raw=False, layer='var_filter')
sc.pl.rank_genes_groups(adata_var, n_genes=25, sharey=False)


#%%
from featuremap.plot import  plot_feature

plot_feature(adata, feature='Zeb2', pseudotime_adjusted=True,embedding='X_featmap_v', pseudotime='ridge_pseudotime', 
             cluster_key='leiden_v', plot_within_cluster=['0','7','37','35','38'], )

