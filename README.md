![FeatureMAP Illustration](./docs/figures/featureMAP.png)

# FeatureMAP: Feature-preserving Manifold Approximation and Projection

Visualizing single-cell data is essential for understanding cellular heterogeneity and dynamics. **FeatureMAP** enhances this process by introducing **gene projection** and **transition/core states**, providing deeper insights into cellular states. While traditional methods like UMAP and t-SNE effectively capture clustering, they often overlook critical gene-level information. FeatureMAP addresses this limitation by integrating concepts from UMAP and PCA, preserving both clustering structures and gene feature variations within a low-dimensional space.

## Description

FeatureMAP presents a novel approach by enhancing manifold learning with pairwise tangent space embedding, ensuring the retention of crucial cellular data features. It introduces two visualization plots: expression embedding (GEX) and variation embedding (GVA).

Here, we demonstrate its effectiveness using a synthetic dataset from ([BEELINE](https://github.com/Murali-group/Beeline)) based on a bifurcation model. Compared to UMAP, FeatureMAP-GEX better preserves cell density, while FeatureMAP-GVA clearly delineates developmental paths.

<!-- ![Bifurcation Embedding](./docs/figures/bifurcation_embedding.png) -->

   <img src="./docs/figures/bifurcation_embedding.png" alt="Transition and Core States"/>



Besides the two-dimensional visualization, FeatureMAP presents three key concepts:

1. **Gene Projection**: Estimating and projecting gene feature loadings, where arrows indicate the direction and magnitude of gene expression changes.
    ![Gene Projection](./docs/figures/gene_contribution.png)

   
2. **Transition and Core States**: Transition and core states are computationally defined based on cell density, curvature, and betweenness centrality. Transition states are characterized by the lowest cell densities, maximal curvature, and highest betweenness centrality, whereas core states exhibit the highest cell densities, minimal curvature, and lowest betweenness centrality.
    <!-- ![Core and Transition States](./docs/figures/core_trans_states.png) -->

    <img src="./docs/figures/core_trans_states.png" alt="Transition and Core States" width="220" height="200"/>


3. **Differential Gene Variation (DGV) Analysis**: The third concept introduces differential gene variation (**DGV**) analysis, which compares transition and core states to identify genes with significant variability. By quantifying gene variation between dynamic transition states and stable core states, DGV highlights regulatory genes likely driving cell-state transitions and differentiation.  
   
    <img src="./docs/figures/DGV.png" alt="DGV"/>


FeatureMAP, a feature-preserving method, enhances the visualization and interpretation of single-cell data. Through analyses of both synthetic and real scRNA-seq data ([TUTORIAL](https://featuremap.readthedocs.io/en/latest/index.html)), FeatureMAP effectively captures intricate clustering structures and identifies key regulatory genes, offering significant advantages for single-cell data analysis.

## Getting Started

### Dependencies

- Python 3.8 or higher
- Required Python libraries: numpy, scipy, matplotlib, umap-learn, scikit-learn
- Operating System: Any (Windows, macOS, Linux)

### Installation

Install directly using pip:

```bash
pip install featuremap
```

## How to use FeatureMAP

### Data Visualization
To apply FeatureMAP in Python with a data matrix (data), where rows represent cells and columns represent genes, use the following command:
```
import featuremap
v_emb = featuremap.FeatureMAP(output_variation=True).fit_transform(data)
x_emb = featuremap.FeatureMAP(output_variation=False).fit_transform(data)


```

#### Parameters:
output_variation: bool (False by default). Decide to generate expression embedding or variation embedding. 

#### Outputs
x_emb: expession embedding to show the clustering

v_emb: variation embedding to show the trajectory


## Documentation
More tutorials are at https://featuremap.readthedocs.io/en/latest/index.html.

## Citation
Our FeatureMAP alogrithm is based on the paper

Yang, Yang, et al. "Interpretable Dimensionality Reduction by Feature Preserving Manifold Approximation and Projection." arXiv preprint arXiv:2211.09321 (2022).

## License
The FeatureMAP package is under BSD-3-Clause license.

