This software package contains an implementation of feature-preserving data visualization tool FeatureMAP, which improves the UMAP algorithm to enable the feature projection and generalizes densMAP algorithm to preseve anisotropic density.

# Installation

FeatureMAP shares the same dependencies as UMAP, including:

* numpy
* scipy
* scikit-learn
* numba==0.57.1

Other dependencies include:
`conda install -c conda-forge scanpy python-igraph leidenalg`

`conda install -c main -c conda-forge -c bioconda quasildr`

# Install Options

PyPI installation of FeatureMAP is performed as:


`pip install featuremap-learn`


For a manual install, first download this package:

```
wget https://github.com/YYT1002/FeatureMAP/archive/main.zip
unzip FeatureMAP-main.zip
rm FeatureMAP-main.zip
cd FeatureMAP-main
```

Install the requirements:

`sudo pip install -r requirements.txt`

or

`conda install scikit-learn numba`

Finally, install the package:

`python setup.py install`


# Usage
Like UMAP, the densMAP package inherits from sklearn classes, and thus drops in neatly next to other sklearn transformers with an identical calling API.
```
import featuremap
from sklearn.datasets import fetch_openml
from sklearn.utils import resample

digits = fetch_openml(name='mnist_784')
subsample, subsample_labels = resample(digits.data, digits.target, n_samples=7000, stratify=digits.target, random_state=1)

x_emb, v_emb,_,_ = featuremap.FeatureMAP().fit_transform(subsample)
```


# Input arguments
There are a number of parameters that can be set for the densMAP class; the major ones inherited from UMAP are:

* n_neighbors: This determines the number of neighboring points used in local approximations of manifold structure. Larger values will result in more global structure being preserved at the loss of detailed local structure. In general this parameter should often be in the range 5 to 50; we set a default of 30.

* min_dist: This controls how tightly the embedding is allowed compress points together. Larger values ensure embedded points are more evenly distributed, while smaller values allow the algorithm to optimise more accurately with regard to local structure. Sensible values are in the range 0.001 to 0.5. For expression embedding, we set min_dist as 0.3; for variation embedding, we set it as 0.5.

* metric: This determines the choice of metric used to measure distance in the input space. The default is 'euclidean'.

# Output arguments

Output includes the expression embedding, variation embeding, gauge embedding of 1st principal component, and gauge embedding of 2nd principal component, e.g.,

`x_emb, v_emb, gauge_1_emb, gauge_2_emb = featuremap.FeatureMAP().fit_transform(subsample)`
