# Running the tutorials

These notebooks show analyses using FeatureMAP 0.0.5. Install the analysis dependencies with:

```bash
pip install "featuremap[features]" phate networkx seaborn
```

Run a notebook from the repository's `docs/notebook` directory. The four mixture-of-Gaussian examples generate their own data. The BEELINE bifurcation example reads the CSV files included under `BEELINE-data/inputs/Synthetic/dyn-BF/dyn-BF-5000-1/`. The pancreatic development example reads the included `datasets/pancreas.h5ad`, with a download URL in the notebook as a fallback.

The CD8 T cell exhaustion dataset is not included in this repository. Supply `CD8_exhaustion_cl13.h5ad` in the notebook's working directory before running that example. Its `AnnData.obs` must include the `clusters` labels used in the plots. The rendered notebooks contain saved results from earlier runs; the documentation build does not regenerate those figures.
