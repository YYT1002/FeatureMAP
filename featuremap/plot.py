
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_density_pseudotime(filtered_data, pseudotime='feat_pseudotime', clusters='clusters', density='density'):
    """
    Plot the density of cells along the pseudotime trajectory.

    Parameters
    ----------
    filtered_data : pd.DataFrame
        The filtered data containing the pseudotime, clusters, and density.
    pseudotime : str
        The column name of the pseudotime.
    clusters : str
        The column name of the clusters.
    density : str
        The column name of the density.

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
    ax.set_ylabel('')
    # ax.set_ylabel(density)
    # ax.legend().remove()
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(f'./figures/pancreas/density_pseudotime_{pseudotime}_beta.png', bbox_inches='tight')
    plt.show()