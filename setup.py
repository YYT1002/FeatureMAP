#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from setuptools import setup
from setuptools import find_packages


#def readme():
#    with open('README.md') as readme_file:
#        return readme_file.read()

with open("README.md", "r") as fh:
    long_description = fh.read()
    
configuration = {
    'name' : 'featuremap-learn',
    'version': '1.0.5',
    'description' : 'FeatureMAP',
    'long_description' : long_description,
    'long_description_content_type' : "text/markdown",
    'classifiers' : [
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.9',
    ],
    'keywords' : 'dimensionality reduction, manifold learning, tangent space embedding',
    'url' : 'https://github.com/YYT1002/FeatureMAP',
    'maintainer' : 'Yang Yang',
    'maintainer_email' : 'yangyangnwpu@gmail.com',
    'license' : 'GPL',
    'packages' : ['featuremap'],
    'install_requires': ['numpy >= 1.13',
                         'scikit-learn >= 0.16',
                          'scipy >= 0.19',
                         'numba >= 0.55.0',
                         'umap-learn >= 0.5.1',
                         ],
    "extras_require": {
         "features": [
            "scanpy",
            "pandas",
            "anndata",
            "matplotlib >= 3.5.1"
        ],
         "core_transition_state": [
            "scanpy",
            "pandas",
            "anndata",
            "matplotlib >= 3.5.1"
        ],
    
    }
}
setup(**configuration)
