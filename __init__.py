from warnings import warn, catch_warnings, simplefilter
from .featuremap_ import FeatureMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba

import pkg_resources