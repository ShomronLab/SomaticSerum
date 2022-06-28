import os
from pathlib import Path

from numpy.random import seed as set_random_seed

set_random_seed(42)

PROJECT_NAME = 'SomaticSerum'

DEBUG = False


# import os
# PROJECT_DIR = Path(os.path.abspath(os.curdir))
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / 'data'
TMP_DIR = PROJECT_DIR / 'tmp'
SRC_DIR = PROJECT_DIR / 'src'
_RESULTS_DIR = PROJECT_DIR / 'results'

PLOTS_DIR = _RESULTS_DIR / 'graphs'
RESULTS_DIR = _RESULTS_DIR / 'output'
