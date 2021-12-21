import scipy.optimize
from itertools import cycle
import seaborn as sns
import multihist as mh
import datetime
import numba
import sys
import os
import scipy
import pandas as pd
import wimprates as wr
import numericalunits as nu
from tqdm import tqdm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import time
import DirectDmTargets as dddm
print('Start import')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 100)


print('Done import')
print("SYSTEM")
print(f"\tRunning on {sys.platform}")
print('\tPython version ' + sys.version.replace('\n', ''))
print(f"\tPython installation {sys.executable}")
print("MODULES")
for module in [dddm, wr]:
    print(
        f'''\t{module.__name__}\n\t\tver.:\t{module.__version__}\n\t\tPath:\t{str(module).split('from')[-1].split('__init__')[0][2:]}\n''')
