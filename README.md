# DirectDetectionDarkMatter-experiments

| Package | CI |
| --- | --- |
|[![Documentation Status](https://readthedocs.org/projects/dddm/badge/?version=latest)](https://dddm.readthedocs.io/en/latest/?badge=latest) | [![CodeFactor](https://www.codefactor.io/repository/github/joranangevaare/dddm/badge)](https://www.codefactor.io/repository/github/joranangevaare/dddm)|
|[![PyPI version shields.io](https://img.shields.io/pypi/v/dddm.svg)](https://pypi.python.org/pypi/dddm/) | [![Pytest](https://github.com/joranangevaare/dddm/workflows/Pytest/badge.svg)](https://github.com/joranangevaare/dddm/actions?query=workflow%3APytest) |
|[![Python Versions](https://img.shields.io/pypi/pyversions/reprox.svg)](https://pypi.python.org/pypi/reprox)| [![Coverage Status](https://coveralls.io/repos/github/JoranAngevaare/dddm/badge.svg?branch=master)](https://coveralls.io/github/JoranAngevaare/dddm?branch=master)|
| | [![DOI](https://zenodo.org/badge/214990710.svg)](https://zenodo.org/badge/latestdoi/214990710)|

Probing the complementarity of several in Direct Detection Dark Matter Experiments to reconstruct
Dark Matter models

# Installation (linux)

Please follow the installation
script [here](https://github.com/JoranAngevaare/dddm/blob/master/.github/scripts/install_on_linux.sh)

For running on multiple cores, I'd advice using `conda install -c conda-forge mpi4py openmpi`

# Author

Joran Angevaare <j.angevaare@nikhef.nl>

# Requirements

- [`wimprates`](https://github.com/joranangevaare/wimprates).
- [`verne`](https://github.com/joranangevaare/verne)
- Optimizer:
    - [`multinest`](https://github.com/JohannesBuchner/PyMultiNest)
    - [`emcee`](https://emcee.readthedocs.io/en/stable/)
    - [`nestle`](http://kylebarbary.com/nestle/)

# Options

- Multiprocessing
- Earth shielding integration
- Computing cluster utilization


