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

For running on multiple cores, I'd advise using `conda install -c conda-forge mpi4py openmpi`

# Author

Joran Angevaare <j.angevaare@nikhef.nl>

# Citation
Please cite [Zenodo](https://zenodo.org/badge/latestdoi/214990710) and [Angevaare, J. R., et al. "Complementarity of direct detection experiments in search of light Dark Matter." Journal of Cosmology and Astroparticle Physics 2022.10 (2022): 004. ](https://iopscience.iop.org/article/10.1088/1475-7516/2022/10/004/meta)

# Requirements


- WIMP spectrum generation modules:
  - [`wimprates`](https://github.com/joranangevaare/wimprates). For generic spectra generation
  - [`verne`](https://github.com/joranangevaare/verne). For generating spectra taking into account earth shielding
  - [`darkelf`](https://github.com/JoranAngevaare/DarkELF). For Ge/Si Migdal spectra generation
- Optimizer:
    - [`multinest`](https://github.com/JohannesBuchner/PyMultiNest). The fastest, but installation can be tricky
    - [`emcee`](https://emcee.readthedocs.io/en/stable/). Used mostly for validation of the other methods
    - [`nestle`](http://kylebarbary.com/nestle/). Fully pythonic, works on all platforms and
    - [`ultranest`]( https://johannesbuchner.github.io/UltraNest/using-ultranest.html). Still in alpha phase but has a lot of nice features

# Options

- Multiprocessing
- Earth shielding integration
- Computing cluster utilization


