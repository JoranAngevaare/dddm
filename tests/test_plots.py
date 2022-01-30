import tempfile

import dddm
import matplotlib.pyplot as plt
import numpy as np
from hypothesis import given, strategies


def test_ll_s():
    dddm.plot_basics.plt_ll_sigma_spec(bins=2)
    plt.clf()
    plt.close()


def test_ll_m():
    dddm.plot_basics.plt_ll_mass_spec(bins=3)
    plt.clf()
    plt.close()


def test_plt_b():
    dddm.plot_basics.plt_priors(itot=10)
    plt.clf()
    plt.close()


def test_ll_function():
    dddm.plot_basics.show_ll_function(20)
    plt.clf()
    plt.close()


def test_simple_hist():
    dddm.plot_basics.simple_hist(np.linspace(0, 3, 3))
    with tempfile.TemporaryDirectory() as tmpdirname:
        dddm.plot_basics.save_canvas('test', save_dir=tmpdirname)
    plt.clf()
    plt.close()


@given(strategies.floats(0, 2),
       )
def test_get_color(a):
    dddm.plot_basics.get_color_from_range(a, _range=(0, max(1, a)))


def test_plt_ll_sigma_det():
    dddm.plot_basics.plt_ll_sigma_det()


def test_plt_ll_mass_det():
    dddm.plot_basics.plt_ll_mass_det()
