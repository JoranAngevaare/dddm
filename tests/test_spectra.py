import dddm
import matplotlib.pyplot as plt
import numericalunits as nu
import numpy as np
import wimprates as wr
from unittest import skipIf


def test_simple_spectrum():
    energies = np.linspace(0.01, 20, 20)

    # dr/dr
    dr = ((nu.keV * (1000 * nu.kg) * nu.year) *
          wr.rate_migdal(energies * nu.keV,
                         mw=5 * nu.GeV / nu.c0 ** 2,
                         sigma_nucleon=1e-35 * nu.cm ** 2))

    plt.plot(energies, dr, label="WIMPrates SHM")
    dr = ((nu.keV * (1000 * nu.kg) * nu.year) *
          wr.rate_migdal(energies * nu.keV,
                         mw=0.5 * nu.GeV / nu.c0 ** 2,
                         sigma_nucleon=1e-35 * nu.cm ** 2))

    plt.plot(energies, dr, label="WIMPrates SHM")

    plt.xlabel("Recoil energy [keV]")
    plt.ylabel("Rate [events per (keV ton year)]")

    plt.xlim(0, energies.max())
    plt.yscale("log")

    plt.ylim(1e-4, 1e8)
    plt.clf()
    plt.close()


def _galactic_spectrum_inner(
        use_SHM,
        det_class=dddm.examples.XenonSimple,
        event_class=dddm.GenSpectrum,
        mw=1,
        sigma=1e-35,
        E_max=None,
        nbins=10):
    if E_max:
        detector = det_class(n_energy_bins=nbins, e_max_kev=E_max)
    else:
        detector = det_class(n_energy_bins=nbins)
    events = event_class(use_SHM, detector)
    return events.get_data(mw, sigma, poisson=False)


def test_detector_spectrum():
    use_SHM = dddm.SHM()
    assert len(_galactic_spectrum_inner(use_SHM))


def test_detector_spectrum():
    use_SHM = dddm.SHM()
    assert len(_galactic_spectrum_inner(use_SHM, event_class=dddm.DetectorSpectrum))


def test_shielded_detector_spectrum():
    use_SHM = dddm.ShieldedSHM(location='XENON')
    assert len(_galactic_spectrum_inner(use_SHM))

    
@skipIf(dddm.utils.is_windows(), "Darkelf only works on linux")
def test_detector_spectra():
    use_SHM = dddm.SHM()
    ct = dddm.test_context()
    for det, det_class in ct._detector_registry.items():
        _galactic_spectrum_inner(
            use_SHM,
            det_class=det_class,
            event_class=dddm.DetectorSpectrum,
            nbins=5)
