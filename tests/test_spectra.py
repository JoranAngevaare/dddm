import DirectDmTargets as dddm
import matplotlib.pyplot as plt
import numericalunits as nu
import numpy as np
import wimprates as wr


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
        det='Xe',
        event_class=dddm.GenSpectrum,
        nbins=10):
    mw = 1
    sigma = 1e-35
    E_max = None
    args = (mw, sigma, use_SHM, dddm.experiment[det])
    events = event_class(*args)
    events.set_config({'n_energy_bins': nbins})
    if E_max:
        events.set_config({'E_max': E_max})
    events.get_data(poisson=False)


def test_detector_spectrum():
    use_SHM = dddm.SHM()
    _galactic_spectrum_inner(use_SHM)


def test_detector_spectrum():
    use_SHM = dddm.SHM()
    _galactic_spectrum_inner(use_SHM, event_class=dddm.DetectorSpectrum)


def test_shielded_detector_spectrum():
    use_SHM = dddm.VerneSHM()
    _galactic_spectrum_inner(use_SHM)


def test_detector_spectra():
    use_SHM = dddm.SHM()
    for det, det_properties in dddm.detector.experiment.items():
        if det_properties['type'] == 'combined':
            # This is not implemented as such
            continue
        if 'bg_func' in det_properties:
            _galactic_spectrum_inner(
                use_SHM, det, event_class=dddm.DetectorSpectrum, nbins=1)
        else:
            _galactic_spectrum_inner(use_SHM, det, nbins=3)
