"""Introduce detector effects into the expected detection spectrum"""

from warnings import warn
import numba
import numpy as np
# from dddm import GenSpectrum
import dddm


def det_res_Xe(E):
    """
    :param E: recoil energy (in keV)
    :return: detector resolution for Xe detector
    """
    return 0.6 * np.sqrt(E)


def det_res_Ar(E):
    """
    :param E: recoil energy (in keV)
    :return: detector resolution for Ar detector
    """
    return 0.7 * np.sqrt(E)


def det_res_Ge(E):
    """
    :param E: recoil energy (in keV)
    :return: detector resolution for Ge detector
    """
    return np.sqrt(0.3 ** 2 + (0.06 ** 2) * E)


def _flat_res(E, resolution):
    """Return a flat resolution spectrum over energy range"""
    return np.full(len(E), resolution)


def det_res_superCDMS5(E):
    # https://arxiv.org/abs/1610.00006
    return _flat_res(E, 5. / 1000)


def det_res_superCDMS10(E):
    # https://arxiv.org/abs/1610.00006
    return _flat_res(E, 10. / 1000)


def det_res_superCDMS25(E):
    # https://arxiv.org/abs/1610.00006
    return _flat_res(E, 25. / 1000)


def det_res_superCDMS50(E):
    # https://arxiv.org/abs/1610.00006
    return _flat_res(E, 50. / 1000)


def det_res_superCDMS100(E):
    # https://arxiv.org/abs/1610.00006
    return _flat_res(E, 100. / 1000)


def det_res_superCDMS110(E):
    # https://arxiv.org/abs/1610.00006
    return _flat_res(E, 110. / 1000)


def det_res_XENON1T(E):
    """
    Detector resolution of XENON1T. See e.g. 1 of
        https://journals.aps.org/prd/pdf/10.1103/PhysRevD.102.072004
    :param E: energy in keV
    :return: resolution at E
    """
    a = 0.310
    b = 0.0037
    return a * np.sqrt(E) + b * E


def er_background_xe(e_min, e_max, nbins):
    """
    :return: ER background for Xe detector in events/keV/t/yr
    """
    # From https://arxiv.org/pdf/2007.08796.pdf
    bg_rate = 12.3  # 1/(keV * t * yr)

    # Assume flat background over entire energy range
    # True to first order below 200 keV
    if e_min > e_max or e_max > 200:
        mes = f'Assume flat background only below 200 keV ({e_min}, {e_max})'
        raise ValueError(mes)
    return np.full(nbins, bg_rate)


def nr_background_xe(e_min, e_max, nbins):
    """
    :return: NR background for Xe detector in events/keV/t/yr
    """
    # From https://arxiv.org/pdf/2007.08796.pdf
    bg_rate = 2.2e-3  # 1/(keV * t * yr)

    # Assume flat background over entire energy range
    # True to first order below 200 keV
    if e_min > e_max or e_max > 200:
        mes = f'Assume flat background only below 200 keV ({e_min}, {e_max})'
        raise ValueError(mes)
    return np.full(nbins, bg_rate)


def migdal_background_superCDMS_Ge_HV(e_min, e_max, nbins):
    """
    :return: background for Ge HV detector in events/keV/t/yr
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 32Si (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 27  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    if e_max > 100:  # 100 keV
        raise ValueError(
            f'Assume flat background only below 100 keV ({e_min}, {e_max})')
    if e_max >= 20:  # keV
        warn(
            f'migdal_background_superCDMS_Si_HV is not strictly valid up to {e_max} keV!')
    return np.full(nbins, bg_rate * conv_units)


def migdal_background_superCDMS_Si_HV(e_min, e_max, nbins):
    """
    :return: background for Si HV detector in events/keV/t/yr
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 32Si (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 300  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    if e_max > 100:  # 100 keV
        raise ValueError(
            f'Assume flat background only below 100 keV ({e_min}, {e_max})')
    if e_max >= 20:  # keV
        warn(
            f'migdal_background_superCDMS_Si_HV is not strictly valid up to {e_max} keV!')
    return np.full(nbins, bg_rate * conv_units)


def migdal_background_superCDMS_Ge_iZIP(e_min, e_max, nbins):
    """
    :return: background for Ge iZIP detector in events/keV/t/yr
    """
    bg_rate = 22  # counts/kg/keV/year see table V: https://arxiv.org/pdf/1610.00006.pdf
    conv_units = 1.0e3  # Tonne
    if e_max >= 20:  # 20 keV
        raise ValueError(
            f'Assume flat background only below 10 keV ({e_min}, {e_max})')
    return np.full(nbins, bg_rate * conv_units)


def migdal_background_superCDMS_Si_iZIP(e_min, e_max, nbins):
    """
    :return: background for Si iZIP detector in events/keV/t/yr
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 3H (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 370  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    if e_max >= 100:
        raise ValueError(
            f'Assume flat background only below 100 keV ({e_min}, {e_max})')
    return np.full(nbins, bg_rate * conv_units)


def nr_background_superCDMS_Ge(e_min, e_max, nbins):
    """
    :return: background for Ge iZIP/HV detector in events/keV/t/yr
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 3H (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 3300 * 1e-6  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne

    # Assume only flat over first 20 keV thereafter negligible.
    energies = np.linspace(e_min, e_max, nbins)
    res = np.zeros(nbins)
    res[energies < 20] = bg_rate * conv_units
    return res


def nr_background_superCDMS_Si(e_min, e_max, nbins):
    """
    :return: background for Si iZIP/HV detector in events/keV/t/yr
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 3H (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 2900 * 1e-6  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne

    # Assume only flat over first 20 keV thereafter negligible.
    energies = np.linspace(e_min, e_max, nbins)
    res = np.zeros(nbins)
    res[energies < 20] = bg_rate * conv_units
    return res


# Set up a dictionary of the different detectors
# Each experiment below lists:
# Name :{Interaction type (type0, exposure [ton x yr] (exp.), cut efficiency (cut_eff),
# nuclear recoil acceptance (nr_eff), energy threshold [keV] (E_thr),
# resolution function (res)

experiment = {
    'Xe_simple': {'material': 'Xe', 'type': 'SI', 'exp': 5., 'cut_eff': 0.8, 'nr_eff': 0.5, 'E_thr': 10.,
                  'location': "XENON", 'res': det_res_Xe, 'n_energy_bins': 10, 'E_max': 100},
    'Ge_simple': {'material': 'Ge', 'type': 'SI', 'exp': 3., 'cut_eff': 0.8, 'nr_eff': 0.9, 'E_thr': 10.,
                  'location': "SUF", 'res': det_res_Ge, 'n_energy_bins': 10, 'E_max': 100},
    'Ar_simple': {'material': 'Ar', 'type': 'SI', 'exp': 10., 'cut_eff': 0.8, 'nr_eff': 0.8, 'E_thr': 30.,
                  'location': "XENON", 'res': det_res_Ar, 'n_energy_bins': 10, 'E_max': 100},
    # --- Ge iZIP --- #
    'Ge_iZIP': {
        'material': 'Ge',
        'type': 'SI',
        'exp': 56 * 1.e-3,  # Tonne year
        'cut_eff': 0.75,  # p. 11, right column
        'nr_eff': 0.85,  # p. 11, left column
        'E_thr': 272. / 1e3,  # table VIII, Enr
        "location": "SNOLAB",
        'res': det_res_superCDMS100,  # table I
        'bg_func': nr_background_superCDMS_Ge,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    'Ge_migd_iZIP': {
        'material': 'Ge',
        'type': 'migdal',
        'exp': 56 * 1.e-3,  # Tonne year
        'cut_eff': 0.75,  # p. 11, right column
        'nr_eff': 0.5,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr': 350. / 1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS50,  # table I
        'bg_func': migdal_background_superCDMS_Ge_iZIP,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    # --- Si iZIP bg --- #
    'Ge_iZIP_Si': {
        'material': 'Si',
        'type': 'SI',
        'exp': 4.8 * 1.e-3,  # Tonne year
        'cut_eff': 0.75,  # p. 11, right column
        'nr_eff': 0.85,  # p. 11, left column
        'E_thr': 166. / 1e3,  # table VIII, Enr
        "location": "SNOLAB",
        'res': det_res_superCDMS110,  # table I
        'bg_func': nr_background_superCDMS_Si,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    'Ge_migd_iZIP_Si': {
        'material': 'Si',
        'type': 'migdal',
        'exp': 4.8 * 1.e-3,  # Tonne year
        'cut_eff': 0.75,  # p. 11, right column
        'nr_eff': 0.675,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr': 175. / 1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS25,  # table I
        'bg_func': migdal_background_superCDMS_Si_iZIP,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    # --- Ge HV bg --- #
    'Ge_HV': {
        'material': 'Ge',
        'type': 'SI',
        'exp': 44 * 1.e-3,  # Tonne year
        'cut_eff': 0.85,  # p. 11, right column
        'nr_eff': 0.85,  # p. 11, left column NOTE: ER type!
        'E_thr': 40. / 1e3,  # table VIII, Enr
        "location": "SNOLAB",
        'res': det_res_superCDMS10,  # table I
        'bg_func': migdal_background_superCDMS_Ge_HV,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    'Ge_migd_HV': {
        'material': 'Ge',
        'type': 'migdal',
        'exp': 44 * 1.e-3,  # Tonne year
        'cut_eff': 0.85,  # p. 11, right column
        'nr_eff': 0.5,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr': 100. / 1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS10,  # table I
        'bg_func': migdal_background_superCDMS_Ge_HV,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    # --- Si HV bg --- #
    'Ge_HV_Si': {
        'material': 'Si',
        'type': 'SI',
        'exp': 9.6 * 1.e-3,  # Tonne year
        # https://www.slac.stanford.edu/exp/cdms/ScienceResults/Publications/PhysRevD.95.082002.pdf
        'cut_eff': 0.85,  # p. 11, right column
        'nr_eff': 0.85,  # p. 11, left column NOTE: ER type!
        'E_thr': 78. / 1e3,  # table VIII, Enr
        "location": "SNOLAB",
        'res': det_res_superCDMS5,  # table I
        'bg_func': migdal_background_superCDMS_Si_HV,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    'Ge_migd_HV_Si': {
        'material': 'Si',
        'type': 'migdal',
        'exp': 9.6 * 1.e-3,  # Tonne year
        # https://www.slac.stanford.edu/exp/cdms/ScienceResults/Publications/PhysRevD.95.082002.pdf
        'cut_eff': 0.85,  # p. 11, right column
        'nr_eff': 0.675,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr': 100. / 1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS5,  # table I
        'bg_func': migdal_background_superCDMS_Si_HV,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    'Xe_migd': {
        'material': 'Xe',
        'type': 'migdal',
        'exp': 20,  # https://arxiv.org/pdf/2007.08796.pdf

        # Combined cut & detection efficiency as in
        # https://arxiv.org/pdf/2007.08796.pdf
        'cut_eff': 0.82,
        'nr_eff': 1,

        'E_thr': 1.0,  # assume https://arxiv.org/abs/2006.09721
        'location': "XENON",
        'res': det_res_XENON1T,  # table I
        'bg_func': er_background_xe,
        'E_max': 5,
        'n_energy_bins': 50,
    },
    'Xe': {
        'material': 'Xe',
        'type': 'SI',
        'exp': 20,  # https://arxiv.org/pdf/2007.08796.pdf

        # Combined cut & detection efficiency as in
        # https://arxiv.org/pdf/2007.08796.pdf
        'cut_eff': 0.82,
        'nr_eff': 1,

        'E_thr': 1.0,  # assume https://arxiv.org/abs/2006.09721
        'location': "XENON",
        'res': det_res_XENON1T,  # table I
        'bg_func': nr_background_xe,
        'E_max': 5,
        'n_energy_bins': 50,
    },
}
# And calculate the effective exposure for each
for name in experiment:
    experiment[name]['exp_eff'] = (experiment[name]['exp'] *
                                   experiment[name]['cut_eff'] *
                                   experiment[name]['nr_eff'])
    experiment[name]['name'] = name
    if 'E_min' not in experiment[name]:
        experiment[name]['E_min'] = 0

# Make a new experiment that is a placeholder for the CombinedInference class.
experiment['Combined'] = {'type': 'combined'}


@numba.njit
def _smear_signal(rate, energy, sigma, bin_width):
    """

    :param rate: counts/bin
    :param energy: energy bin_center
    :param sigma: energy resolution
    :param bin_width: should be scalar of the bin width
    :return: the rate smeared with the specified energy resolution at given
    energy

    This function takes a binned DM-spectrum and takes into account the energy
    resolution of the detector. The rate, energy and resolution should be arrays
    of equal length. The the bin_width
    """
    result = []
    # pylint: disable=consider-using-enumerate
    for i in range(len(energy)):
        res = 0.
        # pylint: disable=consider-using-enumerate
        for j in range(len(rate)):
            # see formula (5) in https://arxiv.org/abs/1012.3458
            res = res + (bin_width * rate[j] *
                         (1. / (np.sqrt(2. * np.pi) * sigma[j])) *
                         np.exp(-(((energy[i] - energy[j]) ** 2.) / (2. * sigma[j] ** 2.)))
                         )
            # TODO
            #  # at the end of the spectrum the bg-rate drops as the convolution does
            #  # not take into account the higher energies.
            #  weight = length / (j-length)
            #  res = res * weight
        result.append(res)
    return np.array(result)


def smear_signal(rate, energy, sigma, bin_width):
    if np.max(sigma) < bin_width:
        # print(f'Resolution {np.mean(sigma)} better than bin_width {bin_width}!')
        return rate
    return _smear_signal(rate, energy, sigma, bin_width)


class DetectorSpectrum(dddm.recoil_rates.spectrum.GenSpectrum):
    add_background = False
    required_detector_fields = 'name material type exp_eff exp exp_eff E_thr res'.split()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'bg_func' in self.config:
            self.add_background = kwargs.get('add_background', True)
        else:
            self.log.debug('No bg_func in experiment config')

    def __str__(self):
        return 'DetectorSpectrum class inherited from GenSpectrum.\nSmears spectrum with detector resolution and implements the energy threshold for the detector'

    def get_events(self):
        """
        :return: Events (binned)
        """
        return self.compute_detected_spectrum()

    def compute_detected_spectrum(self):
        """

        :return: spectrum taking into account the detector properties
        """
        # get the spectrum
        rates = self.spectrum_simple([self.mw, self.sigma_nucleon])

        if self.add_background:
            # pay close attention, the events in the bg_func are already taking into
            # account the det. efficiency et cetera. Hence the number here should be
            # multiplied by the total exposure (rather than the effective exposure that
            # is multiplied by at the end of this subroutine. Hence the bg rates obtained
            # from that function is multiplied by the ratio between the two.
            rates += self.config['bg_func'](self.E_min,
                                            self.E_max,
                                            self.n_bins) * (
                self.config['exp'] / self.config['exp_eff'])
        e_bin_edges = np.array(self.get_bin_edges())
        e_bin_centers = np.mean(e_bin_edges, axis=1)
        bin_width = np.mean(np.diff(e_bin_centers))

        # Smear the rates with the detector resolution
        sigma = self.config['res'](e_bin_centers)
        rates = np.array(smear_signal(rates, e_bin_centers, sigma, bin_width))

        # Set the rate to zero for energies smaller than the threshold
        rates = self.above_threshold(rates, e_bin_edges, self.config['E_thr'])

        # Calculate the total number of events per bin
        rates = rates * bin_width * self.config['exp_eff']
        return rates

    @staticmethod
    @numba.njit
    def above_threshold(rates, e_bin_edges, e_thr):
        """
        Apply threshold to the rates. We are right edge inclusive
        bin edges : |bin0|bin1|bin2|
        e_thr     :        |
        bin0 -> 0
        bin1 -> fraction of bin1 > e_thr
        bin2 -> full content

        :param rates: bins with the number of counts
        :param e_bin_edges: 2d array of the left, right bins
        :param e_thr: energy threshold
        :return: rates with energy threshold applied
        """
        for r_i, r in enumerate(rates):
            left_edge, right_edge = e_bin_edges[r_i]
            if left_edge >= e_thr:
                # From now on all the bins will be above threshold we don't
                # have to set to 0 anymore
                break
            if right_edge <= e_thr:
                # this bin is fully below threshold
                rates[r_i] = 0
                continue
            elif e_thr >= left_edge and e_thr <= right_edge:
                fraction_above = (right_edge - e_thr) / (right_edge - left_edge)
                rates[r_i] = r * fraction_above
            else:
                print(left_edge, right_edge, e_thr)
                raise ValueError('How did this happen?')

        return rates
