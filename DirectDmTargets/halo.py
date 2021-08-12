"""For a given detector get a WIMPrate for a given detector (not taking into
account any detector effects"""

import logging
import os
import shutil

import numericalunits as nu
import numpy as np
import pandas as pd
import verne
import wimprates as wr
from DirectDmTargets import utils
from DirectDmTargets.context import context
from scipy.interpolate import interp1d

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class GenSpectrum:
    def __init__(self, mw, sig, model, det):
        """
        :param mw: wimp mass
        :param sig: cross-section of the wimp nucleon interaction
        :param model: the dark matter model
        :param det: detector name
        """
        assert isinstance(
            det, dict), "Invalid detector type. Please provide dict."
        self.mw = mw  # note that this is not in log scale!
        self.sigma_nucleon = sig  # note that this is not in log scale!
        self.dm_model = model
        self.experiment = det

        self.n_bins = 10
        if self.experiment['type'] in ['SI', 'SI_bg']:
            self.E_min = 0  # keV
            self.E_max = 100  # keV
        elif self.experiment['type'] in ['migdal', 'migdal_bg']:
            self.E_min = 0  # keV
            self.E_max = 10  # keV
        else:
            raise NotImplementedError(
                f'Exp. type {self.experiment["type"]} is unknown')

        if 'E_min' in self.experiment:
            self.E_min = self.experiment['E_min']
        if 'E_max' in self.experiment:
            self.E_max = self.experiment['E_max']

    def __str__(self):
        """
        :return: info
        """
        return f"spectrum_simple of a DM model ({self.dm_model}) in a " \
               f"{self.experiment['name']} detector"

    def get_bin_edges(self):
        return utils.get_bins(
            self.E_min,
            self.E_max,
            self.n_bins),

    def get_bin_centers(self):
        return np.mean(self.get_bin_edges(),
                       axis=1)

    def spectrum_simple(self, benchmark):
        """
        :param benchmark: insert the kind of DM to consider (should contain Mass
         and cross-section)
        :return: returns the rate
        """
        if not isinstance(benchmark, (dict, pd.DataFrame)):
            benchmark = {'mw': benchmark[0], 'sigma_nucleon': benchmark[1]}

        try:
            kwargs = {'material': self.experiment['material']}
        except KeyError as e:
            raise KeyError(f'Invalid experiment {self.experiment}') from e
        if self.experiment['type'] in ['SI', 'SI_bg']:
            rate = wr.rate_wimp_std(self.get_bin_centers(),
                                    benchmark["mw"],
                                    benchmark["sigma_nucleon"],
                                    halo_model=self.dm_model,
                                    **kwargs
                                    )
        elif self.experiment['type'] in ['migdal', 'migdal_bg']:
            # This integration takes a long time, hence, we will lower the
            # default precision of the scipy dblquad integration
            migdal_integration_kwargs = dict(epsabs=1e-4,
                                             epsrel=1e-4)
            convert_units = (nu.keV * (1000 * nu.kg) * nu.year)
            rate = convert_units * wr.rate_migdal(
                self.get_bin_centers() * nu.keV,
                benchmark["mw"] * nu.GeV / nu.c0 ** 2,
                benchmark["sigma_nucleon"] * nu.cm ** 2,
                # TODO should this be different for the different experiments?
                q_nr=0.15,
                halo_model=self.dm_model,
                material=self.experiment['material'],
                **migdal_integration_kwargs
            )
        else:
            raise NotImplementedError(
                f'No type of matching {self.experiment["type"]} interactions.')
        return rate

    def get_events(self):
        """
        :return: Events (binned)
        """
        assert self.experiment != {}, "First enter the parameters of the detector"
        rate = self.spectrum_simple([self.mw, self.sigma_nucleon])
        bin_width = np.diff(
            utils.get_bins(
                self.E_min,
                self.E_max,
                self.n_bins),
            axis=1)[
            :,
            0]
        events = rate * bin_width * self.experiment['exp_eff']
        return events

    def get_poisson_events(self):
        """
        :return: events with poisson noise
        """
        return np.random.exponential(self.get_events()).astype(np.float)

    def get_data(self, poisson=True):
        """

        :param poisson: type bool, add poisson True or False
        :return: pd.DataFrame containing events binned in energy
        """
        result = pd.DataFrame()
        if poisson:
            result['counts'] = self.get_poisson_events()
        else:
            result['counts'] = self.get_events()
        result['bin_centers'] = self.get_bin_centers()
        bins = utils.get_bins(self.E_min, self.E_max, self.n_bins)
        result['bin_left'] = bins[:, 0]
        result['bin_right'] = bins[:, 1]
        result = self.set_negative_to_zero(result)
        return result

    @staticmethod
    def set_negative_to_zero(result):
        mask = result['counts'] < 0
        if np.any(mask):
            log.warning(
                '\n\n----\nWARNING::\nfinding negative rates. Doing hard override!!\n----\n\n')
            result['counts'][mask] = 0
            return result
        return result


class SHM:
    """
        class used to pass a halo model to the rate computation
        must contain:
        :param v_esc -- escape velocity (multiplied by units)
        :param rho_dm -- density in mass/volume of dark matter at the Earth (multiplied by units)
        The standard halo model also allows variation of v_0
        :param v_0 -- v0 of the velocity distribution (multiplied by units)
        :function velocity_dist -- function taking v,t giving normalised
        velocity distribution in earth rest-frame.
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        self.v_0 = 230 * nu.km / nu.s if v_0 is None else v_0
        self.v_esc = 544 * nu.km / nu.s if v_esc is None else v_esc
        self.rho_dm = (0.3 * nu.GeV / nu.c0 ** 2 / nu.cm ** 3
                       if rho_dm is None else rho_dm)

    def __str__(self):
        # Standard Halo Model (shm)
        return 'shm'

    def velocity_dist(self, v, t):
        """
        Get the velocity distribution in units of per velocity,
        :param v: v is in units of velocity
        :return: observed velocity distribution at earth
        """
        return wr.observed_speed_dist(v, t, self.v_0, self.v_esc)


class VerneSHM:
    """
        class used to pass a halo model to the rate computation based on the
        earth shielding effect as calculated by Verne
        must contain:
        :param v_esc -- escape velocity (multiplied by units)
        :param rho_dm -- density in mass/volume of dark matter at the Earth (multiplied by units)
        The standard halo model also allows variation of v_0
        :param v_0 -- v0 of the velocity distribution (multiplied by units)
        :function velocity_dist -- function taking v,t giving normalised
        velocity distribution in earth rest-frame.
       """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None,
                 log_cross_section=None, log_mass=None, location=None):
        # This may seem somewhat counter intuitive. But we want to use similar
        # input as to SHM (see above) to that end, we here divide the input by
        # the respective units
        self.v_0_nodim = 230 if v_0 is None else v_0 / (nu.km / nu.s)
        self.v_esc_nodim = 544 if v_esc is None else v_esc / (nu.km / nu.s)
        self.rho_dm_nodim = 0.3 if rho_dm is None else rho_dm / \
            (nu.GeV / nu.c0 ** 2 / nu.cm ** 3)

        # Here we keep the units dimensionful as these parameters are requested
        # by wimprates and therefore must have dimensions
        self.v_0 = self.v_0_nodim * nu.km / nu.s
        self.v_esc = self.v_esc_nodim * nu.km / nu.s
        self.rho_dm = self.rho_dm_nodim * nu.GeV / nu.c0 ** 2 / nu.cm ** 3

        # in contrast to the SHM, the earth shielding does need the mass and
        # cross-section to calculate the rates.
        self.log_cross_section = -35 if log_cross_section is None else log_cross_section
        self.log_mass = 0 if log_mass is None else log_mass
        self.location = "XENON" if location is None else location

        # Combine the parameters into a single naming convention. This is were
        # we will save/read the velocity distribution (from).
        self.fname = os.path.join(
            'f_params',
            f'loc_{str(self.location)}',
            f'v0_{int(self.v_0_nodim)}',
            f'vesc_{int(self.v_esc_nodim)}',
            f'rho_{self.rho_dm_nodim:.3f}',
            f'sig_{self.log_cross_section:.1f}_mx_{self.log_mass:.2f}'
        )

        self.itp_func = None

    def __str__(self):
        # The standard halo model observed at some location shielded from strongly
        # interacting DM by overburden (rock atmosphere)
        return 'shielded_shm'

    def load_f(self):
        """
        load the velocity distribution. If there is no velocity
            distribution shaved, load one.
        :return:
        """

        # set up folders and names
        file_folder = context['verne_files']
        file_name = os.path.join(file_folder, self.fname + '_avg' + '.csv')
        utils.check_folder_for_file(os.path.join(file_folder, self.fname))

        # Convert file_name and self.fname to folder and name of csv file where
        # to save.
        temp_file_name = utils.add_temp_to_csv(file_name)
        exist_csv = os.path.exists(file_name)
        assertion_string = f'abs file {temp_file_name} should be a string\n'
        assertion_string += f'exists csv {exist_csv} should be a bool'
        log.info(f'load_f::\twrite to {file_name} ({not exist_csv}). '
                 f'Then copy to {temp_file_name}')
        assert (isinstance(temp_file_name, str) and
                isinstance(exist_csv, bool)), assertion_string
        if not exist_csv:
            log.info(f'Using {file_name} for the velocity distribution. '
                     f'Writing to {temp_file_name}')
            df = verne.CalcVelDist.avg_calcveldist(
                m_x=10. ** self.log_mass,
                sigma_p=10. ** self.log_cross_section,
                loc=self.location,
                v_esc=self.v_esc_nodim,
                v_0=self.v_0_nodim,
                N_gamma=4,
            )

            if not os.path.exists(file_name):
                log.info(f'writing to {temp_file_name}')
                df.to_csv(temp_file_name, index=False)
                if not os.path.exists(file_name):
                    log.info(f'moving {temp_file_name} to {file_name}')
                    shutil.move(temp_file_name, file_name)
            else:
                log.warning(f'while writing {temp_file_name}, {file_name} was created')
        else:
            log.info(f'Using {file_name} for the velocity distribution')
            try:
                df = pd.read_csv(file_name)
            except pd.io.common.EmptyDataError as pandas_error:
                os.remove(file_name)
                raise pandas_error
        # Alright now load the data and interpolate that. This is the output
        # that wimprates need
        if not os.path.exists(os.path.abspath(file_name)):
            raise OSError(f'{file_name} should exist. Is there anything at {temp_file_name}')

        if not len(df):
            # Somehow we got an empty dataframe, we cannot continue
            os.remove(file_name)
            raise ValueError(
                f'Was trying to read an empty dataframe from {file_name}:\n{df}')

        x, y = df.keys()
        interpolation = interp1d(
            df[x] * (nu.km / nu.s), df[y] * (nu.s / nu.km), bounds_error=False, fill_value=0)

        def velocity_dist(v_, t_):
            # Wimprates needs to have a two-parameter function. However since we
            # ignore time for now. We make this makeshift transition from a one
            # parameter function to a two parameter function
            return interpolation(v_)

        self.itp_func = velocity_dist

    def velocity_dist(self, v, t):
        """
        Get the velocity distribution in units of per velocity,
        :param v: v is in units of velocity
        :return: observed velocity distribution at earth
        """
        if self.itp_func is None:
            self.load_f()
        return self.itp_func(v, t)
