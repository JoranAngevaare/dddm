import os
import shutil
import numericalunits as nu
import pandas as pd
from dddm import utils, exporter
import warnings
from scipy.interpolate import interp1d
import numpy as np

export, __all__ = exporter()


@export
class ShieldedSHM:
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

    def __init__(self,
                 location,
                 file_folder='./verne_files',
                 v_0=None,
                 v_esc=None,
                 rho_dm=None,
                 log_cross_section=None,
                 log_mass=None,
                 ):
        v_0_nodim = 230 if v_0 is None else v_0 / (nu.km / nu.s)
        v_esc_nodim = 544 if v_esc is None else v_esc / (nu.km / nu.s)
        rho_dm_nodim = (0.3 if rho_dm is None else
                        rho_dm / (nu.GeV / nu.c0 ** 2 / nu.cm ** 3))

        # Here we keep the units dimensionful as these parameters are requested
        # by wimprates and therefore must have dimensions
        self.v_0 = v_0_nodim * nu.km / nu.s
        self.v_esc = v_esc_nodim * nu.km / nu.s
        self.rho_dm = rho_dm_nodim * nu.GeV / nu.c0 ** 2 / nu.cm ** 3

        assert np.isclose(self.v_0_nodim, v_0_nodim), (self.v_0_nodim, v_0_nodim)
        assert np.isclose(self.v_esc_nodim, v_esc_nodim), (self.v_esc_nodim, v_esc_nodim)
        assert np.isclose(self.rho_dm_nodim, rho_dm_nodim), (self.rho_dm_nodim, rho_dm_nodim)

        # in contrast to the SHM, the earth shielding does need the mass and
        # cross-section to calculate the rates.
        self.log_cross_section = -35 if log_cross_section is None else log_cross_section
        self.log_mass = 0 if log_mass is None else log_mass
        self.location = "XENON" if location is None else location

        # Combine the parameters into a single naming convention. This is were
        # we will save/read the velocity distribution (from).
        self.fname = os.path.join(
            'f_params',
            f'loc_{self.location}',
            f'v0_{int(self.v_0_nodim)}',
            f'vesc_{int(self.v_esc_nodim)}',
            f'rho_{self.rho_dm_nodim:.3f}',
            f'sig_{self.log_cross_section:.1f}_mx_{self.log_mass:.2f}',
        )

        self.itp_func = None
        self.log = utils.get_logger(self.__class__.__name__)
        self.file_folder = file_folder

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
        import verne
        # set up folders and names
        file_folder = self.file_folder
        file_name = os.path.join(file_folder, self.fname + '_avg' + '.csv')
        utils.check_folder_for_file(os.path.join(file_folder, self.fname))

        # Convert file_name and self.fname to folder and name of csv file where
        # to save.
        temp_file_name = utils.add_temp_to_csv(file_name)
        exist_csv = os.path.exists(file_name)
        assertion_string = f'abs file {temp_file_name} should be a string\n'
        assertion_string += f'exists csv {exist_csv} should be a bool'
        self.log.info(f'load_f::\twrite to {file_name} ({not exist_csv}). '
                      f'Then copy to {temp_file_name}')
        assert (isinstance(temp_file_name, str) and
                isinstance(exist_csv, bool)), assertion_string
        if not exist_csv:
            self.log.info(f'Using {file_name} for the velocity distribution. '
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
                self.log.info(f'writing to {temp_file_name}')
                df.to_csv(temp_file_name, index=False)
                if not os.path.exists(file_name):
                    self.log.info(f'moving {temp_file_name} to {file_name}')
                    shutil.move(temp_file_name, file_name)
            else:
                self.log.warning(f'while writing {temp_file_name}, {file_name} was created')
        else:
            self.log.info(f'Using {file_name} for the velocity distribution')
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

    def parameter_dict(self):
        """Return a dict of readable parameters of the current settings"""
        return dict(
            v_0=self.v_0_nodim,
            v_esc=self.v_esc_nodim,
            rho_dm=self.rho_dm_nodim,
            log_cross_section=self.log_cross_section,
            log_mass=self.log_mass,
            location=self.location,
        )

    @property
    def v_0_nodim(self):
        return self.v_0 / (nu.km / nu.s)

    @property
    def v_esc_nodim(self):
        return self.v_esc / (nu.km / nu.s)

    @property
    def rho_dm_nodim(self):
        return self.rho_dm / (nu.GeV / nu.c0 ** 2 / nu.cm ** 3)


class VerneSHM(ShieldedSHM):
    def __init__(self, *args, **kwargs):
        warnings.warn("Use ShieldedSHM instead of VerneSHM", DeprecationWarning)
        super().__init__(*args, **kwargs)
