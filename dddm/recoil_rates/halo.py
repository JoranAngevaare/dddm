"""
For a given detector get a WIMPrate for a given detector (not taking into
account any detector effects
"""

import numericalunits as nu
import wimprates as wr
import dddm

export, __all__ = dddm.exporter()


@export
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

    def parameter_dict(self):
        """Return a dict of readable parameters of the current settings"""
        return dict(
            v_0=self.v_0 / (nu.km / nu.s),
            v_esc=self.v_esc / (nu.km / nu.s),
            rho_dm=self.rho_dm / (nu.GeV / nu.c0 ** 2 / nu.cm ** 3),
        )
