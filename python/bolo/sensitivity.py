""" Sensitivity calculation """
import sys

import numpy as np

from collections import OrderedDict as odict

from cfgmdl import Model
from . import physics, noise
from .unit import Unit
from .cfg import Output


def _Trj_over_Tcmb(freqs):
    """ Convert to RJ temperature from CMB temperature """
    factor_spec = physics.Trj_over_Tb(freqs, physics.Tcmb)
    bw = freqs[-1] - freqs[0]
    return np.trapz(factor_spec, freqs)/bw


def bcast_list(array_list):
    """ Broadcast a list of arrays to a single shape """
    bcast = np.broadcast(*array_list)
    return np.array([np.array(v) for v in bcast]).T.reshape([bcast.numiter] + list(bcast.shape))


NSKY_SRC = 4

class Sensitivity(Model): #pylint: disable=too-many-instance-attributes
    """ Sensitivity calculation
    """
    effic = Output()
    opt_power = Output(unit=Unit('pW'))
    tel_rj_temp = Output(unit=Unit('K'))
    sky_rj_temp = Output(unit=Unit('K'))

    NEP_bolo = Output(unit=Unit('aW/rtHz'))
    NEP_read = Output(unit=Unit('aW/rtHz'))
    NEP_ph = Output(unit=Unit('aW/rtHz'))
    NEP_ph_corr = Output(unit=Unit('aW/rtHz'))
    NEP = Output(unit=Unit('aW/rtHz'))
    NEP_corr = Output(unit=Unit('aW/rtHz'))

    NET = Output(unit=Unit('uK-rts'))
    NET_corr = Output(unit=Unit('uK-rts'))

    NET_RJ = Output(unit=Unit('uK-rts'))
    NET_corr_RJ = Output(unit=Unit('uK-rts'))

    NET_arr = Output(unit=Unit('uK-rts'))
    NET_arr_RJ = Output(unit=Unit('uK-rts'))

    corr_fact = Output()
    map_depth = Output(unit=Unit('uK-amin'))
    map_depth_RJ = Output(unit=Unit('uK-amin'))

    def __init__(self, channel):
        """ Constructor """

        super(Sensitivity, self).__init__()

        self._channel = channel
        self._camera = self._channel.camera
        self._instrument = self._camera.instrument
        self._summary = None

        self._freqs = channel.freqs
        self._bandwidth = channel.bandwidth
        self._fsky = self._instrument.sky_fraction
        self._obs_time = self._instrument.obs_time

        self._obs_effic = np.expand_dims(self._instrument.obs_effic, -1)

        self.temps_list = channel.sky_temps + channel.optical_temps + [channel._det_temp]
        self._temps = bcast_list(self.temps_list)
        self.trans_list = channel.sky_effic + channel.optical_effic + [channel._det_effic]
        self._trans = bcast_list(self.trans_list)
        self.emiss_list = channel.sky_emiss + channel.optical_emiss + [channel._det_emiss]
        self._emiss = bcast_list(self.emiss_list)

        self._emiss = self._emiss.reshape((self._emiss.shape[0], self._emiss.shape[1], 1, self._emiss.shape[2]))

        self._chan_effic = np.product(self._trans, axis=0)
        self._elem_effic = np.cumprod(self._trans[::-1], axis=0)

        self._elem_power_by_freq = physics.bb_pow_spec(self._freqs, self._temps, self._emiss*self._elem_effic)
        self._elem_power = np.trapz(self._elem_power_by_freq, self._freqs)
        self._effic = np.trapz(self._chan_effic, self._freqs) / self._bandwidth
        self._tel_effic = np.trapz(self._elem_effic[NSKY_SRC], self._freqs) / self._bandwidth

        self._opt_power = np.sum(self._elem_power, axis=0)
        self._tel_power = np.sum(self._elem_power[NSKY_SRC:], axis=0)
        self._sky_power = np.sum(self._elem_power[:NSKY_SRC], axis=0)

        self._tel_rj_temp = physics.rj_temp(self._tel_power, self._bandwidth, self._tel_effic)
        self._sky_rj_temp = physics.rj_temp(self._sky_power, self._bandwidth, self._tel_effic)

        self._NEP_bolo = self._channel.bolo_NEP(self._opt_power)
        self._NEP_ph, self._NEP_ph_corr = self._channel.photon_NEP(self._elem_power_by_freq)

        self._NEP_read = self._channel.read_NEP(self._opt_power)
        if self._NEP_read is None:
            self._NEP_read = np.sqrt((1 + self._channel.read_frac)**2 - 1.)*np.sqrt(self._NEP_bolo**2 + self._NEP_ph**2)
        self._NEP = np.sqrt(self._NEP_bolo**2 + self._NEP_ph**2 + self._NEP_read**2)
        self._NEP_corr = np.sqrt(self._NEP_bolo**2 + self._NEP_ph_corr**2 + self._NEP_read**2)

        self._NET = noise.NET_from_NEP(self._NEP, self._freqs, self._chan_effic, self._channel.camera.optical_coupling)
        self._NET_corr = noise.NET_from_NEP(self._NEP_corr, self._freqs, self._chan_effic, self._channel.camera.optical_coupling)

        self._Trj_over_Tcmb = _Trj_over_Tcmb(self._freqs)
        self._NET_RJ = self._Trj_over_Tcmb*self._NET
        self._NET_corr_RJ = self._Trj_over_Tcmb*self._NET_corr

        self._NET_arr = self._instrument.NET * noise.NET_arr(self._NET, self._channel.ndet, self._channel.Yield)
        self._NET_arr_RJ = self._instrument.NET * noise.NET_arr(self._NET_RJ, self._channel.ndet, self._channel.Yield)

        self._corr_fact = self._NET_corr / self._NET
        self._map_depth = noise.map_depth(self._NET_arr, self._fsky, self._obs_time, self._obs_effic)
        self._map_depth_RJ = noise.map_depth(self._NET_arr_RJ, self._fsky, self._obs_time, self._obs_effic)
        self.summarize()

    def summarize(self):
        """ Compute and cache summary statistics """
        self._summary = odict()
        for key, prop in self._properties.items():
            self._summary[key] = prop.summarize(self)
        return self._summary


    def print_summary(self, stream=sys.stdout):
        """ Print summary statistics in human-readable format """
        for key, val in self._summary.items():
            stream.write("%s : %s\n" % (key.ljust(20), val))

    def make_sims_table(self, name, table_dict):
        """ Make a table with per-simulation parameters """
        o_dict = odict([(key, prop.__get__(self).flatten()) for key, prop in self._properties.items()])
        try:
            return table_dict.make_datatable(name, o_dict)
        except ValueError as msg:
            for k, v in o_dict.items():
                print (k, v.size)
            raise ValueError(msg)


    def make_sum_table(self, name, table_dict):
        """ Make a table with summary parameters """
        o_dict = odict()
        for val in self._summary.values():
            o_dict.update(val.todict())

        return table_dict.make_datatable(name, o_dict)


    def make_tables(self, base_name, table_dict, save_summary, save_sim):
        """ Make output tables """
        if save_sim:
            self.make_sims_table("%s_sims" % base_name, table_dict)
        if save_summary:
            self.make_sum_table("%s_summary" % base_name, table_dict)
        return table_dict
