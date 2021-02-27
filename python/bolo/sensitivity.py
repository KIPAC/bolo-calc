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
    try:
        bcast = np.broadcast(*array_list)
    except ValueError as msg:
        for arr in array_list:
            print(np.shape(arr))
        raise ValueError(msg)
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
        self._fsky = self._instrument.sky_fraction()
        self._obs_time = self._instrument.obs_time()
        self._obs_effic = self._instrument.obs_effic()
        
        self.temps_list = channel.sky_temps + channel.optical_temps + [channel._det_temp]
        self._temps = bcast_list(self.temps_list)
        self.trans_list = channel.sky_effic + channel.optical_effic + [channel._det_effic]
        self._trans = bcast_list(self.trans_list)
        self.emiss_list = channel.sky_emiss + channel.optical_emiss + [channel._det_emiss]
        self._emiss = bcast_list(self.emiss_list)

        #print(np.shape(self._temps), np.shape(self._trans), np.shape(self._emiss))
        if len(self._emiss.shape) == 2:
            self._emiss = self._emiss.reshape((self._emiss.shape[0], 1, 1, self._emiss.shape[1]))
        elif len(self._emiss.shape) == 3:
            self._emiss = self._emiss.reshape((self._emiss.shape[0], 1, self._emiss.shape[1], self._emiss.shape[2]))
                         
        self._chan_effic = np.product(self._trans, axis=0)
        self._elem_effic = np.cumprod(self._trans[::-1], axis=0)

        self._elem_power_by_freq = physics.bb_pow_spec(self._freqs, self._temps, self._emiss*self._elem_effic)
        self._elem_power = np.trapz(self._elem_power_by_freq, self._freqs)

        self.effic.value = np.trapz(self._chan_effic, self._freqs) / self._bandwidth
        self._tel_effic = np.trapz(self._elem_effic[NSKY_SRC], self._freqs) / self._bandwidth

        self.opt_power.value = np.sum(self._elem_power, axis=0)
        self._tel_power = np.sum(self._elem_power[NSKY_SRC:], axis=0)
        self._sky_power = np.sum(self._elem_power[:NSKY_SRC], axis=0)

        self.tel_rj_temp = physics.rj_temp(self._tel_power, self._bandwidth, self._tel_effic)
        self.sky_rj_temp = physics.rj_temp(self._sky_power, self._bandwidth, self._tel_effic)

        self.NEP_bolo.value = self._channel.bolo_NEP(self.opt_power.value)
        self.NEP_ph.value, self.NEP_ph_corr = self._channel.photon_NEP(self._elem_power_by_freq)
        
        self.NEP_read.value = self._channel.read_NEP(self.opt_power.value)
        if self.NEP_read is None or not np.isfinite(self.NEP_read.value).all():
            self.NEP_read.value = np.sqrt((1 + self._channel.read_frac())**2 - 1.)*np.sqrt(self.NEP_bolo.value**2 + self._NEP_ph.value**2)
        self.NEP.value = np.sqrt(self.NEP_bolo.value**2 + self._NEP_ph.value**2 + self._NEP_read.value**2)
        self.NEP_corr.value = np.sqrt(self._NEP_bolo.value**2 + self._NEP_ph_corr.value**2 + self._NEP_read.value**2)

        self.NET.value = noise.NET_from_NEP(self.NEP.value, self._freqs, self._chan_effic, self._channel.camera.optical_coupling())
        self.NET_corr.value = noise.NET_from_NEP(self.NEP_corr.value, self._freqs, self._chan_effic, self._channel.camera.optical_coupling())

        self._Trj_over_Tcmb = _Trj_over_Tcmb(self._freqs)
        self.NET_RJ.value = self._Trj_over_Tcmb*self._NET.value
        self.NET_corr_RJ.value = self._Trj_over_Tcmb*self._NET_corr.value

        self.NET_arr.value = self._instrument.NET() * noise.NET_arr(self.NET.value, self._channel.ndet, self._channel.Yield())
        self.NET_arr_RJ.value = self._instrument.NET() * noise.NET_arr(self.NET_RJ.value, self._channel.ndet, self._channel.Yield())

        self.corr_fact.value = self.NET_corr.value / self.NET.value
        self.map_depth.value = noise.map_depth(self.NET_arr.value, self._fsky, self._obs_time, self._obs_effic)
        self.map_depth_RJ.value = noise.map_depth(self.NET_arr_RJ.value, self._fsky, self._obs_time, self._obs_effic)
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
        o_dict = odict([(key, prop.__get__(self).from_SI.flatten()) for key, prop in self._properties.items()])
        try:
            return table_dict.make_datatable(name, o_dict)
        except ValueError as msg:
            for k, v in o_dict.items():
                print(k, v.size)
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
