""" Sensitivity calculation """
import sys
import numpy as np
import pdb

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
    P_sat = Output(unit=Unit('pW'))
    G = Output(unit=Unit('pW/K'))
    Flink = Output()

    tel_power = Output(unit=Unit('pW'))
    sky_power = Output(unit=Unit('pW'))

    tel_rj_temp = Output(unit=Unit('K'))
    sky_rj_temp = Output(unit=Unit('K'))

    elem_effic = Output()
    elem_cumul_effic = Output()
    elem_power_from_sky = Output(unit=Unit('pW'))
    elem_power_to_det = Output(unit=Unit('pW'))

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


    #summary_fields = ['effic', 'opt_power', 'tel_rj_temp', 'sky_rj_temp', 'NEP_bolo', 'NEP_read', 'NEP_ph', 'NEP_ph_corr', 'NEP', 'NEP_corr',
    #                      'NET', 'NET_corr', 'NET_RJ', 'NET_corr_RJ', 'NET_arr', 'NET_arr_RJ',  'NET_arr_RJ', 'map_depth', 'map_depth_RJ']
    summary_fields = ['effic', 'opt_power', 'P_sat', 'Flink', 'G', 'tel_rj_temp', 'sky_rj_temp', 'NEP_bolo', 'NEP_read', 'NEP_ph', 'NEP',
                          'NET', 'NET_corr','corr_fact', 'NET_arr']

    optical_output_fields = ['elem_effic', 'elem_cumul_effic', 'elem_power_from_sky', 'elem_power_to_det']

    def __init__(self, channel):
        """ Constructor """

        super(Sensitivity, self).__init__()

        self._channel = channel
        self._camera = self._channel.camera
        self._instrument = self._camera.instrument
        self._channel_name = "%s_%i" % (self._camera.name, self._channel.idx)
        self._summary = None
        self._optical_output = None

        self._freqs = channel.freqs
        self._bandwidth = channel.bandwidth
        self._fsky = self._instrument.sky_fraction()
        self._obs_time = self._instrument.obs_time()
        self._obs_effic = self._instrument.obs_effic()

        self.temps_list = channel.sky_temps + channel.optical_temps + [channel._det_temp]
        self._temps = bcast_list(self.temps_list)

        # Buffer both sides of the transmisison array, because we will be taking cumulatimve products that are offset by one
        # (i.e., we want the product of all the elements downstream of a particular element)
        self.trans_list = [0.] + channel.sky_effic + channel.optical_effic + [channel._det_effic] + [1.]
        self._trans = bcast_list(self.trans_list)
        self.emiss_list = channel.sky_emiss + channel.optical_emiss + [channel._det_emiss]
        self._emiss = bcast_list(self.emiss_list)

        self._elem_names = channel.sky_names + list(self._camera.optics.keys()) + ['detector']
        self._ap_names = list(self._instrument.optics.apertureStops.keys())


        # Fix the shapes of the arrays to match
        if len(self._emiss.shape) == 2:
            self._emiss = self._emiss.reshape((self._emiss.shape[0], 1, 1, self._emiss.shape[1]))
        elif len(self._emiss.shape) == 3:
            self._emiss = self._emiss.reshape((self._emiss.shape[0], 1, self._emiss.shape[1], self._emiss.shape[2]))


        # This total the transmission efficiency of a particular channel as a function of frequency
        # Note that we have to pull out the padding here
        self._chan_effic = np.product(self._trans[1:-1], axis=0)

        # This is the efficiency of a particular element getting to the detector as a function of frequency
        # Note that we have to pull out the padding here
        # Note also that we want to take the cumulative sum starting at the detector side (thus the [::-1] to reverse the iteration direction)
        self._elem_cumul_effic_by_freq = np.cumprod(self._trans[::-1], axis=0)[::-1][2:]

        # This is the power emitted by a particular element
        self._elem_power_by_freq = physics.bb_pow_spec(self._freqs, self._temps, self._emiss)

        # This is the power coming from a particular element getting to the detector as a function of frequency
        self._elem_power_to_det_by_freq = self._elem_power_by_freq*self._elem_cumul_effic_by_freq

        # This is the power coming from up the optical chain getting to a particlar element
        # Note that we have to pull out the padding here
        cumul_power_down = 0.
        cumul_list_down = []
        for elem_power, elem_trans in zip(self._elem_power_by_freq, self._trans[:-2]):
            cumul_power_down = cumul_power_down + elem_power
            cumul_power_down = cumul_power_down * elem_trans
            cumul_list_down.append(cumul_power_down)
        self._elem_sky_power_by_freq = np.array(cumul_list_down)

        # These are integrated accross the bands (using np.trapz to do trapezoid rule integration)
        # Optical power from each element
        self.elem_power_to_det.set_from_SI(np.trapz(self._elem_power_to_det_by_freq, self._freqs))
        self.elem_power_from_sky.set_from_SI(np.trapz(self._elem_sky_power_by_freq, self._freqs))

        # Optical efficiency from each element
        self.elem_effic.set_from_SI(np.trapz(self._trans[1:-1], self._freqs) / self._bandwidth)
        # Cumulative efficiency for the test of the optical chain, by element
        self.elem_cumul_effic.set_from_SI(np.trapz(self._elem_cumul_effic_by_freq, self._freqs) / self._bandwidth)
        # Total channel efficiency
        self.effic.set_from_SI(np.trapz(self._chan_effic, self._freqs) / self._bandwidth)

        # This is the efficiency of all the telescope elements
        self._tel_effic = np.trapz(self._elem_cumul_effic_by_freq[NSKY_SRC], self._freqs) / self._bandwidth

        # From this point, all everything is intergrated across bands and given by channel
        self.opt_power.set_from_SI(np.sum(self.elem_power_to_det.SI, axis=0))

        self.tel_power.set_from_SI(np.sum(self.elem_power_to_det.SI[NSKY_SRC:], axis=0))
        self.sky_power.set_from_SI(np.sum(self.elem_power_from_sky.SI[:NSKY_SRC], axis=0))

        self.tel_rj_temp.set_from_SI(physics.rj_temp(self.tel_power.SI, self._bandwidth, self._tel_effic))
        self.sky_rj_temp.set_from_SI(physics.rj_temp(self.sky_power.SI, self._bandwidth, self._tel_effic))

        self.NEP_bolo.set_from_SI(self._channel.bolo_NEP(self.opt_power.SI))

        nep_np, nep_ph_corr = self._channel.photon_NEP(self._elem_power_to_det_by_freq, self._elem_names, self._ap_names)
        self.NEP_ph.set_from_SI(nep_np)

        self.NEP_ph_corr.set_from_SI(nep_ph_corr)

        if self.NEP_read is None or not np.isfinite(self.NEP_read.SI).all():
            self.NEP_read.set_from_SI(np.sqrt((1 + self._channel.read_frac())**2 - 1.)*np.sqrt(self.NEP_bolo.SI**2 + self.NEP_ph.SI**2))
        else: 
            self.NEP_read.set_from_SI(self._channel.read_NEP(self.opt_power.SI))

        self.NEP.set_from_SI(np.sqrt(self.NEP_bolo.SI**2 + self.NEP_ph.SI**2 + self.NEP_read.SI**2))
        self.NEP_corr.set_from_SI(np.sqrt(self.NEP_bolo.SI**2 + self.NEP_ph_corr.SI**2 + self.NEP_read.SI**2))

        self.NET.set_from_SI(noise.NET_from_NEP(self.NEP.SI, self._freqs, self._chan_effic, self._channel.camera.optical_coupling()))
        self.NET_corr.set_from_SI(noise.NET_from_NEP(self.NEP_corr.SI, self._freqs, self._chan_effic, self._channel.camera.optical_coupling()))

        self._Trj_over_Tcmb = _Trj_over_Tcmb(self._freqs)
        self.NET_RJ.set_from_SI(self._Trj_over_Tcmb*self.NET.SI)
        self.NET_corr_RJ.set_from_SI(self._Trj_over_Tcmb*self.NET_corr.SI)

        self.NET_arr.set_from_SI(self._instrument.NET() * noise.NET_arr(self.NET.SI, self._channel.ndet, self._channel.Yield()))
        self.NET_arr_RJ.set_from_SI(self._instrument.NET() * noise.NET_arr(self.NET_RJ.SI, self._channel.ndet, self._channel.Yield()))

        self.corr_fact.set_from_SI(self.NET_corr.SI / self.NET.SI)
        self.map_depth.set_from_SI(noise.map_depth(self.NET_arr.SI, self._fsky, self._obs_time, self._obs_effic))
        self.map_depth_RJ.set_from_SI(noise.map_depth(self.NET_arr_RJ.SI, self._fsky, self._obs_time, self._obs_effic))

        # JR, find Psat
        to_shape = np.ones((self.corr_fact.value.shape))
        self.P_sat.set_from_SI(self._channel.bolo_Psat(self.opt_power.SI) * to_shape)
        self.G.set_from_SI(self._channel.bolo_G(self.opt_power.SI) * to_shape)
        self.Flink.set_from_SI(self._channel.bolo_Flink() * to_shape)


        self.summarize()
        self.analyze_optical_chain()

    def summarize(self):
        """ Compute and cache summary statistics """
        self._summary = odict()
        for key in self.summary_fields:
            self._summary[key] = self._properties[key].summarize(self)
        return self._summary

    def analyze_optical_chain(self):
        """ Compute and cache optical output statistics """
        self._optical_output = odict()
        for key in self.optical_output_fields:
            self._optical_output[key] = self._properties[key].summarize_by_element(self)
        return self._optical_output

    def print_summary(self, stream=sys.stdout):
        """ Print summary statistics in human-readable format """
        for key, val in self._summary.items():
            stream.write("%s : %s\n" % (key.ljust(20), val))

    def print_optical_output(self, stream=sys.stdout):
        """ Print optical output statistics in human-readable format """

        elem_power_from_sky = self._optical_output['elem_power_from_sky']
        elem_power_to_det = self._optical_output['elem_power_to_det']
        elem_effic = self._optical_output['elem_effic']
        elem_cumul_effic = self._optical_output['elem_cumul_effic']
        stream.write("%s | %s | %s | %s | %s\n" % ("Element".ljust(20), "Power from Sky [pW]".ljust(26), "Power to Det [pW]".ljust(26), "Efficiency".ljust(26), "Cumul. Effic.".ljust(26)))
        for idx, elem in enumerate(self._elem_names):
            stream.write("%s | %s | %s | %s | %s\n" % (elem.ljust(20),
                                                            elem_power_from_sky.element_string(idx),
                                                            elem_power_to_det.element_string(idx),
                                                            elem_effic.element_string(idx),
                                                            elem_cumul_effic.element_string(idx)))

    def make_sims_table(self, name, table_dict):
        """ Make a table with per-simulation parameters """
        o_dict = odict([(key, self._properties[key].__get__(self).value.flatten()) for key in self.summary_fields])
        try:
            return table_dict.make_datatable(name, o_dict)
        except ValueError as msg:
            s = "Column shape mismatch: "
            for k, v in o_dict.items():
                s += "%s %s, " % (k, v.size)
            raise ValueError(s) from msg

    def make_optical_table(self, name, table_dict):
        """ Make a table with optical output parameters """
        o_dict = odict()
        for val in self._optical_output.values():
            o_dict.update(val.todict())
        o_dict['element'] = np.array(self._elem_names)
        o_dict['channel'] = np.array([self._channel_name]*len(self._elem_names))
        return table_dict.make_datatable(name, o_dict)

    def make_sum_table(self, name, table_dict):
        """ Make a table with summary parameters """
        o_dict = odict()
        for val in self._summary.values():
            o_dict.update(val.todict())
        o_dict['channel'] = np.array([self._channel_name])
        return table_dict.make_datatable(name, o_dict)


    def make_tables(self, base_name, table_dict, **kwargs):
        """ Make output tables """
        if kwargs.get('save_sim', True):
            self.make_sims_table("%s_sims" % base_name, table_dict)
        if kwargs.get('save_summary', True):
            self.make_sum_table("%s_summary" % base_name, table_dict)
        if kwargs.get('save_optical', True):
            self.make_optical_table("%s_optical" % base_name, table_dict)
        return table_dict
