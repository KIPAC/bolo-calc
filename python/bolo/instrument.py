""" Class to represent an instrument """

import sys

from collections import OrderedDict as odict

from cfgmdl import Property, Model
from astropy.table import vstack

from .readout import Readout
from .optics import build_optics_class
from .camera import build_cameras
from .sensitivity import Sensitivity

#from .unit import Unit
from .data_utils import TableDict
from .cfg import Variable


class Instrument(Model):
    """ Class to represent an instrument """
    site = Property(dtype=str, required=True)
    sky_temp = Variable(required=True, unit='K')
    obs_time = Variable(required=True, unit='yr')
    sky_fraction = Variable(required=True)
    NET = Variable(required=True)

    custom_atm_file = Property(dtype=str)

    elevation = Variable(required=True)
    pwv = Variable(required=True)
    obs_effic = Variable(required=True)

    readout = Property(dtype=Readout, required=True)
    camera_config = Property(dtype=dict, required=True)
    optics_config = Property(dtype=dict, required=True)
    channel_default = Property(dtype=dict, required=True)

    def __init__(self, **kwargs):
        """ Constructor """
        super(Instrument, self).__init__(**kwargs)
        self.optics = build_optics_class(**self.optics_config)
        self.cameras = build_cameras(self.channel_default, self.camera_config)
        self._tables = None
        self._sns_dict = None
        for key, val in self.cameras.items():
            self.__dict__[key] = val
            val.set_parent(self, key)

    def eval_sky(self, universe, nsamples=0, freq_resol=None):
        """ Sample requested inputs and evaluate the parameters of the sky model """
        universe.sample(nsamples)
        self.obs_effic.sample(nsamples)
        for camera in self.cameras.values():
            camera.eval_sky(universe, freq_resol)

    def eval_instrument(self, nsamples=0, freq_resol=None):
        """ Sample requested inputs and evaluate the parameters of the instrument model """
        for camera in self.cameras.values():
            camera.sample(nsamples)
            camera.eval_optical_chains(nsamples, freq_resol)
            camera.eval_det_response(nsamples, freq_resol)

    def eval_sensitivities(self):
        """ Evaluate the sensitvities """
        self._sns_dict = odict()
        for cam_name, camera in self.cameras.items():
            for chan_name, channel in camera.channels.items():
                full_name = "%s%s" % (cam_name, chan_name)
                self._sns_dict[full_name] = Sensitivity(channel)

    def make_tables(self, basename="", **kwargs):
        """ Make fits tables with output values """
        self._tables = TableDict()
        for key, val in self._sns_dict.items():
            val.make_tables("%s%s" % (basename, key), self._tables, **kwargs)

        # get the summary tables and stack them
        if kwargs.get('save_summary', True):
            sum_keys = [ key for key in self._tables.keys() if key.find('_summary') > 0 ]
            sum_table = vstack([self._tables.pop_table(sum_key) for sum_key in sum_keys])
            self._tables.add_datatable("%ssummary" % basename, sum_table)
        # get the optical tables adn stack them
        if kwargs.get('save_optical', True):
            opt_keys = [ key for key in self._tables.keys() if key.find('_optical') > 0 ]
            opt_table = vstack([self._tables.pop_table(opt_key) for opt_key in opt_keys])
            self._tables.add_datatable("%soptical" % basename, opt_table)
        return self._tables

    def write_tables(self, filename):
        """ Write output fits tables """
        if self._tables:
            self._tables.save_datatables(filename)

    @property
    def tables(self):
        """ Get the out put data tables"""
        return self._tables

    def print_summary(self, stream=sys.stdout):
        """ Print summary stats in humman readable format """
        for key, val in self._sns_dict.items():
            stream.write("%s ---------\n" % key)
            val.print_summary(stream)
            stream.write("---------\n")

    def print_optical_output(self, stream=sys.stdout):
        """ Print summary stats in humman readable format """
        for key, val in self._sns_dict.items():
            stream.write("%s ---------\n" % key)
            val.print_optical_output(stream)
            stream.write("---------\n")

    def run(self, universe, sim_cfg, basename=""):
        """ Run the analysis chain """
        self.eval_sky(universe, sim_cfg.nsky_sim, sim_cfg.freq_resol)
        self.eval_instrument(sim_cfg.ndet_sim, sim_cfg.freq_resol)
        self.eval_sensitivities()
        save_summary = sim_cfg.save_summary
        if max(sim_cfg.nsky_sim, 1) * max(sim_cfg.ndet_sim, 1) == 1:
            save_summary = False
        self.make_tables(basename, save_summary=save_summary, save_sim=sim_cfg.save_sim, save_cfg=sim_cfg.save_optical)
