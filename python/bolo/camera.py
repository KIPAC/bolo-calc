""" Class to model camera """

from collections import OrderedDict as odict


from cfgmdl import Property, Model
from cfgmdl.tools import build_class
from cfgmdl.utils import expand_dict

from .channel import Channel


class Camera_Base(Model):
    """
    Camera model
    """
    boresite_elevation = Property(dtype=float, default=0.)
    optical_coupling = Property(dtype=float, default=1.)
    f_number = Property(dtype=float, default=2.5)
    bath_temperature = Property(dtype=float, default=0.1)
    skip_optical_elements = Property(dtype=list)
    chan_config = Property(dtype=dict, help="Configuation for channels")

    def __init__(self, **kwargs):
        """ Constructor """
        super(Camera_Base, self).__init__(**kwargs)
        self.channels = odict()
        self.optics = None
        self._instrument = None
        for key, val in self.__dict__.items():
            if isinstance(val, Channel):
                self.channels[key] = val

    def set_parent(self, instrument):
        """ Pass information from the parent instrument down the food chain """
        self._instrument = instrument
        self.optics = odict()
        for key, val in instrument.optics.elements.items():
            if key[1:] in self.skip_optical_elements:
                continue
            self.optics[key[1:]] = val
        for idx, chan in enumerate(self.channels.values()):
            chan.set_camera(self, idx)

    def sample(self, nsamples=0):
        """ Sample parameters in all the channels """
        for chan in self.channels.values():
            chan.sample(nsamples)

    def eval_optical_chains(self, nsamples=0, freq_resol=None):
        """ Compute the performance of the elements of the optical chain for each channel """
        for chan in self.channels.values():
            chan.eval_optical_chain(nsamples, freq_resol)

    def eval_sky(self, universe, freq_resol=None):
        """ Compute parameters related to the sky that depend on the particular camera.

        This is mainly handle potential differences in elevation between cameras. """
        for chan in self.channels.values():
            chan.eval_sky(universe, freq_resol)

    def eval_det_response(self, nsample=0, freq_resol=None):
        """ Compute the performance of the detectors of the optical chain """
        for chan in self.channels.values():
            chan.eval_det_response(nsample, freq_resol)

    @property
    def instrument(self):
        """ Return the parent instrument """
        return self._instrument


def build_camera_class(name="Camera", **kwargs):
    """ Build a camera from a configuration dictionary """
    kwcopy = kwargs.copy()
    type_dicts = [{None:Channel}]
    config_dicts = [kwcopy.pop('chan_config')]

    return build_class(name, (Camera_Base, ), config_dicts, type_dicts, **kwcopy)



def build_cameras(def_channel_config, camera_config):
    """ Build a set of cameras from a configuration dictionary """
    cam_full = expand_dict(camera_config)

    ret = odict()
    for key, val in cam_full.items():
        cam_config = val.copy()
        cam_config['chan_config']['default'] = def_channel_config.copy()
        ret[key] = build_camera_class(key, **cam_config)
    return ret
