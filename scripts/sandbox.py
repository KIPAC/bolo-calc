import yaml

from bolo.model.foregrounds import Dust, Universe, Foreground
from bolo.model.opticalChain import build_optics_class
from bolo.model.top import Top

dd = yaml.safe_load(open('config/myExample.yaml'))

#ddict = dict(spectral_index=1.5, amplitude=1.2E-2, scale_frequency=353.0E+9)
#dust = Foreground(**ddict)
#dust2 = Foreground(**ddict)

top = Top(**dd)
optics = build_optics_class(**dd['instrument']['optics'])

#optics_dict = dd['instrument']['optics']
#optics_dict_full = expand_dict_from_defaults_and_elements(optics_dict['default'], optics_dict['elements'])

#optics = {key: OpticalElement(name=key, **val) for key, val in optics_dict_full.items()}

