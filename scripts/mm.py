import yaml

from bolo.model.dummy import Dummy

m = Dummy()

m2 = Dummy(m=4)

m3 = Dummy(m=dict(value=4, unit='K'))

try: m_error = Dummy(m="sss")
except (TypeError, ValueError):
    pass
else:
    print("nope")

