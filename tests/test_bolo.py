"""
Example unit tests for bolo-calc package
"""
import unittest
import yaml
import bolo

class ExampleTestCase(unittest.TestCase):
    def setUp(self):
        self.message = 'Hello, world'

    def tearDown(self):
        pass

    def test_run_single(self):
        dd = yaml.safe_load(open('config/myExample.yaml'))
        dd['sim_config']['config_dir'] = 'config'
        top = bolo.Top(**dd)
        top.sim_config.ndet_sim = 0
        top.sim_config.nsky_sim = 0
        top.instrument.custom_atm_file = 'Bands/atacama_atm.txt'
        top.run()
        top.instrument.print_summary()
        top.instrument.write_tables('test.fits')

    def test_run_multi(self):
        dd = yaml.safe_load(open('config/myExample.yaml'))
        dd['sim_config']['config_dir'] = 'config'
        top = bolo.Top(**dd)
        top.sim_config.ndet_sim = 10
        top.sim_config.nsky_sim = 10
        top.instrument.custom_atm_file = 'Bands/atacama_atm.txt'        
        top.run()
        top.instrument.print_summary()
        top.instrument.write_tables('test.fits')

if __name__ == '__main__':
    unittest.main()
