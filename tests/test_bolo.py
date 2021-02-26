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

    def test_run(self):
        dd = yaml.safe_load(open('config/myExample.yaml'))
        dd['sim_config']['config_dir'] = 'config'
        top = bolo.Top(**dd)
        top.run()
        top.instrument.print_summary()
        top.instrument.write_tables('test.fits')

    def test_failure(self):
        pass

if __name__ == '__main__':
    unittest.main()
