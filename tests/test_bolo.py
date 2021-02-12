"""
Example unit tests for bolo-calc package
"""
import unittest
import bolo

class ExampleTestCase(unittest.TestCase):
    def setUp(self):
        self.message = 'Hello, world'

    def tearDown(self):
        pass

    def test_run(self):
        foo = bolo.Example(self.message)
        self.assertEqual(foo.run(), self.message)

    def test_failure(self):
        self.assertRaises(AttributeError, bolo.bolo-calc)
        foo = bolo.Example(self.message)
        self.assertRaises(RuntimeError, foo.run, True)

if __name__ == '__main__':
    unittest.main()
