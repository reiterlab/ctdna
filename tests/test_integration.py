
import unittest
import sys


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_database_availability(self):
        # check whether module was correctly installed and is available
        import cbmlb
        self.assertTrue('cbmlb' in sys.modules)

    # @unittest.skip('Not yet implemented')
    def test_error_handling(self):
        import cbmlb.cbmlb

        with self.assertRaises(RuntimeError):
            cbmlb.cbmlb.main(['dynamics', '-n', '-100'])


if __name__ == '__main__':
    unittest.main(warnings='ignore')
