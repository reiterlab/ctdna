
import unittest
import sys

import numpy as np
from ctdna.detection import compute_pval_th, calculate_sensitivity
from ctdna.utils import diameter_cells, calculate_elimination_rate
import ctdna.settings as settings


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_database_availability(self):
        # check whether module was correctly installed and is available
        import ctdna
        self.assertTrue('ctdna' in sys.modules)

    def test_actionable_mutations(self):

        # lung cancer
        b_lung = 0.14       # cell birth rate
        d_lung = 0.136      # cell death rate
        q_d_lung = 1.4e-4   # shedding probability

        fpr = 0.01          # false positive rate
        seq_err = 1e-5      # sequencing error rate

        # convert cfDNA half-life time in minutes to an elimination rate per day
        t12_cfdna_mins = 30
        epsilon = calculate_elimination_rate(t12_cfdna_mins)

        # parameters for the gamma-distributed plasma DNA concentrations
        dna_conc_gamma_params = settings.FIT_GAMMA_PARAMS

        seq_eff = 0.5           # sequencing efficiency
        panel_size = 1          # consider exactly one actionable mutation
        n_det_muts = 1              # number of called mutations required for detection
        n_muts_cancer = n_det_muts  # actionable mutation is present in the cancer cells

        tumor_sizes = np.array([diameter_cells(1), diameter_cells(1.5), diameter_cells(2)])

        pval_th = compute_pval_th(
            fpr, panel_size, seq_err, seq_eff, dna_conc_gamma_params, epsilon=epsilon)

        self.assertAlmostEqual(pval_th, 0.03744832, places=6)

        det_probs = calculate_sensitivity(
            b_lung, d_lung, q_d_lung, epsilon, n_det_muts, panel_size, n_muts_cancer,
            pval_th, dna_conc_gamma_params=dna_conc_gamma_params,
            seq_err=seq_err, seq_eff=seq_eff, tumor_sizes=tumor_sizes)

        np.testing.assert_array_almost_equal(det_probs, [0.17622353, 0.55962951, 0.90654751])


    # @unittest.skip('Not yet implemented')
    def test_error_handling(self):
        import ctdna.ctdna

        with self.assertRaises(RuntimeError):
            ctdna.ctdna.main(['dynamics', '-n', '-100'])


if __name__ == '__main__':
    unittest.main(warnings='ignore')
