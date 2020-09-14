
import unittest
import numpy as np

from ctdna.detection import compute_pval_th, calculate_sensitivity
from ctdna.utils import diameter_cells, cells_diameter, calculate_elimination_rate
import ctdna.settings as settings

__author__ = 'Johannes REITER'
__date__ = 'September 14, 2020'


class ExamplesTest(unittest.TestCase):

    def test_ReadmeExample1(self):
        """
        NOTE: any changes here also need to be made in example.ipynb
        """

        # lung cancer
        b_lung = 0.14  # cell birth rate
        d_lung = 0.136  # cell death rate
        q_d_lung = 1.4e-4  # shedding probability

        fpr = 0.01  # false positive rate
        seq_err = 1e-5  # sequencing error rate

        # convert cfDNA half-life time in minutes to an elimination rate per day
        t12_cfdna_mins = 30
        epsilon = calculate_elimination_rate(t12_cfdna_mins)

        # parameters for the gamma-distributed plasma DNA concentrations
        dna_conc_gamma_params = settings.FIT_GAMMA_PARAMS

        seq_eff = 0.5  # sequencing efficiency
        panel_size = 1  # consider exactly one actionable mutation
        n_det_muts = 1  # number of called mutations required for detection
        n_muts_cancer = n_det_muts  # actionable mutation is present in the cancer cells

        # translate tumor diameters [cm] into number of cells
        tumor_sizes = np.array([diameter_cells(1), 1e9, diameter_cells(1.5), diameter_cells(2)])

        # calculate a threshold to call a mutation such that a given false positive rate is achieved
        pval_th = compute_pval_th(
            fpr, panel_size, seq_err, seq_eff, dna_conc_gamma_params, epsilon=epsilon)

        # calculate the probability to detect a mutation of tumors with different sizes
        det_probs = calculate_sensitivity(
            b_lung, d_lung, q_d_lung, epsilon, n_det_muts, panel_size, n_muts_cancer,
            pval_th=pval_th, dna_conc_gamma_params=dna_conc_gamma_params,
            seq_err=seq_err, seq_eff=seq_eff, tumor_sizes=tumor_sizes)

        # diameters = [cells_diameter(size) for size in tumor_sizes]
        # print('Detection probabilities for tumors of sizes: '
        #       + ', '.join(f'{p:.1%} ({d:.1f} cm; {c:.1e})' for p, c, d in zip(det_probs, tumor_sizes, diameters)))

        np.testing.assert_array_almost_equal(det_probs, [0.176224, 0.336485, 0.55963, 0.906548])


if __name__ == '__main__':
    unittest.main()
