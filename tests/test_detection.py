
import unittest
import numpy as np
import scipy as sp

from ctdna.detection import compute_pval_th, calculate_bm_pvalues
import ctdna.settings as settings

__author__ = 'Johannes REITER'
__date__ = 'Jul 10, 2020'


class DetectionTest(unittest.TestCase):

    def setUp(self):
        # sequencing error rate
        self.seq_err = settings.SEQUENCING_ERROR_RATE

        self.plasma_ml = settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000
        self.tube_size = settings.TUBE_SIZE
        self.blood_amount = settings.BLOOD_AMOUNT

        # DNA concentration per plasma mL
        self.dna_conc_gamma_params = settings.FIT_GAMMA_PARAMS
        # Heitzer et al, Nature Reviews Genetics, 2019
        # average yield of a nucleated cell is 6.6 pg of DNA
        self.diploid_genome_weight_ng = settings.DIPLOID_GE_WEIGHT_ng

        # sequencing panel size
        self.cancerseek_ps = settings.CANCERSEEK_SIZE
        self.cappseq_ps = settings.CAPPSEQ_SIZE

    def tearDown(self):
        pass

    def test_compute_pval_th_cancerseek(self):
        """
        Testing computed p-values for CancerSEEK with different sequencing efficiencies
        and at different testing frequencies
        """
        annual_fpr = 0.01

        min_det_muts = 1
        min_supp_reads = 1
        min_det_vaf = 0.0

        seq_eff = 1.0
        pval_th_simulated = 1.8e-5
        pval_th = compute_pval_th(
            annual_fpr, self.cancerseek_ps, self.seq_err, seq_eff, dna_conc_gamma_params=self.dna_conc_gamma_params,
            n_min_det_muts=min_det_muts, min_supp_reads=min_supp_reads, min_det_vaf=min_det_vaf)

        self.assertAlmostEqual(pval_th, pval_th_simulated, places=6)

        seq_eff = 0.5
        pval_th_simulated = 2.1e-5
        pval_th = compute_pval_th(
            annual_fpr, self.cancerseek_ps, self.seq_err, seq_eff, dna_conc_gamma_params=self.dna_conc_gamma_params,
            n_min_det_muts=min_det_muts, min_supp_reads=min_supp_reads, min_det_vaf=min_det_vaf)

        self.assertAlmostEqual(pval_th, pval_th_simulated, places=6)

    def test_calculate_bm_pvalues(self):

        # sequencing efficiency
        seq_eff = 1.0
        sampled_fraction = self.tube_size / self.blood_amount * seq_eff

        # calculate quantiles of genome equivalent distribution circulating in the blood
        # from the distribution of plasma DNA concentrations
        quantiles = np.array([0.25, 0.5, 0.75])

        cfdna_concs = np.array([sp.stats.gamma.ppf(
            qnt, self.dna_conc_gamma_params['shape'], loc=0, scale=self.dna_conc_gamma_params['scale'])
            for qnt in quantiles])

        cfdna_hges_per_ml = cfdna_concs / self.diploid_genome_weight_ng

        # round sampled genomes to nearest integer (assume diploid normal cells)
        genomes_sampled = np.rint(2 * cfdna_hges_per_ml * self.plasma_ml * sampled_fraction).astype(np.int64)

        # calculate p-values if no mutant fragment is observed
        n_mut_frags = np.zeros(len(genomes_sampled))
        pvals = calculate_bm_pvalues(n_mut_frags, genomes_sampled, self.seq_err)
        np.testing.assert_allclose(pvals, [1, 1, 1])

        # calculate p-values if one mutant fragment is observed
        n_mut_frags = np.ones(len(genomes_sampled))
        pvals = calculate_bm_pvalues(n_mut_frags, genomes_sampled, self.seq_err)
        np.testing.assert_array_almost_equal(pvals, [0.070037, 0.121975, 0.191529])

        # calculate p-values if two mutant fragment are observed
        n_mut_frags = 2 * np.ones(len(genomes_sampled))
        pvals = calculate_bm_pvalues(n_mut_frags, genomes_sampled, self.seq_err)
        np.testing.assert_array_almost_equal(pvals, [0.002512, 0.007761, 0.019639])

        # calculate p-values if five mutant fragment are observed
        n_mut_frags = 5 * np.ones(len(genomes_sampled))
        pvals = calculate_bm_pvalues(n_mut_frags, genomes_sampled, self.seq_err)
        np.testing.assert_array_almost_equal(pvals, [1.581121e-08, 2.783244e-07, 3.032505e-06])
