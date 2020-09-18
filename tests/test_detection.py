
import pytest
import numpy as np
import scipy as sp

from ctdna.detection import compute_pval_th, calculate_bm_pvalues, calculate_detection_probability
import ctdna.settings as settings

__author__ = 'Johannes REITER'
__date__ = 'Jul 10, 2020'


@pytest.fixture
def tube_size():
    return 0.015


@pytest.fixture
def blood_amount():
    return 5.0


@pytest.fixture
def plasma_fraction():
    return 0.55


@pytest.fixture
def dna_conc_gamma_params():
    return {'shape': 1.86, 'location': 0.0, 'scale': 3.38}


@pytest.fixture
def diploid_genome_weight_ng():
    return 0.0066


@pytest.mark.parametrize('seq_eff, pval_th_simulated', [(1.0, 1.8e-5), (0.5, 2.1e-5)])
def test_compute_pval_th_cancerseek(seq_eff, pval_th_simulated):
    """
    Testing computed p-values for CancerSEEK with different sequencing efficiencies
    and at different testing frequencies
    """
    annual_fpr = 0.01

    min_det_muts = 1
    min_supp_reads = 1
    min_det_vaf = 0.0

    pval_th = compute_pval_th(
        annual_fpr, settings.CANCERSEEK_SIZE, settings.SEQUENCING_ERROR_RATE, seq_eff,
        dna_conc_gamma_params=settings.FIT_GAMMA_PARAMS,
        n_min_det_muts=min_det_muts, min_supp_reads=min_supp_reads, min_det_vaf=min_det_vaf)

    np.testing.assert_approx_equal(pval_th, pval_th_simulated, significant=2)


@pytest.mark.parametrize('n_mut_frag, pvals_exp',
                         [(0, [1, 1, 1]),
                          (1, [0.070037, 0.121975, 0.191529]),
                          (2, [0.002512, 0.007761, 0.019639]),
                          (5, [1.581121e-08, 2.783244e-07, 3.032505e-06])])
def test_calculate_bm_pvalues(tube_size, blood_amount, plasma_fraction, dna_conc_gamma_params, diploid_genome_weight_ng,
                              n_mut_frag, pvals_exp):

    # sequencing efficiency
    seq_eff = 1.0
    sampled_fraction = tube_size / blood_amount * seq_eff

    # calculate quantiles of genome equivalent distribution circulating in the blood
    # from the distribution of plasma DNA concentrations
    quantiles = np.array([0.25, 0.5, 0.75])

    cfdna_concs = np.array([sp.stats.gamma.ppf(
        qnt, dna_conc_gamma_params['shape'], loc=0, scale=dna_conc_gamma_params['scale'])
        for qnt in quantiles])

    cfdna_hges_per_ml = cfdna_concs / diploid_genome_weight_ng

    # round sampled genomes to nearest integer (assume diploid normal cells)
    plasma_ml = blood_amount * 1000 * plasma_fraction
    genomes_sampled = np.rint(2 * cfdna_hges_per_ml * plasma_ml * sampled_fraction).astype(np.int64)

    # calculate p-values if one mutant fragment is observed
    n_mut_frags = n_mut_frag * np.ones(len(genomes_sampled))
    pvals = calculate_bm_pvalues(n_mut_frags, genomes_sampled, settings.SEQUENCING_ERROR_RATE)
    np.testing.assert_array_almost_equal(pvals, pvals_exp)


def test_calculate_detection_probability(tube_size, blood_amount, plasma_fraction, dna_conc_gamma_params,
                                         diploid_genome_weight_ng):

    n_min_det_muts = 1
    panel_size = 2000
    n_muts_cancer = 10
    seq_eff = 1.0

    plasma_ml = blood_amount * 1000 * plasma_fraction
    sample_fraction = plasma_ml * seq_eff / blood_amount

    quantiles = np.array([0.25, 0.5, 0.75])
    # computing the scale parameter for the gamma distribution of cfDNA hGE in a plasma sample

    # TODO: Stefano, it seems you calculated the number of total genomes (both copies) rather than hGE here
    # note that the function calculate_detection_probability takes hGE and not number of genomes as input
    hge_normal_scale = dna_conc_gamma_params['scale'] * tube_size * plasma_fraction \
                       * 2 / diploid_genome_weight_ng
    n_hge_normal = np.rint(np.array([sp.stats.gamma.ppf(
        qnt, dna_conc_gamma_params['shape'], loc=0, scale=hge_normal_scale)
        for qnt in quantiles])).astype(int)

    # TODO: Stefano, PLEASE ADD A TEST CASE DIRECTLY TESTING FOR THE VALUES OF DET_PROB AND REQUIRED_MT_FRAGS
    # WITHOUT COMPARING TO ANOTHER RESULT FROM THE SAME FUNCTION; KIND OF CIRCULAR

    pval_th = 1e-4
    det_prob_pval, required_mt_frags_pval = calculate_detection_probability(
        n_min_det_muts=n_min_det_muts, panel_size=panel_size, n_muts_cancer=n_muts_cancer,
        hge_tumors=np.zeros(len(n_hge_normal)), n_hge_normal=n_hge_normal, seq_err=settings.SEQUENCING_ERROR_RATE,
        sample_fraction=sample_fraction, pval_th=pval_th)

    det_prob_frags = np.zeros((len(required_mt_frags_pval),))

    for i in range(0, len(required_mt_frags_pval)):
        det_prob = calculate_detection_probability(
            n_min_det_muts=n_min_det_muts, panel_size=panel_size, n_muts_cancer=n_muts_cancer,
            hge_tumors=np.zeros(len(n_hge_normal)), n_hge_normal=n_hge_normal,  seq_err=settings.SEQUENCING_ERROR_RATE,
            sample_fraction=sample_fraction, required_mt_frags=required_mt_frags_pval[i])
        det_prob_frags[i] = det_prob[i]

    np.testing.assert_array_almost_equal(det_prob_pval, det_prob_frags)

    # test whether RuntimeError is raised if both or none of pval_th or required_mt_frags is given
    with pytest.raises(RuntimeError):
        _ = calculate_detection_probability(
            n_min_det_muts=n_min_det_muts, panel_size=panel_size, n_muts_cancer=n_muts_cancer,
            hge_tumors=np.zeros(len(n_hge_normal)), n_hge_normal=n_hge_normal, seq_err=settings.SEQUENCING_ERROR_RATE,
            sample_fraction=sample_fraction, pval_th=None, required_mt_frags=None)

    with pytest.raises(RuntimeError):
        _ = calculate_detection_probability(
            n_min_det_muts=n_min_det_muts, panel_size=panel_size, n_muts_cancer=n_muts_cancer,
            hge_tumors=np.zeros(len(n_hge_normal)), n_hge_normal=n_hge_normal, seq_err=settings.SEQUENCING_ERROR_RATE,
            sample_fraction=sample_fraction, pval_th=1, required_mt_frags=1)
