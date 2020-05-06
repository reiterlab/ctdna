#!/usr/bin/python
"""Detection a tumor from a liquid biopsy"""

import logging
import math
from scipy.stats import poisson, binom
import numpy as np

import cbmlb.settings as settings
from cbmlb.utils import get_plasma_dna_concentrations

__date__ = 'March 5, 2020'
__author__ = 'Johannes REITER'

# get logger
logger = logging.getLogger(__name__)


class Detection:

    def __init__(self, pval_th=1.0, min_det_muts=1, min_supp_reads=1, min_det_vaf=0.0):
        """
        Detection thresholds for biomarker molecules in liquid biopsy
        :param pval_th: only mutations reaching a p-value smaller than this threshold are called
        :param min_det_muts: minimally required detected mutations for a positive screening test, by default None
                             array with number of detected mutations is returned
        :param min_supp_reads: minimal number of supporting reads to call a mutation
        :param min_det_vaf: minimal variant allele frequency to call a mutation
        """

        self.pval_th = pval_th
        self.min_det_muts = min_det_muts
        self.min_supp_reads = min_supp_reads
        self.min_det_vaf = min_det_vaf


def calculate_bm_pvalues(n_mut_frags, coverage, freq_th):
    """
    Calculate the probability for observing more than k mutant fragments if mutant fragments circulate at
    the given frequency (already accounting for sequencing errors)
    :param n_mut_frags: number of mutant fragments
    :param coverage: coverage at this position
    :param freq_th: frequency threshold which already accounts for the sequencing error rate
    :return: p-value that mutation is present at the given or higher frequency
    """
    return 1.0 - binom.cdf(k=n_mut_frags-1, n=coverage, p=freq_th)


def get_pval_th(annual_fpr_th, smp_frq, n_precursors, bn_lesion_size, d_bn, q_d, epsilon,
                muts_per_tumor, panel_size, seq_err, seq_eff, n_sim_tumors,
                min_det_muts=1, min_supp_reads=1, min_det_vaf=0.0,
                biomarker_wt_freq_ml=None, tube_size=settings.TUBE_SIZE,
                diploid_genome_weight_ng=settings.DIPLOID_GE_WEIGHT_ng):
    """
    Calculate the p-value threshold for calling mutations in a tumor as present such that a
    desired annual false-positive rate threshold is achieved
    :param annual_fpr_th:
    :param smp_frq:
    :param n_precursors:
    :param bn_lesion_size:
    :param d_bn:
    :param q_d:
    :param epsilon:
    :param muts_per_tumor:
    :param panel_size:
    :param seq_err:
    :param seq_eff: fraction of molecules in sample that get sequenced; see Chabon et al, Nature 2020
    :param n_sim_tumors:
    :param min_det_muts:
    :param min_supp_reads:
    :param min_det_vaf:
    :param biomarker_wt_freq_ml:
    :param tube_size:
    :param diploid_genome_weight_ng:
    :return:
    """
    tests_per_year = 365.0 / smp_frq
    norm_fpr_th = annual_fpr_th / tests_per_year

    logger.info(f'Calculating p-val threshold for panel size {panel_size:.1e} with seq. err {seq_err:.1e} and '
                + f'seq. eff {seq_eff:.1%} covering {muts_per_tumor} muts of in {n_precursors} lesions '
                + f'used every {smp_frq} days for a desired normalized fpr of {norm_fpr_th:.1e} '
                + f'(annual {annual_fpr_th:.1e}, n={n_sim_tumors:.0e}).')

    if biomarker_wt_freq_ml is None:
        plasma_dna_concs = get_plasma_dna_concentrations(n_sim_tumors, gamma_params=settings.FIT_GAMMA_PARAMS)
        # calculate whole genome equivalents per plasma ml
        wGEs_per_ml = plasma_dna_concs / diploid_genome_weight_ng
    else:
        wGEs_per_ml = biomarker_wt_freq_ml

    # assuming 5 liters of blood and 55% of plasma
    plasma_mL = settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000
    n_bm_wts = plasma_mL * wGEs_per_ml
    # number of mutant biomarkers shed by benign lesions
    n_bm_mts = np.zeros(n_sim_tumors)
    for _ in range(n_precursors):
        n_bm_mts += poisson.rvs(bn_lesion_size * d_bn * q_d / epsilon, size=n_sim_tumors)
    # sum the random biomarker amount from the cancer plus the normal level
    n_bms = n_bm_wts + n_bm_mts
    # calculate VAFs (variant allele frequencies)
    bm_mt_fractions = n_bm_mts / n_bms
    del n_bm_mts

    # assuming ~5 liters of blood
    sample_fraction = tube_size / settings.BLOOD_AMOUNT * seq_eff
    # round elements of the array to the nearest integer
    n_sampled_bms = np.rint(n_bms * sample_fraction).astype(int)
    del n_bms

    # calculate the expected fraction of the biomarker in the bloodstream
    obs_bm_mt_fractions = bm_mt_fractions * (1 - seq_err) + (1 - bm_mt_fractions) * seq_err

    # compute observed number of mutant fragments from actual mutations
    # sampled mutant biomarkers of n sampled total biomarkers
    if muts_per_tumor > 0:
        sampled_bm_mts1 = np.random.binomial(n_sampled_bms, obs_bm_mt_fractions,
                                             size=(muts_per_tumor, len(n_sampled_bms)))
    else:
        sampled_bm_mts1 = np.zeros((muts_per_tumor, len(n_sampled_bms)))
    del obs_bm_mt_fractions

    logger.debug('Computing sequencing errors.')
    # compute observed number of mutant fragments due to sequencing errors from wildtypes
    if panel_size - muts_per_tumor > 0:
        # sampled mutant biomarkers
        sampled_bm_mts2 = np.random.binomial(n_sampled_bms, seq_err,
                                             size=(panel_size - muts_per_tumor, len(n_sampled_bms)))
    else:
        sampled_bm_mts2 = np.zeros((panel_size - muts_per_tumor, len(n_sampled_bms)))

    sampled_bm_mts = np.append(sampled_bm_mts1, sampled_bm_mts2, axis=0)
    del sampled_bm_mts1
    del sampled_bm_mts2

    logger.debug('Calculating p-values.')
    # calculate p-values

    # set frequency of expected background mutant fragments due to sequencing error
    pvals = calculate_bm_pvalues(sampled_bm_mts, n_sampled_bms, seq_err)

    logger.debug('Calculating number of false-positives.')
    # set p-values to 1 if the VAF was under the VAF threshold
    vafs = (sampled_bm_mts / n_sampled_bms)
    pvals[np.where(vafs < min_det_vaf)] = 1
    pvals[np.where(sampled_bm_mts < min_supp_reads)] = 1

    # sort the p-values in each row corresponding to p-vals of all covered basepairs per subject
    pvals_sorted = pvals.transpose()
    pvals_sorted.sort()
    # calculated the allowed number of false-positives for the desired false-positive rate
    n_allowed_fps = math.floor(norm_fpr_th * n_sim_tumors)
    # take the smallest p-values of each subject and then take the smallest ones to find
    # the p-value threshold to stay under the desired false-positive rate
    pval_th = np.sort(pvals_sorted[:, min_det_muts - 1])[max(0, n_allowed_fps - 1)]

    logger.info('p-val threshold: {:.3e} for a panel size of {:.1e} covering {} mutations '.format(
        pval_th, panel_size, muts_per_tumor)
                + 'applied every {} days for a desired normalized fpr of {:.1e} (annual {:.1e}, n={:.0e}).'.format(
        smp_frq, norm_fpr_th, annual_fpr_th, n_sim_tumors))

    del pvals
    del pvals_sorted

    return pval_th
