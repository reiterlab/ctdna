#!/usr/bin/python
"""Detection a tumor from a liquid biopsy"""

import logging
import math
import numpy as np
import scipy as sp
from scipy.stats import poisson, binom

import ctdna.settings as settings
from ctdna.utils import get_plasma_dna_concentrations
from ctdna.bp_formulas import get_bms_at_size

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

    def __str__(self):
        name = f'{self.__class__.__name__}'
        if self.min_det_muts > 1:
            name += f', called muts>={self.min_det_muts}'
        if self.pval_th < 1:
            name += f', p-val<={self.pval_th:.3e}'
        if self.min_supp_reads > 1:
            name += f', supp reads>={self.min_supp_reads}'
        if self.min_det_vaf > 0:
            name += f', min VAF>={self.min_det_vaf}'

        return name


def calculate_bm_pvalues(n_mut_frags, coverage, freq_th):
    """
    Calculate the probability for observing more than k mutant fragments if mutant fragments circulate at
    the given frequency (already accounting for sequencing errors)
    Probability for observing k or more mutant fragments just due to sequencing errors
    :param n_mut_frags: number of mutant fragments
    :param coverage: coverage at this position
    :param freq_th: frequency threshold accounting for the sequencing error rate
    :return: p-value that mutation is present at the given or higher frequency
    """
    # return 1.0 - binom.cdf(k=n_mut_frags-1, n=coverage, p=freq_th)
    return binom.sf(k=n_mut_frags-1, n=coverage, p=freq_th)


def compute_pval_th(annual_fpr_th, panel_size, seq_err, seq_eff, dna_conc_gamma_params=None, wt_hge_per_ml=None,
                    smp_frq=None, n_min_det_muts=1, min_supp_reads=1, min_det_vaf=0.0,
                    n_bn_lesions=0, bn_lesion_size=None, d_bn=None, q_d_bn=None, epsilon=None, muts_per_bn_lesion=None,
                    tube_size=settings.TUBE_SIZE, diploid_genome_weight_ng=settings.DIPLOID_GE_WEIGHT_ng,
                    blood_amount=settings.BLOOD_AMOUNT, plasma_fraction=settings.PLASMA_FRACTION,
                    resolution=1000):
    """
    Compute the p-value threshold for calling mutations in a tumor as present such that a
    desired (annual) false-positive rate threshold is achieved
    :param annual_fpr_th: desired false-positive rate for a single test or over an entire year if smp_frq is given
    :param panel_size: number of base-pairs covered by the sequencing panel
    :param seq_err: sequencing error rate per base-pair
    :param seq_eff: fraction of the sampled molecules that are actually successfully sequenced
    :param dna_conc_gamma_params: plasma DNA concentration is sampled from the given gamma distribution parameters
    :param wt_hge_per_ml: assume a fixed number of normal cfDNA haploid genome equivalents per mL
                          instead the above gamma distribution depicted in Fig. S5B of Avanzini et al (2020)
    :param smp_frq: sampling interval in days to adjust false-positive rate per test (optional)
    :param n_min_det_muts: minimal number of called mutations required for a positive test
    :param min_supp_reads: minimum number of mutant fragments per position to call a position
    :param min_det_vaf: minimal variant allele frequency for a mutation to be called
    :param n_bn_lesions: number of benign lesions shedding ctDNA
    :param bn_lesion_size: size [cells] of each benign lesion
    :param d_bn: death rate of cells in benign lesions
    :param q_d_bn: ctDNA shedding probability per cell death of benign cells
    :param epsilon: ctDNA elimination rate
    :param muts_per_bn_lesion: mutations of benign cells that are covered by the sequencing panel
    :param tube_size: size of blood sample tube [liters; default 0.015 l]
    :param diploid_genome_weight_ng: weight of the DNA of a diploid cell [ng]
    :param blood_amount: amount of blood circulating [liters; default 5 l]
    :param plasma_fraction: fraction of blood that is plasma
    :param resolution: number of cut points to discretize the distribution of plasma DNA concentration
    :return: p-value upper bound threshold for a given maximum annual false-positive rate
    """
    if smp_frq is not None:
        tests_per_year = 365.0 / smp_frq
        norm_fpr_th = annual_fpr_th / tests_per_year
    else:
        norm_fpr_th = annual_fpr_th

    logger.info(f'Calculating p-val threshold for {n_min_det_muts} det. mut panel size {panel_size:.1e} with seq. err '
                + f'{seq_err:.1e} and seq. eff {seq_eff:.1%} used every {smp_frq} days for a desired '
                + f'normalized fpr of {norm_fpr_th:.1e} (annual {annual_fpr_th:.1e}).')

    # assuming 5 liters of blood and 55% of plasma
    plasma_ml = blood_amount * plasma_fraction * 1000
    sampled_fraction = tube_size / blood_amount * seq_eff

    if dna_conc_gamma_params is None and wt_hge_per_ml is None:
        raise RuntimeError('Either parameters for the gamma-distributed cfDNA plasma concentration need to be given '
                           'or some fixed number of normal haploid genome equivalents per plasma mL')

    elif dna_conc_gamma_params is not None:
        # calculate percentiles/permilles (parts per thousands) of genome equivalent distribution in the blood
        # from the distribution of plasma DNA concentrations
        pcts = np.linspace(1.0 / resolution, 1.0 - (1.0 / resolution), resolution - 1)
        # pcts = np.linspace(0.001, 0.999, 999)

        cfdna_concs = np.array([
            sp.stats.gamma.ppf(pct, dna_conc_gamma_params['shape'], loc=0, scale=dna_conc_gamma_params['scale'])
            for pct in pcts])
        cfdna_hge_per_ml = cfdna_concs / diploid_genome_weight_ng
    else:
        cfdna_hge_per_ml = np.array([wt_hge_per_ml])

    # TODO
    if min_supp_reads > 1:
        raise NotImplementedError('Requiring a minimum number of mutant fragments per position is not yet implemented.')

    if min_det_vaf > 0.0:
        raise NotImplementedError('Requiring a minimum variant allele frequency for a considered mutation'
                                  + ' is not yet implemented.')

    # round circulating genomes to nearest integer (two copies of hGE)
    n_genomes_normal = np.rint(2 * cfdna_hge_per_ml * plasma_ml).astype(np.int64)

    # probability to sample mutant fragment due sequencing errors
    smp_seq_err_prob = sampled_fraction * seq_err

    if n_bn_lesions is not None and n_bn_lesions > 0 and muts_per_bn_lesion is not None and muts_per_bn_lesion > 0:
        logger.info(f'Considering {n_bn_lesions} benign lesions with {muts_per_bn_lesion} muts covered by the panel. '
                    + f'Benign cells die with rate d_bn={d_bn:.1e} and have a shedding probability of qd={q_d_bn:.1e}.')

        # mean number of haploid genome equivalents shed by each benign lesions
        hge_precursor_mean = get_bms_at_size(bn_lesion_size, d_bn * q_d_bn, 0, epsilon)
        hges_precursors = np.rint(np.array(np.meshgrid(
            *[hge_precursor_mean for _ in range(n_bn_lesions)])).T.reshape(-1, n_bn_lesions))

        n_genomes = n_genomes_normal + 2 * np.sum(hges_precursors, axis=1)
        bn_vafs = hge_precursor_mean / n_genomes
        normal_vafs = 1 - bn_vafs

        # probability to sample mutant fragment from benign lesion
        smp_bn_prob = sampled_fraction * ((normal_vafs * seq_err) + (bn_vafs * (1 - seq_err)))

    else:
        n_genomes = n_genomes_normal
        # probability to sample mutant fragment from benign lesion
        smp_bn_prob = 0
        muts_per_bn_lesion = 0
        n_bn_lesions = 0

    # extreme maximum of mutated fragments that could be expected under any conditions
    n_max_frags = int(round(binom.ppf(1.0 - 1e-10, n=max(n_genomes), p=smp_seq_err_prob) + 2))
    # calculate for each number of mutant fragments the probability that it is the highest per panel
    # and calculate the associated p-value with it across percentiles of plasma DNA distribution
    prob_k_more = np.zeros((n_max_frags, len(n_genomes)))
    pvals = np.zeros_like(prob_k_more)

    for k in range(n_max_frags, 0, -1):
        # simpler implementation for the case without benign lesions
        # # probability to observe k or more mutant fragments for at least min muts required for detection
        # if min_det_muts == 1:
        #     prob_single_pos = binom.sf(k=k - 1, n=n_genomes, p=smp_seq_err_prob)
        #     prob_k_more[k - 1, :] = binom.sf(k=0, n=panel_size, p=prob_single_pos)
        #     # equivalent to
        #     # prob_k_more[k-1,:] = 1 - (binom.cdf(k=k-1, n=n_genomes, p=seq_err)**panel_size)
        #     logger.debug(f'Probability to observe at least {k} mutant fragments at any basepair of the panel: '
        #                  + f'{np.mean(prob_k_more[k - 1, :]):.3e}')
        # else:
        #     # probability to at least k mutant fragments of each of the minimally called mutations for detection
        #     prob_single_pos = binom.sf(k=k - 1, n=n_genomes, p=smp_seq_err_prob)
        #     prob_k_more[k - 1, :] = binom.sf(k=min_det_muts - 1, n=panel_size, p=prob_single_pos)

        # probability to observe k or more mutant fragments of each of the minimally called mutations for detection
        mt_prob_mt_pos = binom.sf(k=k - 1, n=n_genomes, p=smp_bn_prob)

        # probability to observe k or more mutant fragments at positions not mutated in the benign lesions
        mt_prob_wt_pos = binom.sf(k=k - 1, n=n_genomes, p=smp_seq_err_prob)

        # sum probabilities of combinations in which n_min_det_muts mutations can be detected
        probs = np.zeros((n_min_det_muts + 1, len(n_genomes)))
        for det_muts in range(n_min_det_muts + 1):

            # observe a mutation with at least k fragments at a mutated position in the tumor
            if det_muts == n_min_det_muts:
                probs[det_muts, :] = binom.sf(k=det_muts - 1, n=muts_per_bn_lesion * n_bn_lesions, p=mt_prob_mt_pos)
            else:
                probs[det_muts, :] = binom.pmf(k=det_muts, n=muts_per_bn_lesion * n_bn_lesions, p=mt_prob_mt_pos)

            # observe a mutation with at least k fragments at a position not mutated in the tumor
            probs[det_muts, :] *= binom.sf(k=n_min_det_muts - det_muts - 1,
                                           n=panel_size - muts_per_bn_lesion * n_bn_lesions,
                                           p=mt_prob_wt_pos)

        # sum each column which denotes the probability that at least X mutations are detected
        prob_k_more[k - 1, :] = np.sum(probs, axis=0)
        #         logger.debug(f'Probability to observe at least {k} mutant fragments at the {min_det_muts}th '
        #                      + f'most mutated basepair: {np.mean(prob_k_more[k - 1, :]):.3e}')

        # probability to observe k or more mutant fragments at a basepair due to sequencing errors
        pvals[k - 1, :] = binom.sf(k=k - 1, n=n_genomes, p=smp_seq_err_prob)

    # Run through the flattened array of expected p-values to find the threshold at which
    # the desired normalized annual false positive rate would be reached #####
    # flatten array of p-values for most significantly mutated position of panel for each number of genome equivalents
    pvals_flat = pvals.flatten()
    # normalize probabilities to the number of considered points of the cfDNA concentrations
    prob_k_more_flat = prob_k_more.flatten() / len(n_genomes)
    prob_k_more /= len(n_genomes)
    sorted_ind = pvals_flat.argsort()
    fps_frac = 0
    for i, ind in enumerate(sorted_ind):
        fps_frac += prob_k_more_flat[ind]
        unraveled_index = np.unravel_index(ind, prob_k_more.shape)
        # subtract previous false positives due to having more than k+1 mutant fragments
        # which is included in having more than k mutant fragments
        if unraveled_index[0] < prob_k_more.shape[0] - 1:
            fps_frac -= prob_k_more[unraveled_index[0] + 1, unraveled_index[1]]

        # if with the current p-value threshold more false positives are expected than the desired false positive
        # threshold, use the p-value from the last loop iteration as the threshold
        if fps_frac > norm_fpr_th:
            pval_th = pvals_flat[sorted_ind[i - 1]]
            logger.info(f'Computed a p-value threshold of {pval_th:.3e} for a panel size of {panel_size} '
                        + f'applied every {smp_frq} days for a desired normalized fpr of {norm_fpr_th:.3e} '
                        + f'(annual {annual_fpr_th:.3e}).')
            break

    else:
        unraveled_index = np.unravel_index(sorted_ind[-1], prob_k_more.shape)
        pval_th = pvals_flat[sorted_ind[-1]]
        logger.error(f'Only reached false positive fraction of {fps_frac:.3e} with a p-value '
                     + f'of {pval_th:.3e} corresponding to observing {unraveled_index[0] + 1} '
                     + 'mutant fragments.')
        logger.fatal(
            f'p-value threshold could not be adapted to reach a false positive rate of {annual_fpr_th:.2e} for '
            + f'panel size {panel_size:.0e}, sequencing error rate {seq_err:.1e}, {n_min_det_muts} required mutations.')

    assert 0 < pval_th <= 1, f'computed p-value threshold {pval_th:.3e} has to be within (0,1]'

    return pval_th


def simulate_pval_th(annual_fpr_th, smp_frq, n_precursors, bn_lesion_size, d_bn, q_d, epsilon,
                     muts_per_tumor, panel_size, seq_err, seq_eff, n_sim_tumors,
                     min_det_muts=1, min_supp_reads=1, min_det_vaf=0.0,
                     cfdna_hge_per_ml=None, tube_size=settings.TUBE_SIZE,
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
    :param cfdna_hge_per_ml:
    :param tube_size:
    :param diploid_genome_weight_ng:
    :return:
    """
    tests_per_year = 365.0 / smp_frq
    norm_fpr_th = annual_fpr_th / tests_per_year

    logger.info(f'Simulating p-val threshold for {min_det_muts} det. mut panel size {panel_size:.1e} with seq. err '
                + f'{seq_err:.1e} and seq. eff {seq_eff:.1%} covering {muts_per_tumor} muts of in '
                + f'{n_precursors} lesions used every {smp_frq} days for a desired normalized fpr of {norm_fpr_th:.1e} '
                + f'(annual {annual_fpr_th:.1e}, n={n_sim_tumors:.0e}).')

    if cfdna_hge_per_ml is None:
        plasma_dna_concs = get_plasma_dna_concentrations(n_sim_tumors, gamma_params=settings.FIT_GAMMA_PARAMS)
        # calculate whole genome equivalents per plasma ml
        cfdna_hge_per_ml = plasma_dna_concs / (diploid_genome_weight_ng / 2)

    # assuming 5 liters of blood and 55% of plasma
    plasma_ml = settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000
    cfdna_hge = plasma_ml * cfdna_hge_per_ml
    # number of mutant biomarkers shed by benign lesions
    ctdna_hge = np.zeros(n_sim_tumors)
    for _ in range(n_precursors):
        ctdna_hge += poisson.rvs(bn_lesion_size * d_bn * q_d / epsilon, size=n_sim_tumors)
    # sum the random mutant + wildtype biomarker amount from the cancer plus the normal level
    # because only one copy of ctDNA is mutated that one is modeled with q_d
    n_bms = np.rint(cfdna_hge + ctdna_hge + ctdna_hge).astype(int)
    # calculate VAFs (variant allele frequencies)
    bm_mt_fractions = ctdna_hge / n_bms
    del ctdna_hge

    # assuming ~5 liters of blood
    sample_fraction = tube_size / settings.BLOOD_AMOUNT * seq_eff
    # round elements of the array to the nearest integer
    # n_sampled_bms = np.rint(n_bms * sample_fraction).astype(int)
    n_sampled_bms = np.random.binomial(n_bms, sample_fraction)
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
                + 'applied every {} days for a desired normalized fpr of {:.3e} (annual {:.3e}, n={:.0e}).'.format(
        smp_frq, norm_fpr_th, annual_fpr_th, n_sim_tumors))

    del pvals
    del pvals_sorted

    return pval_th


def calculate_detection_probability(n_min_det_muts, panel_size, n_muts_cancer, hge_tumors,
                                    n_hge_normal, seq_err, sample_fraction, pval_th=None, required_mt_frags=None):
    """
    Calculate the probability to detect a tumor if there are hge_tumor hGE circulating in the entire bloodstream
    :param n_min_det_muts: number of minimally called mutations required for a positive cancer detection test
    :param panel_size: sequencing panel size
    :param n_muts_cancer: number of mutations covered by the panel that are clonally present in the tumor
    :param hge_tumors: array_like numbers of haploid genome equivalents (hGE) circulating in the entire bloodstream
    :param n_hge_normal: number of normal hGE circulating in the entire bloodstream
                        (will be multiplied by two to account for diploid genomes)
    :param seq_err: sequencing error rate per basepair
    :param sample_fraction: fraction of the bloodstream that is sampled
    :param pval_th: p-value threshold to call an individual mutation in the panel
    :param required_mt_frags: minimum number of mutated fragments required to call mutation at a given position
    :return: probability that the test will be positive
    """
    n_genomes_total = 2 * n_hge_normal + 2 * hge_tumors
    tumor_vaf = hge_tumors / n_genomes_total
    normal_vaf = 1 - tumor_vaf

    mt_prob = sample_fraction * ((normal_vaf * seq_err) + (tumor_vaf * (1 - seq_err)))
    seq_err_prob = sample_fraction * seq_err

    if pval_th is None and required_mt_frags is None:
        err_str = ('Either a p-value threshold or a number of required mutant fragments is needed '
                   'to compute detection probability.')
        logger.error(err_str)
        raise RuntimeError(err_str)

    elif pval_th is not None and required_mt_frags is not None:
        err_str = ('Only a p-value threshold or a number of required mutant fragments should be given '
                   'to compute detection probability. Not both.')
        logger.error(err_str)
        raise RuntimeError(err_str)

    elif required_mt_frags is not None:

        # probability to observe required_mt_frags or more mutant fragments of each of the minimally called mutations
        # for detection
        mt_prob_mt_pos = binom.sf(k=required_mt_frags - 1, n=n_genomes_total, p=mt_prob)

        # probability to observe required_mt_frags or more mutant fragments at positions not mutated in the tumor
        mt_prob_wt_pos = binom.sf(k=required_mt_frags - 1, n=n_genomes_total, p=seq_err_prob)

        # sum probabilities of combinations in which n_min_det_muts mutations can be detected
        probs = np.zeros((n_min_det_muts + 1, len(hge_tumors)))
        for det_muts in range(n_min_det_muts + 1):

            # observe a mutation with at least required_mt_frags fragments at a mutated position in the tumor
            if det_muts == n_min_det_muts:
                probs[det_muts, :] = binom.sf(k=det_muts - 1, n=n_muts_cancer, p=mt_prob_mt_pos)
            else:
                probs[det_muts, :] = binom.pmf(k=det_muts, n=n_muts_cancer, p=mt_prob_mt_pos)

            # observe a mutation with at least required_mt_frags fragments at a position not mutated in the tumor
            probs[det_muts, :] *= binom.sf(k=n_min_det_muts - det_muts - 1, n=panel_size - n_muts_cancer,
                                           p=mt_prob_wt_pos)

        # sum each column which denotes the probability that at least X mutations are detected
        det_prob = np.sum(probs, axis=0)
        logger.debug(f'Probability to observe at least {required_mt_frags} mutant fragments at the {n_min_det_muts}th '
                     + f'most mutated basepair: {det_prob}')

        return det_prob

    else:
        # extreme maximum of mutated fragments that could be expected under any conditions
        n_max_frags = int(round(max(binom.ppf(1.0 - 1e-10, n=n_genomes_total, p=seq_err_prob)))) + 2

        prob_k_more = np.zeros((n_max_frags, len(hge_tumors)))
        pvals = np.zeros_like(prob_k_more)

        for k in range(n_max_frags, 0, -1):
            # probability to observe k or more mutant fragments of each of the minimally called mutations for detection
            mt_prob_mt_pos = binom.sf(k=k - 1, n=n_genomes_total, p=mt_prob)

            # probability to observe k or more mutant fragments at positions not mutated in the tumor
            mt_prob_wt_pos = binom.sf(k=k - 1, n=n_genomes_total, p=seq_err_prob)

            # sum probabilities of combinations in which n_min_det_muts mutations can be detected
            probs = np.zeros((n_min_det_muts + 1, len(hge_tumors)))
            for det_muts in range(n_min_det_muts + 1):

                # observe a mutation with at least k fragments at a mutated position in the tumor
                if det_muts == n_min_det_muts:
                    probs[det_muts, :] = binom.sf(k=det_muts - 1, n=n_muts_cancer, p=mt_prob_mt_pos)
                else:
                    probs[det_muts, :] = binom.pmf(k=det_muts, n=n_muts_cancer, p=mt_prob_mt_pos)

                # observe a mutation with at least k fragments at a position not mutated in the tumor
                probs[det_muts, :] *= binom.sf(k=n_min_det_muts - det_muts - 1, n=panel_size - n_muts_cancer,
                                               p=mt_prob_wt_pos)

            # sum each column which denotes the probability that at least X mutations are detected
            prob_k_more[k - 1, :] = np.sum(probs, axis=0)
            logger.debug(f'Probability to observe at least {k} mutant fragments at the {n_min_det_muts}th '
                         + f'most mutated basepair: {np.mean(prob_k_more[k - 1, :]):.3e}')

            # probability to observe k or more mutant fragments at a basepair due to sequencing errors
            pvals[k - 1, :] = binom.sf(k=k - 1, n=n_genomes_total, p=seq_err_prob)

        # detection probability is equivalent to probability that p-value less or equal to p-value threshold is observed
        # take the minimal number of mutated fragments required that achieve a p-value lower or equal to the threshold
        required_mt_frags = np.argmin(pvals > pval_th, axis=0)
        # logger.info(f'{n_min_det_muts} muts required for detection requires: mean {np.mean(required_mt_frags)}, '
        #             + f'median {np.median(required_mt_frags)} mutant fragments.')
        det_prob = np.take_along_axis(prob_k_more, np.expand_dims(required_mt_frags, axis=0), axis=0)[0, :]

        return det_prob, required_mt_frags + 1


def calculate_sensitivity(b, d, q_d, epsilon, n_min_det_muts, panel_size, n_muts_cancer,
                          dna_conc_gamma_params=settings.FIT_GAMMA_PARAMS, tube_size=settings.TUBE_SIZE,
                          seq_err=settings.SEQUENCING_ERROR_RATE, seq_eff=settings.SEQUENCING_EFFICIENCY,
                          resolution=100, hge_tumors=None, tumor_sizes=None, pval_th=None, required_mt_frags=None):
    """
    Calculate the sensitivities to detect a tumor if there are hge_tumor hGE circulating in the entire bloodstream or
    a tumor of certain size is present and sheds ctDNA hGE according to a poisson-distribution
    :param b: cancer cell birth rate
    :param d: cancer cell death rate
    :param q_d: ctDNA hGE shedding rate
    :param epsilon: cfDNA elimination rate (proportional to cfDNA half-life time)
    :param n_min_det_muts: array of minimally required called mutation numbers for a positive test
    :param panel_size: sequencing panel size
    :param n_muts_cancer: number of mutations covered by the panel that are clonally present in the tumor
    :param dna_conc_gamma_params: plasma DNA concentration is sampled from the given gamma distribution parameters
    :param tube_size: amount of blood that is sampled per liquid biopsy [liters]
    :param seq_err: sequencing error rate per base-pair
    :param seq_eff: sequencing efficiency
    :param resolution: number of points of discretized distribution considered for calculations
    :param hge_tumors: number of normal hGE circulating in the entire bloodstream
                        (will be multiplied by two to account for diploid genomes)
    :param tumor_sizes: if hge_tumors is None, then an array of tumor sizes needs to be provided
    :param pval_th: p-value threshold to call individual mutations as present
    :param required_mt_frags: minimum number of mutated fragments required to call mutation at a given position
    :return: array of sensitivities for the given tumor sizes or fixed hGEs
    """

    if hge_tumors is None and tumor_sizes is None:
        raise RuntimeError('Either an array with ctDNA hGE in the bloodstream or '
                           'an array of tumor sizes needs to be given.')

    if pval_th is None and required_mt_frags is None:
        err_str = ('Either a p-value threshold or a number of required mutant fragments is needed '
                   'to compute detection probability.')
        logger.error(err_str)
        raise RuntimeError(err_str)

    elif pval_th is not None and required_mt_frags is not None:
        err_str = ('Only a p-value threshold or a number of required mutant fragments should be given '
                   'to compute detection probability. Not both.')
        logger.error(err_str)
        raise RuntimeError(err_str)

    pcts = np.linspace(1.0 / resolution, 1 - 1.0 / resolution, resolution - 1)

    # amount of blood in an average human
    blood_amount = settings.BLOOD_AMOUNT
    # fraction of plasma in blood
    plasma_fraction = settings.PLASMA_FRACTION
    plasma_ml = blood_amount * plasma_fraction * 1000

    # assuming 5 liters of blood
    sample_fraction = tube_size / blood_amount * seq_eff

    cfdna_concs = np.array([
        sp.stats.gamma.ppf(pct, dna_conc_gamma_params['shape'], loc=0, scale=dna_conc_gamma_params['scale'])
        for pct in pcts])

    cfdna_hge_per_ml = cfdna_concs / settings.DIPLOID_GE_WEIGHT_ng
    cfdna_hge_total = np.rint(cfdna_hge_per_ml * plasma_ml)
    #     logger.info('cfDNA hGEs in bloodstream distribution: '+','.join(f'{hge}' for hge in cfdna_hge_total))

    # calculate circulating tumor hGE in blood stream
    r = b - d
    if hge_tumors is None:
        hge_tumors = np.zeros((len(tumor_sizes), len(pcts)))
        for k, tumor_size in enumerate(tumor_sizes):
            hge_mean = get_bms_at_size(tumor_size, q_d * d, r, epsilon)
            hge_tumors[k, :] = np.array([sp.stats.poisson.ppf(pct, hge_mean) for pct in pcts])

    else:
        tumor_sizes = np.array([hge * (epsilon + r) / (d * q_d) for hge in hge_tumors])
        hge_tumors = np.expand_dims(hge_tumors, axis=1)

    # calculate sensitivity for a fixed specificity with a varying numbers of required mutations for detection
    sensitivities = np.zeros((len(hge_tumors)))

    if pval_th is not None:

        for k, hges in enumerate(hge_tumors):

            tumor_test = np.zeros((len(hges) * len(cfdna_hge_total)))
            required_mt_frags = np.zeros((len(hges) * len(cfdna_hge_total)))
            # combine all percentiles of hGEs with all percentiles of plasma DNA concentrations
            for j in range(len(cfdna_hge_total)):
                tumor_test[j * len(hges):(j + 1) * len(hges)], required_mt_frags[j * len(hges):(j + 1) * len(hges)] = \
                    calculate_detection_probability(n_min_det_muts, panel_size, n_muts_cancer,
                                                    hges, cfdna_hge_total[j], seq_err, sample_fraction, pval_th=pval_th)

            # Calculate true positive rate (TPR), sensitivity, recall
            sensitivities[k] = sum(tumor_test) / float(len(tumor_test))
            logger.info(
                f'{n_min_det_muts} called muts required for detection need: mean {np.mean(required_mt_frags):.3f}, '
                + f'median {np.median(required_mt_frags)} mutant fragments.')

            logger.info(f'Sensitivity for tumor size {tumor_sizes[k]:.1e} (mean {np.mean(hges):.1f} hGE, '
                        + f'{n_min_det_muts} called muts): {sensitivities[k]:.3%} (pv {pval_th:.3e})')

    else:

        for k, hges in enumerate(hge_tumors):

            tumor_test = np.zeros((len(hges) * len(cfdna_hge_total)))
            # combine all percentiles of hGEs with all percentiles of plasma DNA concentrations
            for j in range(len(cfdna_hge_total)):
                tumor_test[j * len(hges):(j + 1) * len(hges)] = \
                        calculate_detection_probability(n_min_det_muts, panel_size, n_muts_cancer,
                                                        hges, cfdna_hge_total[j], seq_err, sample_fraction,
                                                        required_mt_frags=required_mt_frags)

            # Calculate true positive rate (TPR), sensitivity, recall
            sensitivities[k] = sum(tumor_test) / float(len(tumor_test))

            logger.info(f'Sensitivity for tumor size {tumor_sizes[k]:.1e} (mean {np.mean(hges):.1f} hGE, '
                        + f'{n_min_det_muts} called muts): {sensitivities[k]:.3%} (pv {pval_th:.3e})')

    return sensitivities

def calculate_specificity(n_min_det_muts, panel_size, dna_conc_gamma_params=settings.FIT_GAMMA_PARAMS,
                          tube_size=settings.TUBE_SIZE, seq_err=settings.SEQUENCING_ERROR_RATE,
                          seq_eff=settings.SEQUENCING_EFFICIENCY, resolution=100, pval_th=None, required_mt_frags=None):
    """
    Calculate the specificity for a cancer detection test at the given plasma DNA concentration distribution
    :param n_min_det_muts: minimal number of required called mutations for a positive test
    :param panel_size: sequencing panel size
    :param dna_conc_gamma_params: plasma DNA concentration is sampled from the given gamma distribution parameters
    :param tube_size: amount of blood that is sampled per liquid biopsy [liters]
    :param seq_err: sequencing error rate per base-pair
    :param seq_eff: sequencing efficiency
    :param resolution: number of points of discretized distribution considered for calculations
    :param pval_th: p-value threshold to call a mutation as present
    :param required_mt_frags: minimum number of mutated fragments required to call mutation at a given position
    :return: specificity
    """
    if pval_th is None and required_mt_frags is None:
        err_str = ('Either a p-value threshold or a number of required mutant fragments is needed '
                   'to compute detection probability.')
        logger.error(err_str)
        raise RuntimeError(err_str)

    elif pval_th is not None and required_mt_frags is not None:
        err_str = ('Only a p-value threshold or a number of required mutant fragments should be given '
                   'to compute detection probability. Not both.')
        logger.error(err_str)
        raise RuntimeError(err_str)

    # pcts = np.linspace(0.01, 0.99, 99)
    pcts = np.linspace(1.0 / resolution, 1 - 1.0 / resolution, resolution - 1)

    # amount of blood in an average human
    blood_amount = settings.BLOOD_AMOUNT
    # fraction of plasma in blood
    plasma_fraction = settings.PLASMA_FRACTION
    plasma_ml = blood_amount * plasma_fraction * 1000

    # assuming 5 liters of blood
    sample_fraction = tube_size / blood_amount * seq_eff

    cfdna_concs = np.array([
        sp.stats.gamma.ppf(pct, dna_conc_gamma_params['shape'], loc=0, scale=dna_conc_gamma_params['scale'])
        for pct in pcts])

    cfdna_hge_per_ml = cfdna_concs / settings.DIPLOID_GE_WEIGHT_ng
    cfdna_hge_total = np.rint(cfdna_hge_per_ml * plasma_ml)

    # calculate specificity for a given p-value threshold and number of required mutations for detection
    tumor_test = np.zeros(len(pcts))
    for j in range(len(pcts)):
        if pval_th is not None:
            tumor_test[j], _ = calculate_detection_probability(
                n_min_det_muts, panel_size, 0,
                np.zeros(1), cfdna_hge_total[j], seq_err, sample_fraction, pval_th=pval_th)
        else:
            tumor_test[j], = calculate_detection_probability(
                n_min_det_muts, panel_size, 0,
                np.zeros(1), cfdna_hge_total[j], seq_err, sample_fraction, required_mt_frags=required_mt_frags)

    # Calculate true positive rate (TPR), sensitivity, recall
    specificity = 1.0 - (sum(tumor_test) / float(len(tumor_test)))

    logger.info(f'Specificity for {n_min_det_muts} called muts: {specificity:.1%} (pv {pval_th:.3e})')

    return specificity


def calculate_ppv(fpr, tpr, incidence_rate):
    """
    Calculate positive predictive value
    :param fpr: false positive rate
    :param tpr: true positive rate (sensitivity)
    :param incidence_rate: incidence rate or prevalence of a disease
    :return:
    """
    return tpr * incidence_rate / ((fpr * (1.0 - incidence_rate)) + (tpr * incidence_rate))


def calculate_npv(specificity, tpr, incidence_rate):
    """
    Calculate negative predictive value
    :param specificity:
    :param tpr: true positive rate (sensitivity)
    :param incidence_rate: incidence rate or prevalence of a disease
    :return:
    """
    return specificity * (1.0 - incidence_rate) / (specificity * (1.0 - incidence_rate) + (1.0 - tpr) * incidence_rate)
