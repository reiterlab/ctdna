#!/usr/bin/python
"""Taking samples from the bloodstream"""

import logging
import random
import numpy as np
import pandas as pd
from collections import Sequence

import ctdna.settings as settings
from ctdna.detection import calculate_bm_pvalues
from ctdna.utils import stats_string, Output, cells_diameter

__date__ = 'March 5, 2020'
__author__ = 'Johannes REITER'

# get logger
logger = logging.getLogger(__name__)


def perform_longitudinal_sampling(sim_tumor_fps, sampling_freq, fixed_sampling_times, wt_hges_per_ml, det_th,
                                  muts_per_cancer, tube_size, panel_size, seq_err, seq_eff,
                                  imaging_det_size=None, symptomatic_size=None, diagnosis_size=None, n_replications=1,
                                  no_ctDNA=False):
    """
    Take liquid biopsies with the given frequency or at the given times and then see whether the biomarker is above
    the detection threshold
    :param sim_tumor_fps: file path to simulated tumors
    :param sampling_freq: frequency of testing in days (e.g. 365 for annual sampling)
    :param fixed_sampling_times: fixed schedule of sampling (e.g., 1, 366, 731, etc.) for all cases
    :param wt_hges_per_ml: array of wildtype biomarker molecules (haploid genome equivalents) per plasma ml
    :param det_th: instance of detection class defining the threshold for biomarker presence classification
    :param muts_per_cancer: number of mutations in the lesion that are also covered by the panel
    :param tube_size: liquid biopsy sampling tube size in blood liters
    :param panel_size: number of sequenced basepairs on panel
    :param seq_err: sequencing error rate per basepair
    :param seq_eff: sequencing efficiency corresponds to fraction of molecules in sample that get sequenced
    :param imaging_det_size: imaging detection size to compare lead time to imaging-based relapse detection
    :param symptomatic_size: size when tumors become symptomatic and hence will be diagnosed without an additional test
    :param diagnosis_size: size when tumors become symptomatic and hence will be diagnosed without an additional test
    :param n_replications: number of replications with randomized sampling start time
    :param no_ctDNA: if true longitudinal sampling is performed assuming no ctDNA in the bloodstream
    :return: arrays of screening detection sizes, screening detection times, symptomatic times, and lead_times
    """

    lb_det_sizes = list()
    lb_det_times = list()
    lb_det_pos_tests = list()
    symptomatic_det_times = list()
    lead_times = list()

    screening_start_size = 0  # 1e3
    logger.info(f'Imitate cancer early detection in {len(sim_tumor_fps)} subjects through longitudinal sampling '
                + f'with panel of size {panel_size:.1e} covering {muts_per_cancer} mutations, '
                + f'seq err {seq_err:.1e}, seq eff {seq_eff:.1%}, and tube size {tube_size} liters.')
    logger.info(f'Diagnosis threshold: {det_th}')

    if fixed_sampling_times is not None:
        fixed_sampling_times = np.array(fixed_sampling_times)

    for tumor_id, tumor_fp in enumerate(sorted(sim_tumor_fps), 1):

        logger.debug('Reading dynamics from file: {}'.format(tumor_fp))
        # evolution of tumor data and its biomarker levels
        df_tumor = pd.read_csv(tumor_fp, index_col='Time')
        if no_ctDNA:
            df_tumor['Biomarker_total'].values[:] = 0

        # calculate MRD (minimal residual disease) time
        mrd_time = max(0, df_tumor[df_tumor[Output.col_lesion_size] > screening_start_size].index[0])

        # determine annual screening times, first screening date is random
        for repl_id in range(n_replications):
            if fixed_sampling_times is None:
                if sampling_freq is None:
                    raise RuntimeError('Either sampling times or sampling frequency need to be provided '
                                       'for longitudinal sampling!')
                sampling_start_time = mrd_time + random.randint(0, sampling_freq)
                sampling_times = sampling_start_time + np.arange(0, int(df_tumor.index[-1]), sampling_freq)
            else:
                sampling_times = fixed_sampling_times[fixed_sampling_times < int(df_tumor.index[-1])]
                if len(sampling_times) < len(fixed_sampling_times):
                    logger.debug('Some provided sampling times are outside of the simulated tumor time: '
                                 + ', '.join(str(t) for t in
                                             fixed_sampling_times[fixed_sampling_times >= int(df_tumor.index[-1])]))
            if len(sampling_times) == 0:
                logger.warning(f'No valid sampling times for tumor case {tumor_id}. Max time is {df_tumor.index[-1]}:'
                               + f' {tumor_fp}')
                continue

            logger.debug('Sample tumor at times: ' + ', '.join(str(t) for t in sampling_times))
            lesion_sizes = list()
            biomarker_levels = list()
            det_times = list()

            # SCREENING: compute clinical detection by reaching some given size in combination with screening
            if symptomatic_size is not None and symptomatic_size <= df_tumor['LesionSize'].loc[df_tumor.index[-1]]:
                symptomatic_det_time = min(df_tumor[df_tumor['LesionSize'] >= symptomatic_size].index.values)
            else:
                symptomatic_det_time = None

            # SCREENING: compute clinical detection by reaching some given size
            if diagnosis_size is not None and diagnosis_size <= df_tumor['LesionSize'].loc[df_tumor.index[-1]]:
                diagnosis_time = min(df_tumor[df_tumor['LesionSize'] >= diagnosis_size].index.values)
                symptomatic_det_time = diagnosis_time
            else:
                diagnosis_time = None

            # RELAPSE: compute clinical detection by imaging following the same frequency
            if imaging_det_size is not None:
                symptomatic_det_time = None
                # if imaging follows the same sampling frequency as liquid biopsy lead time distributions are biased
                img_scr_times = (mrd_time + random.randint(0, sampling_freq)
                                 + np.arange(0, int(df_tumor.index[-1]), sampling_freq))

                for t in img_scr_times:
                    if t in df_tumor.index and df_tumor.loc[t]['LesionSize'] >= imaging_det_size:
                        imaging_det_time = t
                        symptomatic_det_time = imaging_det_time
                        break
                else:
                    imaging_det_time = None
            else:
                imaging_det_time = None

            # symptomatic detection time independent of any possible early detection through liquid biopsy
            symptomatic_det_times.append(symptomatic_det_time if symptomatic_det_time is not None else imaging_det_time)

            # readout biomarker levels at the selected sampling times
            for t in sampling_times:
                if t in df_tumor.index:

                    # no need to evaluate biomarker levels after the tumor was detected by presenting through symptoms
                    if symptomatic_size is not None and df_tumor.loc[t][Output.col_lesion_size] > symptomatic_size:
                        if len(biomarker_levels) == 0:
                            # tumor became symptomatic before any samplings were performed
                            logger.debug(
                                f'Tumor case {tumor_id} became symptomatic before any samplings were performed.')
                        break

                    else:
                        lesion_sizes.append(df_tumor.loc[t][Output.col_lesion_size])
                        biomarker_levels.append(df_tumor.loc[t][Output.col_bm_amount])
                        det_times.append(t)

            # perform virtual screening for all valid times where tumor is not yet symptomatic
            if len(biomarker_levels):
                # returns boolean array with positive or negative test results for each provided biomarker level
                tumor_tests = simulate_virtual_detection(
                    biomarker_levels, det_th, muts_per_tumor=muts_per_cancer, tube_size=tube_size,
                    panel_size=panel_size, seq_err=seq_err, seq_eff=seq_eff,
                    wt_hge_per_ml=wt_hges_per_ml[tumor_id - 1, repl_id], longitudinal_sampling=True)

                # get indices where test was positive for the first time
                pos_test_idcs = np.where(tumor_tests)[0]

            # if there are no biomarker levels for the given sampling times, there can be no positive test results
            else:
                pos_test_idcs = list()

            if len(pos_test_idcs):
                # track the number of tests until the first positive one
                lb_det_pos_tests.append(pos_test_idcs[0]+1)
                lb_det_sizes.append(lesion_sizes[pos_test_idcs[0]])
                lb_det_times.append(det_times[pos_test_idcs[0]])
                if symptomatic_det_time is not None:    # screening
                    lead_times.append(symptomatic_det_time - det_times[pos_test_idcs[0]])
                elif imaging_det_time is not None:      # relapse
                    lead_times.append(imaging_det_time - det_times[pos_test_idcs[0]])
                else:
                    lead_times.append(np.nan)

                logger.debug(
                    f'Tumor was first detected at a size of {lesion_sizes[pos_test_idcs[0]]:.3e} after '
                    + f'{det_times[pos_test_idcs[0]]} days at {pos_test_idcs[0]+1}th test with a lead '
                    + f'time of {lead_times[-1]} days and a biomarker level of {biomarker_levels[pos_test_idcs[0]]}.')

            else:
                lb_det_sizes.append(np.nan)
                lb_det_times.append(np.nan)
                lb_det_pos_tests.append(np.nan)
                # tumor was not simulated to a large enough size to be detected by a liquid biopsy
                if symptomatic_size is None:
                    logger.error('ERROR: Tumor was never detected: {}'.format(tumor_fp))
                    lead_times.append(np.nan)

                # tumor was not simulated to a large enough size to be detected by a liquid biopsy or become symptomatic
                elif symptomatic_size > df_tumor.loc[df_tumor.index[-1]][Output.col_lesion_size]:
                    logger.error('ERROR: Tumor was never detected: {}'.format(tumor_fp))
                    lead_times.append(np.nan)

                # tumor was not detected by a liquid biopsy and got detected through symptoms
                else:
                    lead_times.append(0.0)

        if len(sim_tumor_fps) > 999 and tumor_id * 100 % len(sim_tumor_fps) == 0 and panel_size > 999:
            logger.info(f'{tumor_id/len(sim_tumor_fps):.0%} Detection sizes [cells]: {stats_string(lb_det_sizes)}')
            logger.info(f'{tumor_id/len(sim_tumor_fps):.0%} Detection times [days]: {stats_string(lb_det_times)}')

    logger.info(f'Performed longitudinal sampling in {len(lb_det_times)} instances '
                + f'({len(sim_tumor_fps)} tumor cases with {n_replications} replications).')
    logger.info(f'Detection sizes [cells]: {stats_string(lb_det_sizes)}')
    logger.info(f'Detection diameters [cm]: {stats_string(cells_diameter(np.array(lb_det_sizes)))}')
    logger.info(f'Detection times [days]: {stats_string(lb_det_times)}')
    logger.info(f'Number of performed tests until first positive: {stats_string(lb_det_pos_tests)}')

    if imaging_det_size or symptomatic_size or diagnosis_size:
        sympt_times_info = ''
        if imaging_det_size is not None:
            sympt_times_info = f'Imaging detection times for detection size {imaging_det_size:.1e}: '
        elif symptomatic_size is not None:
            sympt_times_info = f'Symptomatic times for symptomatic size {symptomatic_size:.1e}: '
        elif diagnosis_size is not None:
            sympt_times_info = f'Diagnosis times for sampling and a diagnosis size of {diagnosis_size:.1e}: '

        sympt_times_info += stats_string(symptomatic_det_times)
        logger.info(sympt_times_info)

    lead_times = np.array(lead_times)
    if imaging_det_size or diagnosis_size or symptomatic_size:
        lead_times_info = get_lead_time_info(imaging_det_size, diagnosis_size, symptomatic_size, lead_times)
        logger.info(lead_times_info)

    return lb_det_sizes, lb_det_times, symptomatic_det_times, lead_times


def get_lead_time_info(imaging_det_size, diagnosis_size, symptomatic_size, lead_times):
    """
    Return a string with mean, median and IQR statistics of given lead time array
    :param imaging_det_size:
    :param diagnosis_size:
    :param symptomatic_size:
    :param lead_times:
    :return:
    """
    lead_times_info = ''
    if imaging_det_size is not None:
        lead_times_info = f'Two-sided lead time to detection by imaging with size threshold {imaging_det_size:.1e}: '
    elif diagnosis_size is not None:
        lead_times_info = f'Two-sided lead time to becoming diagnosed at size {diagnosis_size:.1e}: '
    elif symptomatic_size is not None:
        lead_times_info = f'One-sided lead time before becoming symptomatic at size {symptomatic_size:.1e}: '
    lead_times_info += (
            f'mean {np.nanmean(lead_times):.1f}, median {np.nanmedian(lead_times):.1f}, '
            + f'25th perc {np.nanpercentile(lead_times, 25):.1f}, '
            + f'75th perc {np.nanpercentile(lead_times, 75):.1f}, largest {np.nanmax(lead_times):.1f} days')

    lead_times_info += '; positive lead times in '
    lead_times_info += f'{np.count_nonzero(lead_times[~np.isnan(lead_times)] > 0) / lead_times.size:.1%} cases '
    lead_times_info += f'({np.count_nonzero(np.isnan(lead_times)) / lead_times.size:.1%} NaNs).'

    return lead_times_info


def simulate_virtual_detection(bms, det_th, muts_per_tumor, wt_hge_per_ml, tube_size=settings.TUBE_SIZE,
                               panel_size=settings.PANEL_SIZE, seq_err=settings.SEQUENCING_ERROR_RATE,
                               seq_eff=settings.SEQUENCING_EFFICIENCY, longitudinal_sampling=False):
    """
    Perform virtual screening tests where search for multiple possibly mutated positions
    :param bms: array of mutant biomarker levels in the bloodstream
    :param det_th: instance of detection class defining the threshold for biomarker presence classification
    :param muts_per_tumor: average number of covered mutations per cancer
    :param tube_size: liquid biopsy sampling tube size in liters
    :param wt_hge_per_ml: list of wildtype biomarkers (genome equivalents) per plasma ml
    :param panel_size: number of sequenced basepairs on panel
    :param seq_err: sequencing error rate per base; see Newman et al, Nature Biotechnology 2016
    :param seq_eff: fraction of molecules in sample that get sequenced; see Chabon et al, Nature 2020
    :param longitudinal_sampling: if this flag is True, then screening can be stopped after the first positive tests
    :return: array with number of detected mutation if min_det_muts is None OR
             boolean array with positive or negative test results for each provided biomarker level
    """

    # assuming 5 liters of blood and 55% of plasma
    sampled_fraction = tube_size / settings.BLOOD_AMOUNT * seq_eff
    plasma_ml = settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000

    # biomarker wildtype molecules (e.g., whole genome equivalents) per mL of plasma
    # if not isinstance(wge_per_ml, (Sequence, np.ndarray)):
    #     biomarker_wt_sum = plasma_ml * wge_per_ml
    #     logger.debug('WT biomarkers in plasma: {}'.format(biomarker_wt_sum))
    # else:
    #     biomarker_wt_sum = None

    # compute wildtype genome equivalents per liquid biopsy
    wt_hges = plasma_ml * wt_hge_per_ml

    # perform virtual cancer detection tests by running through the simulated circulating biomarker in the bloodstream
    detected_mutations = np.zeros(len(bms))
    for sub_id, ctdna_hge in enumerate(bms):

        # sum the random biomarker amount from the cancer plus the normal level
        normal_genomes = 2 * wt_hges[sub_id % len(wt_hges)] + ctdna_hge # second copy of ctDNA is assumed to be wildtype
        total_genomes = normal_genomes + ctdna_hge

        # compute discrete number of biomarker molecules that are sampled
        # hge_sampled = round(total_genomes * sampled_fraction)
        hge_sampled = np.random.binomial(total_genomes, sampled_fraction)
        # this number could also be further broken up if instead of genome equivalents genome fragments are considered
        # TODO assume that each basepair is on different genome fragment
        # n_sampled_bms = np.random.binomial(total_genomes, sample_fraction, size=panel_size)

        sampled_bm_mts1, sampled_bm_mts2 = simulate_snv_sequencing(
            panel_size, muts_per_tumor, ctdna_hge, total_genomes, hge_sampled, seq_err)

        # if at least one mutant fragment is required, remove all zeros
        if det_th.pval_th < 1 or det_th.min_supp_reads > 0 or det_th.min_det_vaf > 0:
            sampled_bm_mts = np.concatenate((sampled_bm_mts1[np.nonzero(sampled_bm_mts1)],
                                             sampled_bm_mts2[np.nonzero(sampled_bm_mts2)]), axis=0)
        else:
            sampled_bm_mts = np.concatenate((sampled_bm_mts1, sampled_bm_mts2), axis=0)

        # set frequency of expected background mutant fragments due to sequencing error
        pvals = calculate_bm_pvalues(sampled_bm_mts, hge_sampled, seq_err)
        result_matrix = np.column_stack((pvals <= det_th.pval_th, sampled_bm_mts >= det_th.min_supp_reads,
                                         sampled_bm_mts/hge_sampled >= det_th.min_det_vaf))

        detected_mutations[sub_id] = sum(result_matrix[:, 0] & result_matrix[:, 1] & result_matrix[:, 2])
        # if early detection is run in longitudinal mode, we can stop after the first positive tests
        if (longitudinal_sampling and det_th.min_det_muts is not None
                and detected_mutations[sub_id] >= det_th.min_det_muts):
            break

    # logger.debug('Mean detected artifacts per subject: {:.4e}'.format(np.mean(detected_artifacts)))
    if det_th.min_det_muts is None:
        logger.debug('Detected at least one mutation in {:.3%} ({}/{}) tumors.'.format(
            sum(detected_mutations > 0) / len(bms), sum(detected_mutations), len(bms)))
        return detected_mutations
    else:
        screening_test = detected_mutations >= det_th.min_det_muts
        logger.debug(f'Detected {sum(screening_test) / len(bms):.3%} ({sum(screening_test)}/{len(bms)}) tumors.')
        return screening_test


def simulate_snv_sequencing(panel_size, muts_per_tumor, n_hge_tumor, n_genomes_total, n_hge_sampled, seq_err):
    """
    Sequence independent somatic single nucleotide variants (SNVs) with a sequencing panel of a given size and
    given circulating mixture of mutant and wildtype copies per basepair
    :param panel_size: size of sequencing panel
    :param muts_per_tumor: true number
    :param n_hge_tumor: int or array_like of ints, total number of ctDNA haploid genome equivalents in the bloodstream
    :param n_genomes_total: int or array_like of ints, total number of circulating genome equivalents
    :param n_hge_sampled: int or array_like of ints, number of sampled haploid genome equivalents in liquid biopsy tube
    :param seq_err: sequencing error rate per base pair
    :return: arrays of number of mutant reads for wildtype positions and mutant type positions
    """
    # calculate VAFs (variant allele frequencies)
    bm_mt_fraction = n_hge_tumor / n_genomes_total

    if (not isinstance(n_hge_tumor, (Sequence, np.ndarray))
            and not isinstance(n_genomes_total, (Sequence, np.ndarray))
            and not isinstance(n_hge_sampled, (Sequence, np.ndarray))):
        # single subject
        n_samples = 1
    else:
        # multiple subjects were given
        n_samples = len(n_hge_sampled)

    # calculate the expected fraction of the mutant biomarker at a mutant position/basepair of ctDNA
    obs_bm_mt_mt_bp_fraction = bm_mt_fraction * (1 - seq_err) + (1 - bm_mt_fraction) * seq_err

    # compute observed number of mutant fragments from actual mutations in cancer cells
    if muts_per_tumor > 0:
        sampled_bm_mts1 = np.random.binomial(n_hge_sampled, obs_bm_mt_mt_bp_fraction,
                                             size=(muts_per_tumor, n_samples))
    else:
        sampled_bm_mts1 = np.zeros((0, n_samples))

    # compute observed number of mutant fragments due to sequencing errors from cfDNA and wildtype ctDNA
    if panel_size - muts_per_tumor > 0:
        # expected fraction of the mutant biomarker at a wildtype position/basepair of ctDNA is equivalent to
        # the sequencing normal cfDNA with mimicing sequencing errors
        sampled_bm_mts2 = np.random.binomial(n_hge_sampled, seq_err,
                                             size=(panel_size - muts_per_tumor, n_samples))
    else:
        sampled_bm_mts2 = np.zeros((panel_size - muts_per_tumor, n_samples))

    if n_samples == 1 and not isinstance(n_hge_sampled, (Sequence, np.ndarray)):
        return sampled_bm_mts1[:, 0], sampled_bm_mts2[:, 0]
    else:
        return sampled_bm_mts1, sampled_bm_mts2


def take_liquid_biopsies(ctdna_hge_het, wt_hges_per_ml, tube_size=settings.TUBE_SIZE):
    """
    Take liquid biopsies from circulating biomarkers
    :param ctdna_hge_het: array of biomarker levels; heterozygous haploid genome equivalents
    :param wt_hges_per_ml: array of wildtype biomarkers (haploid genome equivalents) per plasma ml
    :param tube_size: sample size measured in liters, half of it is assumed to be plasma [default 0.015 l]
    :return: tuple of biomarker fraction in circulation, sampled number of biomarkers, biomarker fraction in sample
    """

    # assuming 5 liters of blood and 55% of plasma
    sample_fraction = tube_size / settings.BLOOD_AMOUNT
    plasma_ml = settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000

    logger.info('Biomarkers statistics: {:.3e} median, {:.3e} mean'.format(
        np.median(ctdna_hge_het), np.mean(ctdna_hge_het)))
    logger.info('Sampling {:.3%} of the human plasma with a tube of size {:.1f} ml.'.format(
        sample_fraction, tube_size * 1000))

    # calculate the number of haploid genome equivalents in all plasma
    biomarker_wts = wt_hges_per_ml * plasma_ml
    logger.info('Biomarker frequency in whole plasma: {:.5%} median, {:.5%} mean'.format(
        np.median(ctdna_hge_het / biomarker_wts), np.mean(ctdna_hge_het / biomarker_wts)))

    sampled_bms = np.empty(len(ctdna_hge_het))
    sampled_bms_vafs = np.empty(len(ctdna_hge_het))
    total_bms_vafs = np.empty(len(ctdna_hge_het))

    for i, n_bm_mt in enumerate(ctdna_hge_het):
        n_bm = biomarker_wts[i] + n_bm_mt + n_bm_mt  # second copy of shed ctDNA is not mutated
        n_exp_bm_sample = int(n_bm * sample_fraction)
        bm_mt_fraction = n_bm_mt / n_bm
        total_bms_vafs[i] = bm_mt_fraction
        sampled_bms[i] = np.random.binomial(n_exp_bm_sample, bm_mt_fraction)
        sampled_bms_vafs[i] = sampled_bms[i] / n_exp_bm_sample
        # logger.debug(f'Sampled {sampled_bms[i]} mtBM molecules at VAF {sampled_bms_vafs[i]:.4%} from overall ctDNA '
        #              + f'concentration of {bm_mt_fraction:.3%} and frequency of {n_bm_mt:.3e}.')

    n_cancers_2_mt_bms = sum(1 for i in sampled_bms if i >= 2)
    logger.info('Mean of {:.4g} and median of {:.4g} ctDNA whole genome equivalents in sample.'.format(
        np.mean(sampled_bms), np.median(sampled_bms)))
    logger.info('At least 2 mutant ctDNA equivalents in sample for {:.3%} ({}/{}) of tumors.'.format(
        n_cancers_2_mt_bms / len(ctdna_hge_het), n_cancers_2_mt_bms, len(ctdna_hge_het)))

    return total_bms_vafs, sampled_bms, sampled_bms_vafs
