#!/usr/bin/python
"""Taking liquid biopsies"""

import logging
import random
from collections import Sequence
import numpy as np
import pandas as pd

import cbmlb.settings as settings
from cbmlb.detection import calculate_bm_pvalues

__date__ = 'March 5, 2020'
__author__ = 'Johannes REITER'

# get logger
logger = logging.getLogger(__name__)


def perform_longitudinal_sampling(sim_tumor_fps, sampling_freq, fixed_sampling_times, wGEs_per_ml, det_th,
                                  muts_per_cancer, tube_size, panel_size, seq_err, seq_eff,
                                  imaging_det_size=None, symptomatic_size=None, diagnosis_size=None, n_replications=1):
    """
    Take liquid biopsies with the given frequency or at the given times and then see whether the biomarker is above
    the detection threshold
    :param sim_tumor_fps: file path to simulated tumors
    :param sampling_freq: frequency of testing in days (e.g. 365 for annual sampling)
    :param fixed_sampling_times: fixed schedule of sampling (e.g., 1, 366, 731, etc.) for all cases
    :param wGEs_per_ml: wildtype biomarker molecules per plasma ml
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
    :return:
    """

    lb_det_sizes = list()
    lb_det_times = list()
    symptomatic_det_times = list()
    lead_times = list()

    screening_start_size = 0  # 1e3
    logger.info(f'Imitate cancer early detection in {len(sim_tumor_fps)} subjects through longitudinal sampling '
                + f'with panel of size {panel_size:.1e} covering {muts_per_cancer} mutations, '
                + f'seq err {seq_err:.1e}, seq eff {seq_eff:.1%}, and tube size {tube_size} liters.')

    for tumor_id, tumor_fp in enumerate(sorted(sim_tumor_fps)):

        logger.debug('Reading dynamics from file: {}'.format(tumor_fp))
        # evolution of tumor data and its biomarker levels
        df_tumor = pd.read_csv(tumor_fp, index_col='Time')
        # calculate MRD (minimal residual disease) time
        mrd_time = max(0, df_tumor[df_tumor.LesionSize > screening_start_size].index[0])

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
                # if logger.isEnabledFor(logging.DEBUG)
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
            symptomatic_det_times.append(symptomatic_det_time if not None else imaging_det_time)

            # readout biomarker levels at the selected sampling times
            for t in sampling_times:
                if t in df_tumor.index:

                    # no need to evaluate biomarker levels after the tumor was detected by presenting through symptoms
                    if symptomatic_size is not None and df_tumor.loc[t]['LesionSize'] > symptomatic_size:
                        if len(biomarker_levels) == 0:
                            # tumor became symptomatic before any samplings were performed
                            logger.debug(
                                f'Tumor case {tumor_id} became symptomatic before any samplings were performed.')
                        break

                    else:
                        lesion_sizes.append(df_tumor.loc[t]['LesionSize'])
                        biomarker_levels.append(df_tumor.loc[t]['BiomarkerLevel'])
                        det_times.append(t)

            # perform virtual screening for all valid times where tumor is not yet symptomatic
            if len(biomarker_levels):
                # returns boolean array with positive or negative test results for each provided biomarker level
                tumor_tests = perform_virtual_detection(
                    biomarker_levels, det_th, muts_per_tumor=muts_per_cancer, tube_size=tube_size,
                    panel_size=panel_size, seq_err=seq_err, seq_eff=seq_eff,
                    biomarker_wt_freq_ml=wGEs_per_ml[tumor_id, repl_id], longitudinal_sampling=True)

                # get indices where test was positive
                pos_test_times_indices = np.where(tumor_tests)[0]

            # if there are no biomarker levels for the given sampling times, there can be no positive test results
            else:
                pos_test_times_indices = list()

            if len(pos_test_times_indices):
                lb_det_sizes.append(lesion_sizes[pos_test_times_indices[0]])
                lb_det_times.append(det_times[pos_test_times_indices[0]])
                if symptomatic_det_time is not None:    # screening
                    lead_times.append(symptomatic_det_time - det_times[pos_test_times_indices[0]])
                elif imaging_det_time is not None:      # relapse
                    lead_times.append(imaging_det_time - det_times[pos_test_times_indices[0]])
                else:
                    lead_times.append(np.nan)

                logger.debug(
                    f'Tumor was first detected at a size of {lesion_sizes[pos_test_times_indices[0]]:.3e} after '
                    + f'{det_times[pos_test_times_indices[0]]} days with a lead time of '
                    + f'{lead_times[-1]} days and a biomarker level of {biomarker_levels[pos_test_times_indices[0]]}.')

            else:
                lb_det_sizes.append(np.nan)
                lb_det_times.append(np.nan)
                # tumor was simulated to a large enough size to be detected by a liquid biopsy
                if symptomatic_size is None:
                    logger.error('ERROR: Tumor was never detected: {}'.format(tumor_fp))
                    lead_times.append(np.nan)

                # tumor was simulated to a large enough size to be detected by a liquid biopsy or become symptomatic
                elif symptomatic_size > df_tumor.loc[df_tumor.index[-1]]['LesionSize']:
                    logger.error('ERROR: Tumor was never detected: {}'.format(tumor_fp))
                    lead_times.append(np.nan)

                # tumor was not detected by a liquid biopsy and got detected through symptoms
                else:
                    lead_times.append(0.0)

    logger.info(f'Performed longitudinal sampling in {len(lb_det_times)} instances '
                + f'({len(sim_tumor_fps)} tumor cases with {n_replications} replications).')
    return lb_det_sizes, lb_det_times, symptomatic_det_times, lead_times


def perform_virtual_detection(bms, det_th, muts_per_tumor, tube_size=settings.TUBE_SIZE,
                              panel_size=settings.PANEL_SIZE, seq_err=settings.SEQUENCING_ERROR_RATE,
                              seq_eff=settings.SEQUENCING_EFFICIENCY,
                              biomarker_wt_freq_ml=settings.NO_WT_BIOMARKERS_ML, longitudinal_sampling=False):
    """
    Perform virtual screening tests where search for multiple possibly mutated positions
    :param bms: array of biomarker levels
    :param det_th: instance of detection class defining the threshold for biomarker presence classification
    :param muts_per_tumor: average number of covered mutations per cancer
    :param tube_size: liquid biopsy sampling tube size in liters
    :param biomarker_wt_freq_ml: either constant number of wildtype biomarkers per plasma ml or list
    :param panel_size: number of sequenced basepairs on panel
    :param seq_err: sequencing error rate per base; see Newman et al, Nature Biotechnology 2016
    :param seq_eff: fraction of molecules in sample that get sequenced; see Chabon et al, Nature 2020
    :param longitudinal_sampling: if this flag is True, then screening can be stopped after the first positive tests
    :return: array with number of detected mutation if min_det_muts is None OR
             boolean array with positive or negative test results for each provided biomarker level
    """

    # assuming 5 liters of blood and 55% of plasma
    sample_fraction = tube_size / settings.BLOOD_AMOUNT * seq_eff
    plasma_ml = settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000

    # biomarker wildtype molecules (e.g., whole genome equivalents) per mL of plasma
    if not isinstance(biomarker_wt_freq_ml, (Sequence, np.ndarray)):
        biomarker_wt_sum = plasma_ml * biomarker_wt_freq_ml
        logger.debug('WT biomarkers in plasma: {}'.format(biomarker_wt_sum))
    else:
        biomarker_wt_sum = None

    # perform virtual cancer detection tests by running through the simulated circulating biomarker in the bloodstream
    # screening_test = []
    detected_mutations = np.zeros(len(bms))
    for sub_id, n_bm_mt in enumerate(bms):

        # sum the random biomarker amount from the cancer plus the normal level
        if biomarker_wt_sum is not None:
            n_bm = biomarker_wt_sum + n_bm_mt
        else:
            n_bm = (plasma_ml * biomarker_wt_freq_ml[sub_id % len(biomarker_wt_freq_ml)]) + n_bm_mt

        # compute discrete number of biomarker molecules that will be sampled
        # this number could also be further broken up if instead of genome equivalents genome fragments are considered
        n_sampled_bm = round(n_bm * sample_fraction)
        # assert n_samples > 0, 'BM WT freq {}, BM MT {}, total BM {}, samples {}'.format(
        #     biomarker_wt_freq_ml[sub_id], n_bm_mt, n_bm, n_samples)
        # calculate VAFs (variant allele frequencies)
        bm_mt_fraction = n_bm_mt / n_bm
        # calculate the expected fraction of the biomarker in the bloodstream
        obs_bm_mt_fraction = bm_mt_fraction * (1 - seq_err) + (1 - bm_mt_fraction) * seq_err

        if n_sampled_bm > 0:
            # compute observed number of mutant fragments from actual mutations
            # sampled mutant biomarkers of n sampled total biomarkers
            if muts_per_tumor > 0:
                sampled_bm_mts1 = np.random.binomial(n_sampled_bm, obs_bm_mt_fraction, size=muts_per_tumor)
            # detected_mutations[sub_id] = len(np.where((sampled_bm_mts1 >= min_supp_reads)
            #                                            & (sampled_bm_mts1/n_samples >= min_det_vaf))[0])
            else:
                sampled_bm_mts1 = np.zeros(0)

            # compute observed number of mutant fragments due to sequencing errors from wildtypes
            if panel_size - muts_per_tumor > 0:
                # sampled mutant biomarkers
                sampled_bm_mts2 = np.random.binomial(n_sampled_bm, seq_err, size=panel_size - muts_per_tumor)
                # detected_artifacts[sub_id] = len(np.where((sampled_bm_mts2 >= min_supp_reads)
                #                                           & (sampled_bm_mts2/n_samples >= min_det_vaf))[0])
                # detected_mutations[sub_id] += detected_artifacts[sub_id]
            else:
                sampled_bm_mts2 = np.zeros(0)

            # if at least one mutant fragment is required, remove all zeros
            if det_th.pval_th < 1 or det_th.min_supp_reads > 0 or det_th.min_det_vaf > 0:
                sampled_bm_mts = np.concatenate((sampled_bm_mts1[np.nonzero(sampled_bm_mts1)],
                                                 sampled_bm_mts2[np.nonzero(sampled_bm_mts2)]), axis=0)
            else:
                sampled_bm_mts = np.concatenate((sampled_bm_mts1, sampled_bm_mts2), axis=0)

            # set frequency of expected background mutant fragments due to sequencing error
            pvals = calculate_bm_pvalues(sampled_bm_mts, n_sampled_bm, seq_err)
            result_matrix = np.column_stack((pvals <= det_th.pval_th, sampled_bm_mts >= det_th.min_supp_reads,
                                             sampled_bm_mts/n_sampled_bm >= det_th.min_det_vaf))

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


def take_liquid_biopsies(bms, tube_size=settings.TUBE_SIZE, biomarker_wt_freq_ml=settings.NO_WT_BIOMARKERS_ML):
    """
    Take liquid biopsies from circulating biomarkers
    :param bms: array of biomarker levels
    :param tube_size: sample size measured in liters, half of it is assumed to be plasma [default 0.015 l]
    :param biomarker_wt_freq_ml: number of wildtype biomarkers per plasma ml
                                 [default 1500 GE, genome equivalents, see Heitzer et al, NRG 2019]
    :return: tuple of biomarker fraction in circulation, sampled number of biomarkers, biomarker fraction in sample
    """

    # assuming 5 liters of blood and 55% of plasma
    sample_fraction = tube_size / settings.BLOOD_AMOUNT

    logger.info('Biomarkers statistics: {:.3e} median, {:.3e} mean'.format(
        np.median(bms), np.mean(bms)))
    logger.info('Sampling {:.3%} of the human plasma with a tube of size {:.1f} ml.'.format(
        sample_fraction, tube_size * 1000))

    # calculate the number of whole genome equivalents in all plasma
    biomarker_wt_sum = biomarker_wt_freq_ml * settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000
    logger.info('Biomarker frequency in whole plasma: {:.5%} median, {:.5%} mean'.format(
        np.median(bms) / biomarker_wt_sum, np.mean(bms) / biomarker_wt_sum))

    sampled_bms = np.empty(len(bms))
    sampled_bms_vafs = np.empty(len(bms))
    total_bms_vafs = np.empty(len(bms))

    for i, n_bm_mt in enumerate(bms):
        n_bm = biomarker_wt_sum + n_bm_mt
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
        n_cancers_2_mt_bms / len(bms), n_cancers_2_mt_bms, len(bms)))

    return total_bms_vafs, sampled_bms, sampled_bms_vafs
