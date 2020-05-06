#!/usr/bin/python
"""Biomarker shedding simulations"""
import logging
import os
import numpy as np
import pandas as pd


from cbmlb.lesion import PT, Precursor
from cbmlb.utils import Output

__date__ = 'October 21, 2018'
__author__ = 'Johannes REITER'

# get logger
logger = logging.getLogger(__name__)

# ########################################### USAGE #########################################
# python3 lesions_sim.py -b 0.1 -d 0.1 -i 1 -n 10 -M 3e8 -T 300010 -o '../data/dynamics/'
# python3 lesions_sim.py -b 0.07 -d 0.07 -i 1 -n 20 -M 3.4e7 -T 1000 -o '../data/dynamics/'
# python3 lesions_sim.py -b 0.14 -d 0.136 -n 100 -M 0 -T 3000 -o '../data/detection/' --qd=0.0 --qb=0.0 --lambda1=2.0e-5 --exact_th=1.0e7
# ###########################################################################################


def sim_n_pt_subjects(n_subjects, b, d, q_d, epsilon, q_b=0.0, lambda_1=0.0, det_size=None, sim_time=None,
                      exact_th=1e4, starting_id=1, dynamics_output_fp=None, distr_output_fp=None):
    """
    Simulate the growth dynamics and the shedding of biomarkers in n subjects
    :param n_subjects: number of independently simulated realizations
    :param b: birth rate of cells per day
    :param d: death rate of cells per day
    :param q_d: shedding rate per cell per day
    :param epsilon: ctDNA elimination rate per day
    :param q_b: biomarker shedding probability per cell division (proliferation)
    :param lambda_1: biomarker shedding probability per unit of time (necrosis)
    :param det_size: primary tumor detection size
    :param sim_time: number of days to simulate the biomarker shedding of the lesion
    :param exact_th: max primary tumor size for exact simulation
    :param starting_id: starting identifier for simulated cases
    :param dynamics_output_fp: export detailed dynamics to the given file path (needs to contain <<>> to be
        replaced with subject ID)
    :param distr_output_fp: save temporary results of the biomarker distribution to the given file path
    :return: array with biomarker level at given detection size of the primary tumor
    """

    bms_at_det_size = list()
    pt_sizes = list()
    ages = list()

    for sub_id in range(starting_id, n_subjects + 1):
        while True:

            # initialize primary tumor
            pt = PT(b, d, q_d, epsilon, exact_th, q_b=q_b, lambda_1=lambda_1)

            if det_size is not None and det_size > 1:
                # simulate tumor until it reaches detection size
                pt_size = pt.sim_to_size(det_size)
            elif sim_time is not None and sim_time > 0:
                # simulate tumor until it reaches an age
                pt_size = pt.sim_for_time(sim_time)
            else:
                raise RuntimeError('Either a detection size or a simulation time needs to be provided!')

            if pt_size > 0:
                logger.debug('PT {} detected with size {:.1e} at age {:.1f} days with {} biomarkers.'.format(
                    sub_id, pt_size, pt.age, pt.bm))

                pt_sizes.append(pt_size)
                ages.append(pt.age)
                bms_at_det_size.append(pt.bm)

                logger.debug('Biomarkers at lesion {}: {:.3e} median, {:.3e} mean'.format(
                    'size {:.1e} (t={:.0e})'.format(det_size, np.mean(ages)) if det_size is not None
                    else 'age {:.1e} (size {:.0e})'.format(sim_time, np.mean(pt_sizes)),
                    np.median(bms_at_det_size), np.mean(bms_at_det_size)))

                if (sub_id / n_subjects * 100) % 5 == 0:
                    logger.info('{:.1%} progress: mean biomarkers at lesion {} with d={:.3f}: {:.4e}, var {:.4e}'
                                .format(sub_id / n_subjects,
                                        'size {:.1e} (t={:.1e})'.format(det_size, np.mean(ages)) if det_size is not None
                                        else 'age {:.0e} (size {:.1e})'.format(sim_time, np.mean(pt_sizes)),
                                        d, np.mean(bms_at_det_size), np.var(bms_at_det_size)))

                # path to dynamics output file is given, export to CSV file
                if dynamics_output_fp is not None:
                    if '<<>>' not in dynamics_output_fp:
                        raise RuntimeError(
                            'Output filename needs to contain <<>> to be replaced with subject ID: {}'.format(
                                dynamics_output_fp))

                    pt.export_history(dynamics_output_fp.replace('<<>>', '{:04d}'.format(sub_id)))

                # save temporary results to given file
                elif distr_output_fp is not None:
                    # append new results if file already exists and contains some results
                    if sub_id > 1 and os.path.isfile(distr_output_fp):
                        with open(distr_output_fp, 'a') as f:
                            f.write(f'{pt.bm}\n')
                    else:
                        pd.DataFrame(bms_at_det_size, columns=[Output.col_bm_amount]).to_csv(distr_output_fp,
                                                                                             index=False)

                break

    logger.info('Simulated {} PT subjects. Biomarkers at lesion {}: {:.5e} median, {:.5e} mean, {:.5e} variance'.format(
        n_subjects, 'size {:.1e} (mean t={:.0e})'.format(det_size, np.mean(ages)) if det_size is not None
        else 'age {:.1e} (mean size {:.1e})'.format(sim_time, np.mean(pt_sizes)),
        np.median(bms_at_det_size), np.mean(bms_at_det_size), np.var(bms_at_det_size)))

    return bms_at_det_size


def sim_n_steady_lesions(n_lesions, lesion_size, d, q_d, epsilon, sim_time, exact_th=1e4, starting_id=1,
                         output_fp=None):
    """
    Simulate the shedding of biomarkers in n lesions of constant size
    :param n_lesions: number of independently simulated realizations
    :param lesion_size: size of the lesion
    :param d: death rate of cells per day
    :param q_d: shedding rate per cell per day
    :param epsilon: ctDNA elimination rate per day
    :param sim_time: number of days to simulate the biomarker shedding of the lesion
    :param exact_th: max primary tumor size for exact simulation
    :param starting_id: starting identifier for simulated cases
    :param output_fp: export detailed dynamics to the given file path (needs to contain <<>> to be replaced with
                      subject ID)
    :return: array with biomarker level at given detection size of the primary tumor
    """

    bms_at_end_time = list()

    for sub_id in range(starting_id, n_lesions + 1):
        while True:
            pc = Precursor(d, q_d, epsilon, lesion_size, exact_th)
            pc_size = pc.sim_for_time(sim_time)
            if pc_size > 0:
                logger.debug('Precursor {} after {:.0f} days at size {:.1e} with {} biomarkers.'.format(
                    sub_id, pc.age, pc_size, pc.bm))
                bms_at_end_time.append(pc.bm)
                logger.debug('Biomarkers at lesion size {:.1e}: {:.3e} median, {:.3e} mean'.format(
                    lesion_size, np.median(bms_at_end_time), np.mean(bms_at_end_time)))

                if (sub_id / n_lesions * 100) % 5 == 0:
                    logger.info('{:.1%} progress: biomarkers at lesion size {:.1e}: {:.3e} median, {:.3e} mean'.format(
                        sub_id / n_lesions, lesion_size, np.median(bms_at_end_time), np.mean(bms_at_end_time)))

                # path to output file is given, export to CSV file
                if output_fp is not None:
                    if '<<>>' not in output_fp:
                        raise RuntimeError(
                            'Output filename needs to contain <<>> to be replaced with subject ID: {}'.format(
                                output_fp))

                    pc.export_history(output_fp.replace('<<>>', '{:03d}'.format(sub_id)))

                break

    logger.info('Simulated {} steady lesions. Biomarkers at lesion size {:.1e}: {:.3e} median, {:.3e} mean'.format(
        n_lesions, lesion_size, np.median(bms_at_end_time), np.mean(bms_at_end_time)))

    return bms_at_end_time


def export_data(fp, bms, sampled_bms, sampled_bms_vafs):
    """
    Export simulated data to given CSV file
    :param fp:
    :param bms:
    :param sampled_bms:
    :param sampled_bms_vafs:
    :return:
    """

    df_bms = pd.DataFrame(bms, columns=[Output.col_bm_amount])
    df_bms[Output.col_bm_sampled] = sampled_bms
    df_bms = df_bms.assign(biomarker_sampled_VAF=pd.Series(np.round(sampled_bms_vafs, 10)).values)
    df_bms.to_csv(fp, index=False)

    logger.info('Exported simulation results of {} cases to {}.'.format(len(df_bms), os.path.abspath(fp)))
