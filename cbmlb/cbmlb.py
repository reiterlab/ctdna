#!/usr/bin/python
"""Script to run CBMLB from the command line"""
import logging
import sys
import os
import argparse
import math
import numpy as np
import pandas as pd


import cbmlb.settings as settings
from cbmlb.utils import get_filename_template, Output, get_plasma_dna_concentrations, create_directory
from cbmlb.utils import get_tumor_dynamics_fps, add_parser_parameter_args, calculate_elimination_rate
from cbmlb.lesions_sim import sim_n_pt_subjects, export_data, sim_n_steady_lesions
from cbmlb.roc import create_roc_plot
from cbmlb.detection import Detection
from cbmlb.sampling import take_liquid_biopsies, perform_longitudinal_sampling

# create logger
logger = logging.getLogger('cbmlb')
logger.propagate = False
logger.setLevel(logging.INFO)
# logger = logging.getLogger('lifd.{}'.format(__name__))
# create file handler which logs even debug messages
fh = logging.FileHandler('cbmlb.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s:%(lineno)d %(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

__author__ = 'Johannes REITER'
__date__ = 'Feb 21, 2020'


def usage():
    """
    Give the user feedback on how to call the tool
    Terminates the tool afterwards
    """
    logger.warning('Usage: cbmlb [{dynamics,distribution,detection}] [-b <birth rate>] [-d <death rate>] '
                   '[-n <No. subjects>] [-M <detection size>]\n')
    logger.warning('Example: cbmlb distribution -b 0.14 -d 0.136 -n 10 -M 1e8 --q_d=1.5e-4')
    sys.exit(2)


def main(raw_args=None):
    """
    Main function of CBMLB (Circulating biomarker liquid biopsy module)
    :param raw_args: imitate calling function from command line
    :return:
    """

    parent_parser = argparse.ArgumentParser(
        description='CBMLB computes the expected detection size for a given biomarker and examination schedule.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False)

    add_parser_parameter_args(parent_parser)

    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='mode')

    # modes of cbmlb
    parser_dyn = subparsers.add_parser('dynamics', help='Simulate biomarker dynamics over time. ',
                                       parents=[parent_parser])
    parser_distr = subparsers.add_parser('distribution', parents=[parent_parser],
                                         help='Simulate biomarker distribution at a given tumor size or tumor age.')
    parser_roc = subparsers.add_parser('roc', parents=[parent_parser],
                                       help='Compute ROC for biomarker detection.')
    parser_det = subparsers.add_parser('detection', parents=[parent_parser],
                                       help='Simulate the detection of a biomarker for a growing tumor.')

    # sub-arguments in detection mode
    parser_det.add_argument('sampling_frequency', nargs='?', choices=('annual', 'biannual', 'biennial', 'triennial'),
                            default='annual', help='Frequency of liquid biopsy')

    parser_det.add_argument('-s', '--sampling_time', type=lambda s: map(float, s.split(',')), default=[],
                            dest='sampling_times',
                            help='Times at which a liquid biopsy is taken in days of tumor age')

    group = parser_det.add_mutually_exclusive_group()
    group.add_argument('--symptomatic_size', help='size when a tumor becomes diagnosed due to symptoms',
                       type=float, default=settings.SYMPTOMATIC_SIZE)
    group.add_argument('--imaging_det_size', help='size when a relapsing tumor becomes detected by imaging',
                       type=float, default=None)

    parser_det.add_argument('--n_covered_muts', help='number of expected mutations covered by the panel per cancer',
                            type=int, default=1)
    parser_det.add_argument('--min_reads', help='minimal variant reads for mutation detection',
                            type=int, default=1)

    parser_det.add_argument('--n_replications', help='number of times iterative sampling is randomized',
                            type=int, default=1)

    # sub-arguments in ROC mode
    parser_roc.add_argument('--pval_min_log', help='10-exponent of lowest p-value to consider', type=float,
                            default=-4.0)
    parser_roc.add_argument('--pval_max_log', help='10-exponent of highest p-value to consider', type=float,
                            default=0.0)
    parser_roc.add_argument('--pval_n_intervals', help='number of p-values to consider', type=int, default=65)

    args = parser.parse_args(raw_args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info('Run CBMLB in verbose mode.')

    if args.mode is None:
        logger.warning('No mode was selected to run cbmlb.')
        usage()

    if args.no_subjects < 0:
        raise RuntimeError(f'Number of cases to be simulated needs to be positive: {args.no_subjects}')
    if args.starting_id < 1:
        raise RuntimeError(f'Starting id needs to be >0: {args.starting_id}')

    if args.birth_rate is None:
        b = settings.BIRTH_RATE
    else:
        b = args.birth_rate[0]

    d = args.death_rate

    if b <= 0:
        raise RuntimeError(f'Birth rate has to be positive: {b}')

    if d < 0:
        raise RuntimeError(f'Birth rate has to be positive: {d}')

    if args.t12_mins <= 0:
        raise RuntimeError(f'Half-life time of biomarker has to be positive: {args.t12_mins}')

    # convert cfDNA half life time in minutes to the elimination rate per day
    epsilon = calculate_elimination_rate(args.t12_mins)

    # calculate cfDNA shedding rate
    q_d = args.q_d
    q_b = args.q_b
    shedding_rate = q_d * d + q_b * b + args.lambda1  # shedding per cell death, division and per unit of time

    # number of benign lesions
    n_precursors = args.n_bn_lesions

    if shedding_rate <= 0:
        raise RuntimeError(f'Effective biomarker shedding rate has to be positive: {shedding_rate}')

    # stopping condition of simulations
    if args.sim_time is not None:
        if args.sim_time <= 0:
            raise RuntimeError(f'Simulation end time of tumor has to be positive: {args.sim_time}')

        # simulate tumors to the given age
        sim_time = args.sim_time
        det_size = None

    else:
        if args.det_size[0] <= 0:
            raise RuntimeError(f'Tumor detection size {args.det_size[0]} needs to be positive.')

        # simulate to a given tumor size
        sim_time = None
        det_size = args.det_size[0]

    logger.info('{} is initializing for {} realizations with liquid biopsies of tube size: {} L.'.format(
        __name__, args.no_subjects - args.starting_id, args.tube_size))

    logger.info(f'Malignant cells divide with {b}, die with {d}, '
                + 'grow {:.3%} per day. Extinction probability of {:.3%}.'.format(b - d, d / b))
    logger.info('Biomarker half life of {} mins leads to an elimination rate epsilon of {:.4f} per day.'.format(
        args.t12_mins, epsilon))
    logger.info(f'Shedding probability per cell death is q_d={q_d:.3e}, per cell division is q_b={q_b:.3e}, '
                + f'lambda_1={args.lambda1:.3e}.')
    logger.info('Total shedding rate per cell per day is {:.3e}.'.format(shedding_rate))

    # setup output directories
    Output(args.output_dir)
    starting_id = args.starting_id

    # SIMULATE GROWING TUMORS
    if b - d > 0:

        # #########################################################################################################
        # simulate biomarker distribution at a certain tumor size
        # #########################################################################################################
        if args.mode == 'distribution':
            logger.info('Simulating biomarker distribution of malignant tumors until '
                        + ('size of {:.1e} cells.'.format(det_size) if det_size is not None
                           else 'age of {} days.'.format(sim_time)))
            if args.starting_id != 1:
                logger.warning(f'Starting ID argument {args.starting_id} is ignored in distribution mode.')
            # file naming according to parameter values
            fn_pattern = get_filename_template(b=b, d=d, t12_cfdna_mins=args.t12_mins, exact_th=args.exact_th,
                                               q_d=q_d, q_b=q_b, lambda_1=args.lambda1,
                                               det_size=det_size, t=sim_time, n_runs=args.no_subjects)

            # export simulated data to CSV file
            dist_fp = os.path.join(Output.bmdistr_dir, 'bmdistr{}.csv'.format(fn_pattern))
            if not args.rerun and os.path.isfile(dist_fp):
                df_bms = pd.read_csv(dist_fp)
                starting_id = len(df_bms) + 1
                logger.info('Found previously simulated biomarker distribution. '
                            'Start simulation with case {}.'.format(starting_id))

            else:
                starting_id = 1
                df_bms = None

            if starting_id <= args.no_subjects:
                bms_at_det_time = sim_n_pt_subjects(
                    args.no_subjects, b, d, q_d, epsilon, q_b=q_b, lambda_1=args.lambda1,
                    det_size=det_size, sim_time=sim_time, exact_th=args.exact_th,
                    distr_output_fp=dist_fp, starting_id=starting_id)

                if df_bms is not None:
                    # concatenate new results to existing ones
                    bms_at_det_time = np.concatenate(
                        (df_bms[Output.col_bm_amount].values, np.array(bms_at_det_time)), axis=0)

                total_bms_vafs, sampled_bms, sampled_bms_vafs = take_liquid_biopsies(
                    bms_at_det_time, tube_size=args.tube_size, biomarker_wt_freq_ml=args.biomarker_wt_freq_ml)

                export_data(dist_fp, bms_at_det_time, sampled_bms, sampled_bms_vafs)

            else:
                logger.info('Biomarker distribution for the same settings was previously simulated. '
                            'If you want to regenerate the output provide the option --rerun')

        # #########################################################################################################
        # simulate growth dynamics of individual tumors
        # #########################################################################################################
        elif args.mode == 'dynamics':
            logger.info('Simulating growth dynamics of malignant tumors until '
                        + ('size of {:.1e} cells.'.format(det_size) if det_size is not None
                           else 'age of {} days.'.format(sim_time)))
            if args.exact_th >= 0:
                logger.info('Deterministic growth is assumed once tumor reaches a size of {:.1e} cells.'.format(
                    args.exact_th))
            fn_pattern = get_filename_template(b=b, d=d, t12_cfdna_mins=args.t12_mins, exact_th=args.exact_th,
                                               q_d=q_d, q_b=q_b, lambda_1=args.lambda1,
                                               det_size=det_size, t=sim_time)
            dyn_dir = create_directory(os.path.join(Output.dyn_dir, f'dynamics{fn_pattern}'))
            dynamics_fp = os.path.join(dyn_dir, 'dynamics{}_<<>>.csv'.format(fn_pattern))

            # TODO find the individual tumors that are not yet simulated

            if (not args.rerun and
                    any(f.startswith('dynamics'+fn_pattern) and
                        f.endswith('.csv') for f in os.listdir(dyn_dir))):

                # found previously simulated tumor files
                highest_existing_id = max(
                    int(f[-8:-4]) if starting_id <= int(f[-8:-4]) <= args.no_subjects else -1
                    for f in os.listdir(dyn_dir)
                    if f.startswith('dynamics'+fn_pattern) and f.endswith('.csv'))
                starting_id = max(highest_existing_id+1, starting_id)

                if highest_existing_id != -1:
                    logger.info(f'Found previously simulated evolving tumors with highest id of {highest_existing_id}.')
            else:
                starting_id = starting_id

            if starting_id <= args.no_subjects:
                logger.info('Start simulation with case {}.'.format(starting_id))
                _ = sim_n_pt_subjects(args.no_subjects, b, d, q_d, epsilon, q_b=q_b, lambda_1=args.lambda1,
                                      det_size=det_size, sim_time=sim_time, exact_th=args.exact_th,
                                      starting_id=starting_id, dynamics_output_fp=dynamics_fp)
            else:
                logger.info('Cases {}-{} already exist.'.format(args.starting_id, starting_id - 1))

        # #########################################################################################################
        # simulate detection via longitudinal sampling
        # #########################################################################################################
        elif args.mode == 'detection':

            # check whether any tumors with the given settings have been simulated before
            tumor_fps = get_tumor_dynamics_fps(Output.dyn_dir, b, d, args.t12_mins, q_d, det_size,
                                               exact_th=args.exact_th)

            if len(tumor_fps) == 0:
                logger.warning('No previously simulated tumors found to perform longitudinal sampling at '
                               + f'{os.path.abspath(Output.dyn_dir)}.')
                logger.warning(f'Run {__name__} in dynamics mode to simulate growing and shedding tumors.')
                sys.exit(0)

            # only take the first X cases
            if args.no_subjects > 0:
                tumor_fps = tumor_fps[:args.no_subjects]

            # were custom sampling times, e.g., -s 0,365,730
            sampling_times = [t for t in args.sampling_times]
            if len(sampling_times):
                sampling_freq = None
                sampling_times = np.array(sampling_times)
                logger.info(f'Found {len(tumor_fps)} previously simulated tumors to perform sampling at times: '
                            + ', '.join(str(t) for t in sampling_times))

            else:
                sampling_times = None
                logger.info(f'Found {len(tumor_fps)} previously simulated tumors to perform {args.sampling_frequency} '
                            'sampling.')
                if args.sampling_frequency == 'monthly':
                    sampling_freq = 30
                    # exp_time_to_size = cond_growth_etime_supexp(b, d, det_size)
                    # sampling_times = list(range(0, exp_time_to_size+1000, sampling_freq))
                elif args.sampling_frequency == 'quarterly':
                    sampling_freq = 91
                elif args.sampling_frequency == 'biannual':
                    sampling_freq = 183
                elif args.sampling_frequency == 'annual':
                    sampling_freq = 365
                elif args.sampling_frequency == 'biennial':
                    sampling_freq = 2*365
                elif args.sampling_frequency == 'triennial':
                    sampling_freq = 3*365
                else:
                    # should never happen because the default value is annual
                    sampling_freq = 365

            # Create random samples of biomarker wildtype molecules (plasma DNA concentrations) per mL in humans
            # number of distinct wildtype biomarker levels over time per subject
            longitudinally_varying_bmwt_levels = 20
            plasma_dna_concs = get_plasma_dna_concentrations(
                len(tumor_fps) * args.n_replications * longitudinally_varying_bmwt_levels,
                gamma_params=settings.FIT_GAMMA_PARAMS)
            plasma_dna_concs = plasma_dna_concs.reshape((len(tumor_fps), args.n_replications,
                                                         longitudinally_varying_bmwt_levels))

            # calculate wildtype whole genome equivalents per plasma ml
            wGEs_per_ml = plasma_dna_concs / settings.DIPLOID_GE_WEIGHT_ng

            # TODO implement custom detection
            det_th = Detection(min_supp_reads=args.min_reads)

            det_size_col = 'lb_det_size'
            det_time_col = 'lb_det_time'
            notes_col = 'Notes'
            # RELAPSE
            if args.imaging_det_size is not None:
                lb_det_sizes, lb_det_times, _, lead_times = perform_longitudinal_sampling(
                    tumor_fps, sampling_freq, sampling_times, wGEs_per_ml, det_th,
                    args.n_covered_muts, args.tube_size, args.panel_size, args.seq_err,
                    imaging_det_size=args.imaging_det_size, n_replications=args.n_replications)

                lead_time_col = 'lead_time_imaging'
                df_det = pd.DataFrame({'lb_det_size': lb_det_sizes, 'lb_det_time': lb_det_times,
                                      lead_time_col: lead_times})

                caption = f'# '
                df_det.append({det_size_col: caption, det_time_col: '', lead_time_col: ''},
                              ignore_index=True)

            # SCREENING
            else:
                sympt_size = args.symptomatic_size
                lb_det_sizes, lb_det_times, symptomatic_det_times, lead_times = perform_longitudinal_sampling(
                    tumor_fps, sampling_freq, sampling_times, wGEs_per_ml, det_th,
                    args.n_covered_muts, args.tube_size, args.panel_size, args.seq_err,
                    symptomatic_size=sympt_size, n_replications=args.n_replications)

                if len(lb_det_sizes) == 0:
                    logger.warning('No valid samplings for the provided sampling times. Try different parameters. ')
                    sys.exit(0)

                symp_time_col = 'symptomatic_det_time'
                diag_size_col = 'diagnosis_size'
                lead_time_col = 'lead_time_diagnosis'
                df_det = pd.DataFrame(
                    {det_size_col: lb_det_sizes, 'lb_det_time': lb_det_times,
                     symp_time_col: symptomatic_det_times, lead_time_col: lead_times})

                # determine tumor size at diagnosis (minimum of detection size or when tumor became symptomatic)
                diag_sizes = list()
                for idx, row in df_det.iterrows():

                    if row[det_time_col] is not None and row[symp_time_col] is not None:
                        if row[det_time_col] < row[symp_time_col]:
                            diag_sizes.append(row[det_size_col])
                        else:
                            diag_sizes.append(sympt_size)

                    elif row[det_time_col] is not None:
                        diag_sizes.append(row[det_size_col])

                    elif row[symp_time_col] is not None:
                        diag_sizes.append(sympt_size)

                    else:
                        diag_sizes.append(pd.NA)
                df_det[diag_size_col] = diag_sizes

                info_str = (
                        f'Longitudinal sampling for {len(df_det[det_size_col]) / args.n_replications} cases with '
                        + f'P<={det_th.pval_th:.1e}, min muts {det_th.min_det_muts}, '
                        + f'min variant reads {det_th.min_supp_reads}, min VAF {det_th.min_det_vaf}.')
                logger.info(info_str)
                res_det_str = (
                    f'Detection sizes for {df_det[det_size_col].count()} detected cases:'
                    + f'mean {np.nanmean(df_det[det_size_col]):.3e}, median {np.nanmedian(df_det[det_size_col]):.3e},'
                    + f' 25th perc {np.nanpercentile(df_det[det_size_col], 25):.3e}, '
                    + f'75th perc {np.nanpercentile(df_det[det_size_col], 75):.3e}, '
                    + f'largest {np.nanmax(df_det[det_size_col]):.3e}')
                logger.info(res_det_str)

                res_diag_str = f'Diagnosis sizes with a symptomatic size of {sympt_size:.2e}: '
                res_diag_str += (
                        f'mean {np.nanmean(df_det[diag_size_col]):.3e}, '
                        + f'median {np.nanmedian(df_det[diag_size_col]):.3e}, '
                        + f'25th perc {np.nanpercentile(df_det[diag_size_col], 25):.3e}, '
                        + f'75th perc {np.nanpercentile(df_det[diag_size_col], 75):.3e}')

                logger.info(res_diag_str)

                df_det[det_size_col] = df_det.apply(lambda row: '{:.3e}'.format(row[det_size_col]), axis=1)

                empty_row = {det_size_col: '', det_time_col: '', lead_time_col: '', symp_time_col: '', notes_col: ''}

                def add_df_comment(df, col, comment, row):
                    """
                    Substitute entry in col of row with comment and append it to the dataframe
                    :param df:
                    :param col:
                    :param comment:
                    :param row:
                    :return: appended dataframe
                    """
                    row[col] = f'# {comment}'
                    return df.append(row, ignore_index=True)

                df_det = add_df_comment(df_det, det_size_col, info_str, empty_row)
                df_det = add_df_comment(df_det, det_size_col, res_det_str, empty_row)
                df_det = add_df_comment(df_det, det_size_col, res_diag_str, empty_row)
                if sampling_times is not None:
                    sampling_str = 'Fixed sampling times: ' + ', '.join(str(t) for t in sampling_times)
                    df_det = add_df_comment(df_det, det_size_col, sampling_str, empty_row)

                # df_det = df_det.append(, ignore_index=True)
                # df_det = df_det.append({det_size_col: f'# {res_diag_str}', det_time_col: '', lead_time_col: '',
                #                         symp_time_col: '', notes_col: ''}, ignore_index=True)

                fn_pattern_det = get_filename_template(
                    b=b, d=d, t12_cfdna_mins=args.t12_mins, exact_th=args.exact_th, q_d=q_d, q_b=q_b,
                    lambda_1=args.lambda1, smp_frq=sampling_freq, seq_err=args.seq_err, seq_eff=args.seq_eff,
                    n_replications=args.n_replications, n_runs=len(tumor_fps))
                if sampling_freq is None and sampling_times is not None:
                    fn_pattern_det += f'_smps={len(sampling_times)}'

                # export longitudinal sampling analysis results
                # # df_det = df_det.astype({det_size_col: float})
                det_fp = os.path.join(
                    Output.detection_dir, f'detection{fn_pattern_det}.csv')
                df_det.to_csv(det_fp, index=False, float_format='%.5e')

        # #########################################################################################################
        # compute ROC
        # #########################################################################################################
        elif args.mode == 'roc':

            create_roc_plot(
                b, d, args.b_bn, args.d_bn, q_d, args.t12_mins,
                n_precursors, args.bn_lesion_size, args.det_size, args.no_subjects,
                muts_per_cancer=args.n_cancer_muts, muts_per_precursor=args.n_bn_lesion_muts,
                tube_size=args.tube_size, panel_size=args.panel_size, seq_err=args.seq_err, seq_eff=args.seq_eff,
                min_det_muts=args.min_det_muts,
                pval_min_log=args.pval_min_log, pval_max_log=args.pval_max_log, pval_n_intervals=args.pval_n_intervals)

    # SIMULATE TUMORS WITH CONSTANT SIZE
    else:
        if b != args.death_rate:
            raise RuntimeError('Birth and death rates need to be equal for benign tumor simulation.')

        if args.mode == 'dynamics':
            logger.info('Simulating shedding dynamics of benign tumors at a size of {:.1e} cells.'.format(
                args.det_size[0]))
            fn_pattern = get_filename_template(b=b, d=b, t12_cfdna_mins=args.t12_mins, exact_th=args.exact_th,
                                               q_d=q_d, q_b=q_b, lambda_1=args.lambda1,
                                               det_size=det_size, t=args.sim_time)
            dynamics_fp = os.path.join(Output.dyn_dir, 'dynamics{}_<<>>.csv'.format(fn_pattern))
            _ = sim_n_steady_lesions(args.no_subjects, det_size, b, q_d, epsilon, args.sim_time,
                                     exact_th=args.exact_th, starting_id=args.starting_id, output_fp=dynamics_fp)

        else:
            raise NotImplementedError('Only dynamics mode is supported for constant-sized benign lesions.')


if __name__ == '__main__':

    main()
