#!/usr/bin/python
"""Script to run ctdna from the command line"""
import logging
import sys
import os
import argparse
import numpy as np
import pandas as pd

import ctdna.settings as settings
from ctdna.utils import get_filename_template, Output, get_plasma_dna_concentrations, create_directory
from ctdna.utils import cells_diameter, diameter_cells
from ctdna.utils import get_tumor_dynamics_fps, add_parser_parameter_args, calculate_elimination_rate, stats_string
from ctdna.lesions_sim import sim_n_pt_subjects, export_data, sim_n_steady_lesions
from ctdna.roc import create_roc_plot
from ctdna.detection import Detection, compute_pval_th
from ctdna.sampling import take_liquid_biopsies, perform_longitudinal_sampling, get_lead_time_info

# create logger
logger = logging.getLogger(__name__)

__author__ = 'Johannes REITER'
__date__ = 'Feb 21, 2020'


def usage():
    """
    Give the user feedback on how to call the tool
    Terminates the tool afterwards
    """
    logger.warning('Usage: ctdna [{dynamics,distribution,detection}] [-b <birth rate>] [-d <death rate>] '
                   '[-n <No. subjects>] [-M <detection size>]\n')
    logger.warning('Example: ctdna distribution -b 0.14 -d 0.136 -n 10 -M 1e8')
    sys.exit(2)


def main(raw_args=None):
    """
    Main function of package ctdna (circulating tumor DNA)
    :param raw_args: imitate calling function from command line
    :return:
    """

    parent_parser = argparse.ArgumentParser(
        description='Package ctdna computes the expected tumor detection size for a biomarker and sampling frequency.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False)

    add_parser_parameter_args(parent_parser)

    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='mode')

    # modes of ctdna
    parser_dyn = subparsers.add_parser('dynamics', help='Simulate biomarker dynamics over time. ',
                                       parents=[parent_parser])

    parser_distr = subparsers.add_parser('distribution', parents=[parent_parser],
                                         help='Simulate biomarker distribution at a given tumor size or tumor age.')
    parser_roc = subparsers.add_parser('roc', parents=[parent_parser],
                                       help='Compute ROC for biomarker detection.')
    parser_det = subparsers.add_parser('detection', parents=[parent_parser],
                                       help='Simulate the detection of a biomarker for a growing tumor.')

    # sub-arguments in dynamics mode
    parser_dyn.add_argument('--treatment_start_size', help='Start therapy when the tumor reaches the given size',
                            type=float, default=None)
    parser_dyn.add_argument('--birth_rate_treat', type=float, default=None,
                            help='birth rate of cells during treatment; if negative, tumor is removed by surgery')
    parser_dyn.add_argument('--death_rate_treat', type=float, default=None,
                            help='death rate of cells during treatment; if negative, tumor is removed by surgery')

    # sub-arguments in detection mode
    parser_det.add_argument('sampling_frequency', nargs='?',
                            choices=['daily', 'weekly', 'semimonthly', 'monthly', 'quarterly', 'semiannually',
                                     'annually', 'biennially', 'triennially'],
                            default=None, help='Frequency of liquid biopsy')

    parser_det.add_argument('-s', '--sampling_time', type=lambda s: map(float, s.split(',')), default=[],
                            dest='sampling_times',
                            help='Times at which a liquid biopsy is taken in days of tumor age')

    parser_det.add_argument('--no_ctDNA', dest='no_ctDNA', default=False, action='store_true')

    group = parser_det.add_mutually_exclusive_group()
    group.add_argument('--diagnosis_size', type=float, default=settings.DIAGNOSIS_SIZE,
                       help='size when a tumor becomes diagnosed due to symptoms (ignored by sampling)')
    group.add_argument('--symptomatic_size', type=float, default=None,
                       help='size when a tumor becomes diagnosed due to symptoms (stops sampling)')
    group.add_argument('--imaging_det_size', help='size when a relapsing tumor becomes detected by imaging',
                       type=float, default=None)

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
        logger.warning('No mode was selected to run ctdna.')
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
        __name__, args.no_subjects - args.starting_id + 1, args.tube_size))

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

                # Create random samples of biomarker wildtype molecules (plasma DNA concentrations) per mL in humans
                # according to the specified gamma distribution
                if args.biomarker_wt_freq_ml is None:

                    # sample random plasma cfDNA concentrations to calculate wildtype cfDNA per plasma mL
                    plasma_dna_concs = get_plasma_dna_concentrations(
                        len(bms_at_det_time), gamma_params=settings.FIT_GAMMA_PARAMS)

                    # calculate wildtype whole genome equivalents per plasma ml
                    wt_hge_per_ml = plasma_dna_concs / settings.DIPLOID_GE_WEIGHT_ng

                else:
                    # number of wildtype genome equivalents is fixed at the given number
                    wt_hge_per_ml = np.ones(len(bms_at_det_time)) * args.biomarker_wt_freq_ml

                # take liquid biopsies
                total_bms_vafs, sampled_bms, sampled_bms_vafs = take_liquid_biopsies(
                    bms_at_det_time, wt_hge_per_ml, tube_size=args.tube_size)

                export_data(dist_fp, bms_at_det_time, sampled_bms, sampled_bms_vafs)

            else:
                logger.info('Biomarker distribution for the same settings was previously simulated. '
                            'If you want to regenerate the output provide the option --rerun')

        # #########################################################################################################
        # simulate growth dynamics of individual tumors
        # #########################################################################################################
        elif args.mode == 'dynamics':
            logger.info('Simulating growth dynamics of malignant tumors until '
                        + (f'size of {det_size:.1e} cells.' if det_size is not None else f'age of {sim_time} days.'))

            if args.exact_th >= 0:
                logger.info('Deterministic growth is assumed once tumor reaches a size of {:.1e} cells.'.format(
                    args.exact_th))

            tx_start = args.treatment_start_size
            b_treat = args.birth_rate_treat
            d_treat = args.death_rate_treat
            if tx_start is not None and tx_start > 1:
                logger.info(f'Treatment starts when tumor reaches a size of {tx_start:.1e} cells.')
                if b_treat is None:
                    b_treat = b
                if d_treat is None:
                    d_treat = d
                logger.info(f'During treatment, tumor grows with r\'={b_treat-d_treat:.1e} and rates of '
                            f'b\'={b_treat} and d\'={d_treat}')

            fn_pattern = get_filename_template(b=b, d=d, t12_cfdna_mins=args.t12_mins, exact_th=args.exact_th,
                                               q_d=q_d, q_b=q_b, lambda_1=args.lambda1,
                                               tx_start=tx_start, b_treat=b_treat, d_treat=d_treat,
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
                                      tx_start=tx_start, b_treat=b_treat, d_treat=d_treat,
                                      starting_id=starting_id, dynamics_output_fp=dynamics_fp)
            else:
                logger.info('Cases {}-{} already exist.'.format(args.starting_id, starting_id - 1))

        # #########################################################################################################
        # simulate detection via longitudinal sampling
        # #########################################################################################################
        elif args.mode == 'detection':

            diag_size = args.diagnosis_size
            sympt_size = args.symptomatic_size

            # check whether any tumors with the given settings have been simulated before
            tumor_fps = get_tumor_dynamics_fps(Output.dyn_dir, b, d, args.t12_mins, q_d, det_size,
                                               exact_th=args.exact_th)

            if len(tumor_fps) == 0:
                logger.warning('No previously simulated tumors found to perform longitudinal sampling at '
                               + f'{os.path.abspath(Output.dyn_dir)}.')
                logger.warning(f'Run {__name__} in dynamics mode to simulate growing and shedding tumors.')
                sys.exit(0)

            logger.info(f'Found {len(tumor_fps)} previously simulated tumors.')

            # only take the first X cases
            if args.no_subjects > 0:
                tumor_fps = tumor_fps[:args.no_subjects]
                logger.info(f'Selected only a subset of {len(tumor_fps)} previously simulated tumors.')

            # were custom sampling times, e.g., -s 0,365,730
            sampling_times = [t for t in args.sampling_times]
            if len(sampling_times):
                sampling_freq = None
                sampling_times = np.array(sampling_times)
                logger.info('Perform sampling at times: {} '.format(', '.join(str(t) for t in sampling_times)))

            else:
                if args.sampling_frequency is None:
                    logger.error('Either a sampling frequency (e.g., monthly, annually, etc) or sampling times '
                                 + 'need to be given. ')
                    usage()

                sampling_times = None
                logger.info(f'Perform {args.sampling_frequency} sampling.')

                if args.sampling_frequency == 'daily':
                    sampling_freq = 1
                elif args.sampling_frequency == 'weekly':
                    sampling_freq = 7
                elif args.sampling_frequency == 'semimonthly':
                    sampling_freq = 15
                elif args.sampling_frequency == 'monthly':
                    sampling_freq = 30
                    # exp_time_to_size = cond_growth_etime_supexp(b, d, det_size)
                    # sampling_times = list(range(0, exp_time_to_size+1000, sampling_freq))
                elif args.sampling_frequency == 'quarterly':
                    sampling_freq = 91
                elif args.sampling_frequency == 'semiannually':
                    sampling_freq = 183
                elif args.sampling_frequency == 'annually':
                    sampling_freq = 365
                elif args.sampling_frequency == 'biennially':
                    sampling_freq = 2*365
                elif args.sampling_frequency == 'triennially':
                    sampling_freq = 3*365
                else:
                    # should never happen because the default value is annual
                    sampling_freq = 365

            # limit the number of longitudinally different biomarker wildtype levels to the given number for efficiency
            longitudinally_varying_bmwt_levels = 20
            # Create random samples of biomarker wildtype molecules (plasma DNA concentrations) per mL in humans
            # number of distinct wildtype biomarker levels over time per subject
            if args.biomarker_wt_freq_ml is None:
                # sample random plasma cfDNA concentrations to calculate wildtype cfDNA per plasma mL
                dna_conc_gamma_params = settings.FIT_GAMMA_PARAMS
                plasma_dna_concs = get_plasma_dna_concentrations(
                    len(tumor_fps) * args.n_replications * longitudinally_varying_bmwt_levels,
                    gamma_params=dna_conc_gamma_params)

                plasma_dna_concs = plasma_dna_concs.reshape((len(tumor_fps), args.n_replications,
                                                             longitudinally_varying_bmwt_levels))

                # calculate wildtype whole genome equivalents per plasma ml
                wt_hge_per_ml = plasma_dna_concs / settings.DIPLOID_GE_WEIGHT_ng

            else:
                # number of wildtype genome equivalents is fixed at the given number
                dna_conc_gamma_params = None
                wt_hge_per_ml = args.biomarker_wt_freq_ml * np.ones(
                    (len(tumor_fps), args.n_replications, longitudinally_varying_bmwt_levels))

            if args.pval_th is None and args.annual_fpr is not None:
                # calculate a p-value threshold for classifying an observed number of mutant fragments across a panel
                # as a positive test at a desired false positive rate (1-specificity)

                pval_th = compute_pval_th(
                    args.annual_fpr, args.panel_size, args.seq_err, args.seq_eff, smp_frq=sampling_freq,
                    dna_conc_gamma_params=dna_conc_gamma_params, wt_hge_per_ml=args.biomarker_wt_freq_ml,
                    n_min_det_muts=args.min_muts, min_supp_reads=args.min_reads, min_det_vaf=args.min_vaf,
                    n_bn_lesions=args.n_bn_lesions, bn_lesion_size=args.bn_lesion_size, d_bn=args.d_bn, q_d_bn=args.q_d,
                    epsilon=epsilon, muts_per_bn_lesion=args.n_bn_lesion_muts)

            elif args.annual_fpr is None and args.pval_th is not None:
                pval_th = args.pval_th

            else:
                raise RuntimeError('Either an annual false positive rate threshold (e.g. "--annual_fpr 0.01") or '
                                   'a p-value threshold (e.g., "--pval_th 1e-5") is required in detection mode '
                                   '(but not both can be given).')

            # set threshold for classifying a sample as positive and diagnosing cancer
            det_th = Detection(pval_th=pval_th, min_det_muts=args.min_muts, min_supp_reads=args.min_reads,
                               min_det_vaf=args.min_vaf)

            # RELAPSE
            if args.imaging_det_size is not None:
                lb_det_sizes, lb_det_times, _, lead_times = perform_longitudinal_sampling(
                    tumor_fps, sampling_freq, sampling_times, wt_hge_per_ml, det_th,
                    args.n_muts, args.tube_size, args.panel_size, args.seq_err, args.seq_eff,
                    imaging_det_size=args.imaging_det_size, n_replications=args.n_replications, no_ctDNA=args.no_ctDNA)

                df_det = pd.DataFrame({Output.col_det_size: lb_det_sizes, Output.col_det_time: lb_det_times,
                                       Output.col_lead_t_img: lead_times})

                caption = f'# '
                df_det.append({Output.col_det_size: caption, Output.col_det_time: '', Output.col_lead_t_img: ''},
                              ignore_index=True)

                empty_row = {Output.col_det_size: '', Output.col_det_time: '', Output.col_lead_t_img: '',
                             Output.col_notes: ''}

            # SCREENING
            else:
                lb_det_sizes, lb_det_times, symptomatic_det_times, lead_times = perform_longitudinal_sampling(
                    tumor_fps, sampling_freq, sampling_times, wt_hge_per_ml, det_th,
                    args.n_muts, args.tube_size, args.panel_size, args.seq_err, args.seq_eff,
                    diagnosis_size=diag_size, symptomatic_size=sympt_size, n_replications=args.n_replications,
                    no_ctDNA=args.no_ctDNA)

                if len(lb_det_sizes) == 0:
                    logger.warning('No valid samplings for the provided sampling times. Try different parameters. ')
                    sys.exit(0)

                df_det = pd.DataFrame(
                    {Output.col_det_size: lb_det_sizes, Output.col_det_time: lb_det_times,
                     Output.col_symp_time: symptomatic_det_times, Output.col_lead_t_diag: lead_times})

                # determine tumor size at diagnosis (minimum of detection size or when tumor became symptomatic)
                diag_sizes = list()
                for idx, row in df_det.iterrows():
                    if row[Output.col_det_time] is not None and row[Output.col_symp_time] is not None:
                        if row[Output.col_det_time] < row[Output.col_symp_time]:
                            diag_sizes.append(row[Output.col_det_size])
                        else:
                            diag_sizes.append(diag_size)
                    elif row[Output.col_det_time] is not None:
                        diag_sizes.append(row[Output.col_det_size])
                    elif row[Output.col_symp_time] is not None:
                        diag_sizes.append(diag_size)
                    else:
                        diag_sizes.append(pd.NA)

                df_det[Output.col_diag_size] = diag_sizes

                empty_row = {Output.col_det_size: '', Output.col_det_time: '', Output.col_lead_t_diag: '',
                             Output.col_symp_time: '', Output.col_notes: ''}

            # ######## Write results to file ###########
            info_str = (
                f'Longitudinal sampling for {len(df_det[Output.col_det_size]) / args.n_replications} subjects with '
                + f'{args.n_replications} replications and threshold for {det_th}.')
            logger.info(info_str)
            str_res_all = (f'Detection sizes for {df_det[Output.col_det_size].count()} cases [cells]: '
                           + f'{stats_string(df_det[Output.col_det_size])}')
            logger.info(str_res_all)
            str_res_dm = (f'Detection diameters for {df_det[Output.col_det_size].count()} cases [cm]: '
                          + f'{stats_string(cells_diameter(df_det[Output.col_det_size]))}')
            logger.info(str_res_dm)

            if (sympt_size is not None or diag_size is not None) and Output.col_symp_time in df_det.columns.values:

                sympt_vol = sympt_size if diag_size is None else diag_size

                # sizes of screen-detected cancers
                det_sizes = df_det[
                    df_det[Output.col_det_time] < df_det[Output.col_symp_time]][Output.col_det_size].to_numpy()

                str_res_det = (f'Detection sizes of {det_sizes.size} ({det_sizes.size/df_det.shape[0]:.3%}) '
                               + f'detected cases [cells]: {stats_string(det_sizes)}')
                logger.info(str_res_det)
                str_res_det_dm = (f'Detection diameters of {det_sizes.size} ({det_sizes.size/df_det.shape[0]:.3%}) '
                                  + f'detected cases [cm]: {stats_string(cells_diameter(det_sizes))}')
                logger.info(str_res_det_dm)

                # diagnosis volumes in number of cells
                res_diag_str = f'Diagnosis sizes if sampling stops at symptomatic size of {sympt_vol:.2e} [cells]: '
                res_diag_str += stats_string(df_det[Output.col_diag_size])
                logger.info(res_diag_str)

                # diagnosis sizes in tumor diameters
                res_diag_dm_str = f'Diagnosis diameters if sampling stops at symptomatic size of {sympt_vol:.2e} [cm]: '
                res_diag_dm_str += stats_string(cells_diameter(df_det[Output.col_diag_size]))
                logger.info(res_diag_dm_str)

                # at which stages are the cancers detected by screening
                res_staging_str = 'Tumors detected with diameters: '
                stage_diameter_ths = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5,7), (7, 100)]
                cells_ths = [diameter_cells(d_ths[1]) for d_ths in stage_diameter_ths]
                size_fractions = [sum(det_sizes <= cell_th)/len(det_sizes) for cell_th in cells_ths]
                stage_fractions = list()
                for i in range(len(size_fractions)):
                    if i == 0:
                        stage_fractions.append(size_fractions[i])
                    else:
                        stage_fractions.append(size_fractions[i] - size_fractions[i-1])

                res_staging_str += ','.join(f'{d_th[0]}< {d_th[1]}cm={stage_frac:.3%}'
                                            for d_th, stage_frac in zip(stage_diameter_ths, stage_fractions))
                logger.info(res_staging_str)

            else:
                sympt_vol = None
                str_res_det = None
                str_res_det_dm = None
                res_diag_str = None
                res_diag_dm_str = None
                res_staging_str = None

            df_det[Output.col_det_size] = df_det.apply(lambda row: '{:.3e}'.format(row[Output.col_det_size]), axis=1)

            def add_df_comment(df, comment, line):
                """
                Substitute entry in col of row with comment and append it to the dataframe
                :param df:
                :param comment:
                :param line:
                :return: appended dataframe
                """
                line[Output.col_det_size] = '#'
                line[Output.col_det_time] = comment
                return df.append(line, ignore_index=True)

            df_det = add_df_comment(df_det, info_str, empty_row)
            df_det = add_df_comment(df_det, str_res_all, empty_row)
            df_det = add_df_comment(df_det, str_res_dm, empty_row)
            if sympt_vol is not None:
                df_det = add_df_comment(df_det, str_res_det, empty_row)
                df_det = add_df_comment(df_det, str_res_det_dm, empty_row)

            if args.imaging_det_size or diag_size or sympt_size:
                lead_times_info = get_lead_time_info(args.imaging_det_size, diag_size, sympt_size, lead_times)
                logger.info(lead_times_info)
                df_det = add_df_comment(df_det, lead_times_info, empty_row)

                # also add information for one-sided (only positive lead times)
                if diag_size is not None:
                    pos_lead_times = np.array(lead_times)
                    pos_lead_times = pos_lead_times[pos_lead_times > 0]
                    pos_lead_times_info = get_lead_time_info(None, None, diag_size, pos_lead_times)
                    logger.info(pos_lead_times_info)
                    df_det = add_df_comment(df_det, pos_lead_times_info, empty_row)

            if res_diag_str is not None:
                df_det = add_df_comment(df_det, res_diag_str, empty_row)
            if res_diag_dm_str is not None:
                df_det = add_df_comment(df_det, res_diag_dm_str, empty_row)
            if res_staging_str is not None:
                df_det = add_df_comment(df_det, res_staging_str, empty_row)
            if sampling_times is not None:
                sampling_str = 'Fixed sampling times: ' + ', '.join(str(t) for t in sampling_times)
                df_det = add_df_comment(df_det, sampling_str, empty_row)

            fn_pattern_det = get_filename_template(
                b=b, d=d, t12_cfdna_mins=args.t12_mins, exact_th=args.exact_th, q_d=q_d, q_b=q_b,
                lambda_1=args.lambda1, mpc=args.n_muts, tube_size=args.tube_size, ps=args.panel_size,
                smp_frq=sampling_freq, seq_err=args.seq_err, seq_eff=args.seq_eff,
                pval_th=det_th.pval_th, min_det_muts=det_th.min_det_muts, min_supp_reads=det_th.min_supp_reads,
                min_det_vaf=det_th.min_det_vaf,
                pcs=args.n_bn_lesions, det_size=args.bn_lesion_size, mpp=args.n_bn_lesion_muts, d_bn=args.d_bn,
                n_replications=args.n_replications, n_runs=len(tumor_fps))

            if sampling_freq is None and sampling_times is not None:
                fn_pattern_det += f'_smps={len(sampling_times)}'

            # export longitudinal sampling analysis results
            # # df_det = df_det.astype({Output.col_det_size: float})
            det_fp = os.path.join(
                Output.detection_dir, f'detection{fn_pattern_det}.csv')
            df_det.to_csv(det_fp, index=False, float_format='%.5e')
            logger.info(f'Sampling results written to file: {os.path.abspath(det_fp)}')

        # #########################################################################################################
        # compute ROC
        # #########################################################################################################
        elif args.mode == 'roc':

            create_roc_plot(
                b, d, args.b_bn, args.d_bn, q_d, args.t12_mins,
                n_precursors, args.bn_lesion_size, args.det_size, args.no_subjects,
                muts_per_cancer=args.n_muts, muts_per_precursor=args.n_bn_lesion_muts,
                tube_size=args.tube_size, panel_size=args.panel_size, seq_err=args.seq_err, seq_eff=args.seq_eff,
                min_det_muts=args.min_muts,
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
