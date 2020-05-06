#!/usr/bin/python
"""Various helper methods for the cbmlb-package"""
import logging
import os
import pathlib
import math
import numpy as np
import pandas as pd

import cbmlb.settings as settings

__date__ = 'February 24, 2020'
__author__ = 'Johannes REITER'

# get logger
logger = logging.getLogger(__name__)


class Output:

    root_dir = None
    dyn_dir = None
    bmdistr_dir = None
    detection_dir = None
    roc_dir = None

    col_bm_amount = 'biomarker_total'
    col_bm_sampled = 'biomarker_sampled'

    def __init__(self, output_directory=settings.OUTPUT_DIR_NAME):
        """
        Set up output directories
        :param output_directory:
        """

        logger.info(f'Setting up output directories at {os.path.abspath(output_directory)}.')

        # create directory for output files
        Output.root_dir = create_directory(os.path.abspath(output_directory))
        # create directory for dynamics files
        Output.dyn_dir = create_directory(os.path.join(Output.root_dir, 'dynamics'))
        # create directory for biomarker distribution files
        Output.bmdistr_dir = create_directory(os.path.join(Output.root_dir, 'distribution'))
        # create directory for receiver operating characteristic results
        Output.roc_dir = create_directory(os.path.join(Output.root_dir, 'roc'))
        # create directory for detection through longitudinal sampling files
        Output.detection_dir = create_directory(os.path.join(Output.root_dir, 'detection'))


def create_directory(folder):
    """
    Create the given directory and missing parent directories
    :param folder:
    :return:
    """
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    if os.path.exists(os.path.dirname(folder)):
        return folder
    else:
        raise RuntimeError('Could not create output folder {} '.format(folder))


def add_parser_parameter_args(parser):
    """
    Add arguments to parser
    :param parser: argument parser argparse
    """

    parser.add_argument('-v', '--verbose', action='store_true', help='run in verbose/debugging mode.')

    parser.add_argument('-b', '--birth_rate', help='birth rate of cancer cells', nargs='*',
                        type=float, default=None)
    parser.add_argument('-d', '--death_rate', help='death rate of cancer cells',
                        type=float, default=settings.DEATH_RATE)

    parser.add_argument('-i', '--starting_id', help='Start simulation from given subject id',
                        type=int, default=1)

    parser.add_argument('-n', '--no_subjects', help='number of instances per parameter setting',
                        type=int, default=0)

    parser.add_argument('--q_d', help='shedding probability per cell death event', type=float,
                        default=settings.Q_D_LUNG)
    parser.add_argument('--q_b', help='shedding probability per cell division event', type=float, default=0.0)
    parser.add_argument('--lambda1', help='shedding rate per cell per day', type=float, default=0.0)
    parser.add_argument('--t12_mins', help='cfDNA half life in minutes', type=float, default=settings.T12_MINS)

    # sequencing settings
    parser.add_argument('--panel_size', help='sequencing panel size', type=int, default=settings.PANEL_SIZE)
    parser.add_argument('--seq_err', help='sequencing error rate', type=float,
                        default=settings.SEQUENCING_ERROR_RATE)
    parser.add_argument('--seq_eff', help='sequencing efficiency', type=float,
                        default=settings.SEQUENCING_EFFICIENCY)

    # relevant for virtual detection
    parser.add_argument('--b_bn', help='birth rate of benign cells', type=float, default=None)
    parser.add_argument('--d_bn', help='death rate of benign cells', type=float, default=None)
    parser.add_argument('--bn_lesion_size', help='benign lesion size', type=float, default=None)
    parser.add_argument('--n_bn_lesions', help='number of benign lesions', type=int, default=1)

    parser.add_argument('--n_cancer_muts', help='number of mutations in cancer', type=int, default=None)
    parser.add_argument('--n_bn_lesion_muts', help='number of mutations in benign lesions', type=int, default=None)

    parser.add_argument('-M', '--det_size',
                        help='primary tumor detection size where biomarker level is evaluated',
                        nargs='+', type=float, default=settings.PT_DETSIZE)
    parser.add_argument('-T', '--sim_time',
                        help='simulation running time for tumors [days]',
                        type=float, default=None)

    parser.add_argument('--min_det_muts', help='minimal number of called mutations such that the detection is positive',
                        type=int, default=1)
    parser.add_argument('--tube_size', help='liquid biopsy sampling tube size in liters [default 0.015]',
                        type=float, default=settings.TUBE_SIZE)

    parser.add_argument('--biomarker_wt_freq_ml', help='mean number of wildtype biomarkers per plasma ml',
                        type=int, default=settings.NO_WT_BIOMARKERS_ML)

    parser.add_argument('--exact_th', help='approximate growth of tumor after it reaches a threshold',
                        type=float, default=settings.EXACT_THRESHOLD)

    parser.add_argument('-o', '--output_dir', help='output directory',
                        type=str, default=os.path.join(settings.OUTPUT_DIR_NAME))

    # parser.add_argument('-s', '--source_dir', help='source directory',
    #                     type=str, default=settings.SRC_DIR)

    parser.add_argument('--rerun', action='store_true', help='rerun analysis.')


def get_filename_template(b=None, d=None, t12_cfdna_mins=None,
                          q_d=None, q_b=None, lambda_1=None, det_size=None, nu=None, n_runs=None, t=None,
                          exact_th=None, min_det_muts=None, pval_th=None, fpr=None, min_supp_reads=None,
                          min_det_vaf=None, mpc=None, mpp=None, pcs=None, ps=None, tube_size=None,
                          seq_err=None, seq_eff=None, smp_frq=None, n_replications=None, suffix=None):
    """
    Produces template for naming of output files
    :param b: birth rate per day
    :param d: death rate per day
    :param t12_cfdna_mins: half-life time of biomarker
    :param q_d: biomarker shedding probability per cell death event
    :param q_b: biomarker shedding probability per cell division (proliferation)
    :param lambda_1: biomarker shedding probability per unit of time (necrosis)
    :param det_size: simulated tumor end size
    :param nu: biomarker shedding rate
    :param n_runs: number of realizations/subjects
    :param pcs: number of precursor lesions
    :param t: simulated time in days
    :param exact_th: growth of lesion is approximated after it reached the threshold
    :param mpc: number of covered mutations per cancer
    :param mpp: number of covered mutations per precursor lesion
    :param ps: number of mutations covered by sequencing panel
    :param tube_size: liters of sampled blood (default 0.015 l, 15 mL)
    :param seq_err: sequencing error rate per base pair
    :param seq_eff: sequencing efficiency corresponds to fraction of molecules in sample that get sequenced
    :param pval_th: maximum p-value for calling a mutation
    :param fpr: false positive rate
    :param smp_frq: liquid biopsy longitudinal sampling frequency in days
    :param min_det_muts: number of detected mutations required for a positive tests
    :param min_supp_reads: minimal number of mutant reads that need to support a mutation to be called
    :param min_det_vaf: minimal VAF (variant allele frequency) of a mutation to be called
    :param n_replications: number of performed replications for screening result
    :param suffix:
    :return:
    """

    pattern = (
            ('_b={}'.format(b) if b is not None else '') +
            ('_d={}'.format(d) if d is not None else '') +
            ('_t12={}'.format(t12_cfdna_mins) if t12_cfdna_mins is not None else '') +
            ('_qd={:.1e}'.format(q_d) if q_d is not None and q_d > 0 else '') +
            ('_qb={:.1e}'.format(q_b) if q_b is not None and q_b > 0 else '') +
            ('_l1={:.1e}'.format(lambda_1) if lambda_1 is not None and lambda_1 > 0 else '') +
            ('_M={:.1e}'.format(det_size) if det_size is not None else '') +
            ('_nu={}'.format(nu) if nu is not None else '') +
            ('_pcs={}'.format(pcs) if pcs is not None else '') +
            ('_t={:.1e}'.format(t) if t is not None else '') +
            ('_eth={:.0e}'.format(exact_th) if exact_th is not None and exact_th != settings.EXACT_THRESHOLD else '') +
            ('_n={:.0e}'.format(n_runs) if n_runs is not None else '') +
            ('_mpc={}'.format(mpc) if mpc is not None else '') +
            ('_mpp={}'.format(mpp) if mpp is not None else '') +
            ('_ps={}'.format(ps) if ps is not None else '') +
            ('_ts={}'.format(tube_size) if tube_size is not None else '') +
            ('_sqer={:.1e}'.format(seq_err) if seq_err is not None else '') +
            ('_sqef={:.1f}'.format(seq_eff) if seq_eff is not None and seq_eff != 1 else '') +
            ('_P<={:.1e}'.format(pval_th) if pval_th is not None else '') +
            ('_fpr={:.1e}'.format(fpr) if fpr is not None else '') +
            ('_smpfrq={}'.format(smp_frq) if smp_frq is not None else '') +
            ('_muts={}'.format(min_det_muts) if min_det_muts is not None else '') +
            ('_reads={}'.format(min_supp_reads) if min_supp_reads is not None and min_supp_reads > 1 else '') +
            ('_vaf>={:.1e}'.format(min_det_vaf) if min_det_vaf is not None and min_det_vaf > 0 else '') +
            ('_r={}'.format(n_replications) if n_replications is not None and n_replications > 1 else '') +
            (suffix if suffix is not None else ''))

    # replace points with commas because latex cannot handle points in file names (interprets it as file type)
    # pattern = pattern.replace('.', '_')

    return pattern


def get_tumor_dynamics_fps(parent_dyn_dir, b, d, t12, q_d, det_size, exact_th=settings.EXACT_THRESHOLD):
    """
    Find previously simulated tumor dynamics files
    :param parent_dyn_dir: parent folder where simulated files are stored
    :param b: birth rate per day
    :param d: death rate per day
    :param t12: half-life time of biomarker
    :param q_d: biomarker shedding probability per cell death event
    :param det_size: simulated tumor end size
    :param exact_th: growth of lesion is approximated after it reached the threshold
    :return: list of file paths
    """
    tumor_fps = list()
    fn_pattern = get_filename_template(b=b, d=d, t12_cfdna_mins=t12,
                                       q_d=q_d, det_size=det_size, exact_th=exact_th)
    dyn_dir = os.path.join(parent_dyn_dir, f'dynamics{fn_pattern}')
    for fn in sorted(os.listdir(dyn_dir)):
        if fn_pattern in fn and fn.endswith('.csv'):
            tumor_fps.append(os.path.join(dyn_dir, fn))

    logger.info(f'Found {len(tumor_fps)} tumor dynamics files with matching pattern of {fn_pattern}.')
    return tumor_fps


def extract_parameter_values(filename):
    fn = os.path.basename(filename)
    left = 0
    params = dict()
    while fn[left:].find('=') != -1:
        eq = fn[left:].find('=')
        name = fn[left + fn[left:left + eq].rfind('_') + 1:left + eq]
        if fn[left + eq:].find('_') != -1:
            value = fn[left + eq + 1:left + eq + fn[left + eq:].find('_')]
        else:
            value = fn[left + eq + 1:left + eq + fn[left + eq:].find('.')]
        left += eq + 1
        params[name] = value

    return params


def get_plasma_dna_concentrations(size, gamma_params=None, beta_params=None):
    """
    Create random samples of plasma DNA concentrations per mL in humans
    :param size: number of needed sampled plasma DNA concentrations
    :param gamma_params: sample from fitted gamma distribution with the given parameters
    :param beta_params: sample from fitted beta distribution with the given parameters
    """

    if gamma_params is not None:
        plasma_dna_concs = np.random.gamma(gamma_params['shape'], gamma_params['scale'], size=size)
        return plasma_dna_concs

    elif beta_params is not None:
        plasma_dna_concs = np.random.beta(
            beta_params['alpha'], beta_params['beta'], size=size) * beta_params['scale'] + beta_params['loc']
        # ensure that all values are positive
        if all(plasma_dna_concs > 0):
            return plasma_dna_concs
        else:
            pos_vals = plasma_dna_concs[plasma_dna_concs > 0]
            new_vals = get_plasma_dna_concentrations(size - len(pos_vals), beta_params=beta_params)
            return np.concatenate((pos_vals, new_vals))
    else:
        raise NotImplementedError(
            'Only Gamma and Beta distributions are implemented to mimic the plasma DNA distributions.')


def read_bms_data(fn, data_dir):
    bms_fp = os.path.join(data_dir, fn)
    df_bms = pd.read_csv(bms_fp)

    return df_bms


def calculate_elimination_rate(lambda_cfdna_mins=settings.T12_MINS):
    """
    Calculate the cell-free DNA elimination rate (epsilon)
    :param lambda_cfdna_mins: half life time of cfDNA and ctDNA in minutes
    :return: cell-free DNA elimination rate
    """

    lambda_days = lambda_cfdna_mins / (60 * 24)
    # cfDNA elimination rate per day (epsilon)
    epsilon = math.log(2) / lambda_days
    logger.info(f'cfDNA half life of {lambda_cfdna_mins} mins leads to an elimination rate epsilon of {epsilon:.3f} '
                + 'per day.')
    return epsilon


def calculate_shedding_rate(ge_per_ml_per_cm3, epsilon, n_cells_per_cm3=1e9,
                            blood_amount=settings.BLOOD_AMOUNT, plasma_fraction=settings.PLASMA_FRACTION):
    """
    Calculate ctDNA shedding rate per cell per day
    :param ge_per_ml_per_cm3: genome equivalents per plasma mL per cubic centimeter (or mL) of tumor volume
    :param epsilon: cfDNA/ctDNA elimination rate
    :param n_cells_per_cm3: assumed number of cells per cubic centimeter (or mL) of tumor volume
    :param blood_amount: amount of blood in the blood stream
    :param plasma_fraction: fraction of plasma in the blood
    :return: shedding rate
    """

    plasma_ml = blood_amount * plasma_fraction * 1000
    q = ge_per_ml_per_cm3 * plasma_ml * epsilon / n_cells_per_cm3
    logger.info(f'Shedding rate per unit of time is {q:.3e} if there are {ge_per_ml_per_cm3:.3e} biomarker units '
                + 'per plasma ml for a tumor of 1 cm^3.')
    return q


def calculate_shedding_probability(ge_per_ml_per_cm3, d, epsilon, n_cells_per_cm3=1e9,
                                   blood_amount=settings.BLOOD_AMOUNT, plasma_fraction=settings.PLASMA_FRACTION):
    """
    Calculate ctDNA shedding probability q_d per cell death
    :param ge_per_ml_per_cm3: genome equivalents per plasma mL per cubic centimeter (or mL) of tumor volume
    :param d: cancer cell death rate
    :param epsilon: cfDNA/ctDNA elimination rate
    :param n_cells_per_cm3: assumed number of cells per cubic centimeter (or mL) of tumor volume
    :param blood_amount: amount of blood in the blood stream
    :param plasma_fraction: fraction of plasma in the blood
    :return: shedding probability q_d per cell ceath
    """
    qd = calculate_shedding_rate(ge_per_ml_per_cm3, epsilon, n_cells_per_cm3, blood_amount, plasma_fraction) / d
    logger.info(f'Shedding probability per cell death event is {qd:.3e} if there are {ge_per_ml_per_cm3:.3e} biomarker '
                + f'units per plasma ml for a tumor of 1 cm^3 with d={d:.3f} per day.')
    return qd


def sphere_volume(d):
    return 4.0/3 * (d/2)**3 * math.pi


def diameter_cells(d_cm):
    return diameter_volume(d_cm) * 1e9


def diameter_volume(d_cm):
    return 4.0/3 * (d_cm/2)**3 * math.pi


def longest_diameter_volume(d_cm):
    return 4.0/3 * (d_cm/2)**3 * math.pi


def volume_diameter(v_cells):
    """
    Takes volume in number of cells and returns diameter in cm
    """
    return (6 * v_cells/1e9 / math.pi) ** (1.0/3)
