#!/usr/bin/python
"""Calculate receiver operating characteristics for circulating biomarker"""

import logging
import os
import numpy as np

import ctdna.settings as settings
from ctdna.utils import get_filename_template, Output
from ctdna.utils import get_plasma_dna_concentrations, calculate_elimination_rate
from ctdna.sampling import simulate_virtual_detection
from ctdna.detection import Detection
from ctdna.bp_formulas import get_random_bms_at_size

__date__ = 'October 21, 2018'
__author__ = 'Johannes REITER'


# get logger
logger = logging.getLogger(__name__)

cancerseek_size = 4500
cappseq_size = 300000


try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import rgb2hex

    colors = sns.color_palette()
    orange = (1.0, 0.4980392156862745, 0.054901960784313725)
    blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    blue_dark = (0.06, 0.25, 0.6)
    blue_light = (0.25, 0.55, 0.9)
    purple = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    red = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    magenta = (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
    turquoise = (0.01, 0.78, 0.81)
    colors = [colors[0], colors[1], red, turquoise, magenta, colors[2], colors[4]] + colors[6:] + [blue_dark,
                                                                                                   blue_light]
except ImportError:
    logger.warning('Plotting libraries are not available!')
    # plotting is not available on cluster
    plt = None
    colors = None
    sns = None


def calculate_roc(n_samples, n_precursors, b, d, b_bn, d_bn, q_d, t12, bn_lesion_size,
                  muts_per_cancer, muts_per_precursor, det_sizes,
                  tube_size, panel_size, seq_err, seq_eff,
                  min_det_muts=1, min_supp_reads=0, min_det_vaf=0.0,
                  pval_min_log=-4, pval_max_log=0, pval_n_intervals=65):
    """
    Calculate Receiver Operating Characteristic (ROC) for the given biomarker level of primary tumors
    and multiple benign lesions
    :param n_samples: number of samples over which ROC is calculated
    :param n_precursors:
    :param b: birth rate per day
    :param d: death rate per day
    :param b_bn: birth rate of cells in benign lesion per day
    :param d_bn: death rate of cells in benign lesion per day
    :param q_d: biomarker shedding probability per cell death event
    :param t12: half-life time of biomarker in minutes
    :param bn_lesion_size: number of cells in benign lesion
    :param muts_per_cancer:
    :param muts_per_precursor:
    :param det_sizes:
    :param tube_size: liquid biopsy sampling tube size in blood liters
    :param panel_size: number of sequenced basepairs on panel
    :param seq_err: sequencing error rate per basepair
    :param seq_eff: sequencing efficiency corresponds to fraction of molecules in sample that get sequenced
    :param min_det_muts: minimal number of called mutations such that the detection is positive
    :param min_supp_reads: minimal number of supporting reads to call a mutation
    :param min_det_vaf: minimal variant allele frequency to call a mutation
    :param pval_min_log: define p-val min exponent (10-base) to explore in ROC
    :param pval_max_log: define p-val max exponent (10-base) to explore in ROC
    :param pval_n_intervals: # 81, 41 161 33, 65 or 5 for testing
    :return:
    """

    logger.info(f'Create ROC curve for panel size {panel_size} requiring a p-value for {n_precursors} benign lesions.')
    if n_precursors > 0:
        logger.info(f'Benign lesions are of size {bn_lesion_size:.1e}, b_bn={b_bn}, d_bn={d_bn}.')

    pvals = [0.0] + list(np.logspace(pval_min_log, pval_max_log, pval_n_intervals))
    pvals = sorted(pvals[:-1]   # + [0.99, 0.999, 0.9999]
                   + [1])

    # create random samples of plasma DNA concentrations per mL in humans
    plasma_dna_concs = get_plasma_dna_concentrations(n_samples, gamma_params=settings.FIT_GAMMA_PARAMS)
    # calculate whole genome equivalents per plasma ml
    hges_per_ml = plasma_dna_concs / (settings.DIPLOID_GE_WEIGHT_ng / 2)

    # ############# Compute false positives from benign lesions #############
    fprs = compute_false_positives(n_samples, n_precursors, b_bn, d_bn, q_d, t12, bn_lesion_size, muts_per_precursor,
                                   hges_per_ml, tube_size, panel_size, seq_err, seq_eff,
                                   min_det_muts, min_det_vaf, min_supp_reads, pvals, pval_min_log, pval_max_log)

    logger.debug('Obtained false positive rates: ' + ', '.join('{:.4f}'.format(fpr) for fpr in fprs))

    # ####### Compute true positives from cancers #########
    tprss = list()
    # convert cfDNA half life time in minutes to the elimination rate per day
    epsilon = calculate_elimination_rate(t12)
    for det_size in det_sizes:
        tps_fn_pattern = get_filename_template(
            b=b, d=d, t12_cfdna_mins=t12, q_d=q_d, det_size=det_size, mpc=muts_per_cancer, min_det_muts=min_det_muts,
            min_supp_reads=min_supp_reads, min_det_vaf=min_det_vaf, ps=panel_size, seq_err=seq_err, seq_eff=seq_eff,
            n_runs=n_samples)

        tps_fp = os.path.join(Output.roc_dir, 'roc_pval{:.0f}{:.0f}{:.0f}_tps{}.csv'.format(
            pval_min_log, pval_max_log, len(pvals), tps_fn_pattern))
        if os.path.exists(tps_fp) and os.path.isfile(tps_fp):
            tprs = np.genfromtxt(tps_fp, delimiter=',')
            logger.info('Read file with previously computed true positive rates: {}'.format(tps_fp))
        else:
            logger.info(f'Missing precomputed true positive rates file for det size {det_size:.1e}: {tps_fp}')

            tprs = [0.0]  # true positive rate
            bm_tumors = get_random_bms_at_size(det_size, d * q_d, b-d, epsilon, size=n_samples)

            for pval in pvals:
                tpr = simulate_sensitivity(
                    bm_tumors, muts_per_cancer, min_det_muts, pval_th=pval, min_supp_reads=min_supp_reads,
                    min_det_vaf=min_det_vaf, wt_hge_per_ml=hges_per_ml, tube_size=tube_size,
                    panel_size=panel_size, seq_err=seq_err, seq_eff=seq_eff)
                tprs.append(tpr)

            # save computed false positive rates to file for faster figure creation
            np.savetxt(tps_fp, tprs, delimiter=',')
            logger.info('Saved computed true positive rates to: {}'.format(tps_fp))

        tprss.append(tprs)
        logger.debug('Obtained true positive rates for detection size {:.1e}: '.format(det_size)
                     + ', '.join('{:.4f}'.format(tpr) for tpr in tprs))
        logger.info(
            'Sensitivity and specificity for p-value thresholds at size {:.1e}: '.format(det_size)
            + '; '.join('{:.1e}: {:.2e}, {:.2e}'.format(pval, tpr, 1 - fpr) for pval, tpr, fpr in zip(pvals, tprs[1:],
                                                                                                      fprs[1:])))

    fprss = [fprs for _ in range(len(tprss))]

    return fprss, tprss


def compute_false_positives(n_samples, n_precursors, b_bn, d_bn, q_d, t12, bn_lesion_size, muts_per_precursor,
                            wt_hge_per_ml, tube_size, panel_size, seq_err, seq_eff,
                            min_det_muts, min_det_vaf, min_supp_reads, pvals, pval_min_log, pval_max_log):
    """
    Calculate the false-positive rate for the given number of samples
    :param n_samples: number of samples where the false positives are considered
    :param n_precursors:
    :param b_bn: birth rate of cells in benign lesion per day
    :param d_bn: death rate of cells in benign lesion per day
    :param q_d: biomarker shedding probability per cell death event
    :param t12: half-life time of biomarker
    :param bn_lesion_size:
    :param muts_per_precursor:
    :param wt_hge_per_ml: array of wildtype biomarker amount (hGE) per plasma mL for each sample
    :param tube_size:
    :param panel_size:
    :param seq_err:
    :param seq_eff: sequencing efficiency corresponds to fraction of molecules in sample that get sequenced
    :param min_det_muts:
    :param min_det_vaf:
    :param min_supp_reads:
    :param pvals:
    :param pval_min_log:
    :param pval_max_log:
    :return:
    """

    # convert cfDNA half life time in minutes to the elimination rate per day
    epsilon = calculate_elimination_rate(t12)
    fps_fn_pattern = get_filename_template(
        b=b_bn, d=d_bn, t12_cfdna_mins=t12, q_d=q_d, det_size=bn_lesion_size, mpp=muts_per_precursor,
        min_det_muts=min_det_muts, min_supp_reads=min_supp_reads, min_det_vaf=min_det_vaf,
        ps=panel_size, seq_err=seq_err, seq_eff=seq_eff, n_runs=n_samples)
    fps_fp = os.path.join(Output.roc_dir, 'roc_pval{:.0f}{:.0f}{:.0f}_fps_pcs={}{}.csv'.format(
        pval_min_log, pval_max_log, len(pvals), n_precursors, fps_fn_pattern))

    # check whether false positives were previously computed
    if os.path.exists(fps_fp) and os.path.isfile(fps_fp):
        fprs = np.genfromtxt(fps_fp, delimiter=',')
        logger.info('Read file with previously computed false positive rates: {}'.format(fps_fp))

    else:
        logger.info('Missing precomputed false positive rates file: {}'.format(fps_fp))

        fprs = [0.0]  # false positive rate
        for pval in pvals:
            fpr = simulate_specificity(
                n_samples, n_precursors, muts_per_precursor, min_det_muts=min_det_muts,
                pval_th=pval, min_supp_reads=min_supp_reads,
                min_det_vaf=min_det_vaf, wt_hge_per_ml=wt_hge_per_ml, tube_size=tube_size,
                panel_size=panel_size, seq_err=seq_err, seq_eff=seq_eff,
                bn_lesion_size=bn_lesion_size, b_bn=b_bn, d_bn=d_bn, q_d_bn=q_d, elimination_rate=epsilon)
            fprs.append(fpr)

        # save computed false positive rates to file for faster figure creation
        np.savetxt(fps_fp, fprs, delimiter=',')
        logger.info('Saved computed false positive rates to: {}'.format(fps_fp))

    return fprs


def simulate_specificity(n_samples, n_precursors, n_muts, min_det_muts, pval_th, min_supp_reads, min_det_vaf,
                         wt_hge_per_ml, tube_size, panel_size, seq_err, seq_eff,
                         bn_lesion_size=None, b_bn=None, d_bn=None, q_d_bn=None, elimination_rate=None):
    """
    Calculate the specificity for the given number of samples
    :param n_samples: number of samples where the false positives are considered
    :param n_precursors: number of benign lesions shedding ctDNA
    :param n_muts:
    :param min_det_muts:
    :param pval_th:
    :param min_supp_reads:
    :param min_det_vaf:
    :param wt_hge_per_ml: array of wildtype biomarker (hGE) amount per plasma mL for each sample
    :param tube_size: liquid biopsy sampling tube size in blood liters
    :param panel_size: number of sequenced basepairs on panel
    :param seq_err: sequencing error rate per basepair
    :param seq_eff: sequencing efficiency corresponds to fraction of molecules in sample that get sequenced
    :param bn_lesion_size: number of cells in benign lesion
    :param b_bn: birth rate of cells in benign lesion per day
    :param d_bn: death rate of cells in benign lesion per day
    :param q_d_bn: biomarker shedding probability per cell death event of a benign cell
    :param elimination_rate: biomarker elimination rate per day
    :return:
    """

    det_th = Detection(pval_th=pval_th, min_det_muts=None, min_supp_reads=min_supp_reads,
                       min_det_vaf=min_det_vaf)
    # calculate detected number of mutations just from sequencing noise
    det_muts = simulate_virtual_detection(
            np.zeros(n_samples), det_th, muts_per_tumor=0, tube_size=tube_size,
            panel_size=panel_size, seq_err=seq_err, seq_eff=seq_eff,
            wt_hge_per_ml=wt_hge_per_ml)

    # run through all precursor lesions
    for i in range(n_precursors):
        bm_amount = get_random_bms_at_size(bn_lesion_size, d_bn * q_d_bn, b_bn - d_bn, elimination_rate, size=n_samples)

        # consider sequencing errors of absent mutations only in first run otherwise these would be distinct biopsies
        det_muts += simulate_virtual_detection(
            bm_amount, det_th, muts_per_tumor=n_muts, tube_size=tube_size,
            # consider just mutations per precursor as sequencing errors of wildtype biomarkers were considered above
            panel_size=n_muts, seq_err=seq_err, seq_eff=seq_eff, wt_hge_per_ml=wt_hge_per_ml)

    screening_test = det_muts >= min_det_muts
    fpr = sum(screening_test) / float(n_samples)

    return fpr


def simulate_sensitivity(bm_amounts, n_muts, min_det_muts, pval_th, min_supp_reads, min_det_vaf,
                         wt_hge_per_ml, tube_size, panel_size, seq_err, seq_eff):
    """
    Calculate the true-positive rate for the given tumors
    :param bm_amounts: biomarker amount circulating in blood stream that was shed from the tumor
    :param n_muts: number of mutations present in tumor that are also covered by the sequencing panel
    :param pval_th: only mutations reaching a p-value smaller than this threshold are called
    :param min_det_muts: number of detected mutations required for a positive tests
    :param min_supp_reads: minimal number of mutant reads that need to support a mutation to be called
    :param min_det_vaf: minimal VAF (variant allele frequency) of a mutation to be called
    :param wt_hge_per_ml: array of wildtype biomarker (hGE) amount per plasma mL for each sample
    :param tube_size: liquid biopsy sampling tube size in blood liters
    :param panel_size: number of sequenced basepairs on panel
    :param seq_err: sequencing error rate per basepair
    :param seq_eff: sequencing efficiency corresponds to fraction of molecules in sample that get sequenced
    :return:
    """

    # set threshold for classifying a sample with present cancer
    det_th = Detection(pval_th=pval_th, min_det_muts=min_det_muts, min_supp_reads=min_supp_reads,
                       min_det_vaf=min_det_vaf)

    tumor_test = simulate_virtual_detection(
        bm_amounts, det_th, muts_per_tumor=n_muts, tube_size=tube_size,
        panel_size=panel_size, seq_err=seq_err, seq_eff=seq_eff, wt_hge_per_ml=wt_hge_per_ml)

    # Calculate true positive rate (TPR), sensitivity, recall
    tpr = sum(tumor_test) / float(len(bm_amounts))
    #     logger.debug('M: {:.1e}, min muts {}, min reads {}, min VAF {:.1e}: TPR (sensitivity) {:.3%}'.format(
    #         det_size, min_det_muts, min_supp_reads, min_det_vaf, tpr))

    return tpr


def plot_roc_curve(fprss, tprss, labels, xlim=(0, 1), ylim=(0, 1), legend=True,
                   linestyle='--', linewidth=1.3, alpha=0.75,
                   figsize=(3.6, 2.7), colors=colors, title=None, output_fp=None):

    fig, ax = plt.subplots(figsize=figsize)

    for i, (fprs, tprs) in enumerate(zip(fprss, tprss)):
        ax.plot(fprs, tprs, linestyle=linestyle, lw=linewidth, alpha=alpha,
                fillstyle='none', color=colors[i], label=labels[i])

    ax.set_xlabel('False positive rate (1 - specificity)', fontsize=12)
    ax.set_ylabel('True positive rate (sensitivity)', fontsize=12)
    ext = 0.02
    ax.set_xlim(xlim[0] - ext, xlim[1] + ext)
    ax.set_ylim(ylim[0] - ext, ylim[1] + ext)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_bounds(ylim[0], ylim[1])
    ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

    outward = 5
    for line in ['left', 'bottom']:
        ax.spines[line].set_position(('outward', outward))

    plt.tick_params(axis='both', which='major', labelsize=11)
    #     plt.grid(b=True, which='both', color='0.8', linestyle=':', alpha=1)

    if legend:
        leg = plt.legend(loc='lower right', facecolor='white', frameon=True, framealpha=1.0, fancybox=False,
                         prop={'size': 10})
        leg.get_frame().set_facecolor('white')
        # set the alpha value of the legend: it will be translucent
        leg.get_frame().set_alpha(1)
        leg.get_frame().set_linewidth(0.0)

    if title is not None:
        ax.set_title(title)

    if output_fp is not None:
        plt.savefig(output_fp, dpi=150, bbox_inches='tight', transparent=True)

    logger.info('Plotted ROC curve: {}'.format(output_fp))


def create_roc_plot(b, d, b_bn, d_bn, q_d, t12, n_precursors, bn_lesion_size, det_sizes, n_samples,
                    muts_per_cancer, muts_per_precursor, tube_size, panel_size, seq_err, seq_eff, min_det_muts,
                    pval_min_log=-4, pval_max_log=0, pval_n_intervals=65, plt_dir=''):
    """
    Create ROC plot
    :param b:
    :param d:
    :param b_bn:
    :param d_bn:
    :param q_d:
    :param t12:
    :param n_precursors:
    :param bn_lesion_size:
    :param det_sizes:
    :param n_samples:
    :param muts_per_cancer:
    :param muts_per_precursor:
    :param tube_size:
    :param panel_size:
    :param seq_err:
    :param seq_eff: sequencing efficiency corresponds to fraction of molecules in sample that get sequenced
    :param min_det_muts:
    :param pval_min_log: 10-exponent of lowest p-value to consider
    :param pval_max_log: 10-exponent of highest p-value to consider
    :param pval_n_intervals: number of p-values to consider
    :param plt_dir: directory where plots are saved
    :return:
    """

    min_supp_reads = 0
    min_det_vaf = 0.0

    fprss, tprss = calculate_roc(
        n_samples, n_precursors, b, d, b_bn, d_bn, q_d, t12, bn_lesion_size,
        muts_per_cancer, muts_per_precursor, det_sizes, tube_size, panel_size, seq_err, seq_eff,
        min_det_muts=min_det_muts, min_supp_reads=min_supp_reads, min_det_vaf=min_det_vaf,
        pval_min_log=pval_min_log, pval_max_log=pval_max_log, pval_n_intervals=pval_n_intervals)

    labels = list()
    for det_size, fprs, tprs in zip(det_sizes, fprss, tprss):
        auc = np.trapz(tprs, fprs)
        labels.append('AUC: {:.4%} (M: {:.1e})'.format(auc, det_size))
    logger.info(', '.join(str(l) for l in labels))

    if plt is not None:
        if n_precursors > 0:
            pc_fn_pattern = get_filename_template(b=b_bn, det_size=bn_lesion_size)
        else:
            pc_fn_pattern = ''
        tumor_fn_pattern = get_filename_template(b=b, d=d, t12_cfdna_mins=t12, q_d=q_d)

        seq_fn_pattern = get_filename_template(
            mpc=muts_per_cancer, mpp=None if n_precursors == 0 else muts_per_precursor,
            ps=panel_size, seq_err=seq_err, seq_eff=seq_eff, pcs=n_precursors)

        pnl_roc_fp = os.path.join(
            plt_dir,
            'pnl_ROC_pval_minmuts={}{}{}{}_res={}{}{}.pdf'.format(
                min_det_muts, ('' if min_supp_reads == 0 else '_minreads={}'.format(min_supp_reads)),
                '' if min_det_vaf == 0.0 else '_minvaf={}'.format(min_det_vaf),
                seq_fn_pattern, len(fprss[0]),  # resolution (number of thresholds explored)
                pc_fn_pattern, tumor_fn_pattern))

        title = None

        plot_roc_curve(fprss, tprss, labels, colors=colors, legend=False, linestyle='-', linewidth=2.0,
                       title=title, output_fp=pnl_roc_fp)
