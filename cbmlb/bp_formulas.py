""" Implementations of exact results for branching process """
import numpy as np

__date__ = 'October 21, 2018'
__author__ = 'Johannes REITER'


EULERS_CONSTANT = 0.577215664901532


def get_growth_to_biomarker(b0, d0, shedding_rate, n_bms=None):
    """
    Get exponentially distributed growth/decline to the occurrence of 1 or n biomarker;
    biomarker is shed per unit of time
    :param b0: birth rate of the wildtype
    :param d0: death rate of the wildtype
    :param shedding_rate: biomarker shedding rate (sum of shedding per cell death, division, and per time unit)
    :param n_bms: number of drawn random samples
    :return: array of n drawn random numbers
    """
    return np.random.exponential(abs(b0 - d0) / shedding_rate, size=n_bms)


def get_time_to_biomarker(sc_sizes, b0, d0, shedding_rate):
    """
    Get exponentially distributed time to the occurrence of 1 or n mutations and the growth of the ancestral subclones
    in that time;
    biomarker is only shed per cell death event
    :param sc_sizes: size of subclone that generates the new biomarker, if n_bms is not None needs
                     to be an array of that size
    :param b0: birth rate of the wildtype
    :param d0: death rate of the wildtype
    :param shedding_rate: biomarker shedding rate (sum of shedding per cell death, division, and per time unit)
    :return: array of n_bms times, array of n_bms growths
    """
    growth = get_growth_to_biomarker(b0, d0, shedding_rate)
    return np.log((sc_sizes + growth) / sc_sizes) / (b0 - d0), growth


def get_time_to_event_constant_pop(rate, n, n_events=None):
    """
    Get exponentially distributed waiting time to the next n events in a population of fixed finite size
    :param rate: event rate [per time unit]
    :param n: population size
    :param n_events: number of next events simulated
    :return: array of random waiting times to next n events
    """
    return np.random.exponential(1.0 / (rate * n), size=n_events)


def get_growth_fraction_rate(r, time):
    """
    Calculate the growth fraction
    :param r: growth rate of the clone
    :param time: time in days
    :return: growth fraction
    """
    return np.exp(r * time)


def get_bms_at_size(tumor_size, shedding_rate, growth_rate, elimination_rate):
    """
    Get the mean number of biomarkers for a tumor at a given size
    :param tumor_size: size of the tumor
    :param shedding_rate: biomarker shedding rate per unit of time [per day]
    :param growth_rate: tumor growth rate per day
    :param elimination_rate: biomarker elimination rate per day
    :return:
    """
    return tumor_size * shedding_rate / (growth_rate + elimination_rate)


def get_random_bms_at_size(tumor_size, shedding_rate, growth_rate, elimination_rate, size=None):
    """
    Get a random number of circulating biomarkers for a tumor at a given size
    :param tumor_size: size of the tumor
    :param shedding_rate: biomarker shedding rate per unit of time [per day]
    :param growth_rate: tumor growth rate per day
    :param elimination_rate: biomarker elimination rate per day
    :param size: if size is None (default), a single value is returned otherwise an array of
    :return:
    """

    bm_mean = get_bms_at_size(tumor_size, shedding_rate, growth_rate, elimination_rate)
    return np.random.poisson(bm_mean, size=size)


def get_growth_fraction_rate(r, time):
    """
    Calculate the growth fraction
    :param r: growth rate of the clone
    :param time: time in days
    :return: growth fraction
    """
    return np.exp(r * time)


def get_growth_fraction(b, d, time):
    """
    Calculate the growth fraction
    :param b: birth rate of the clone
    :param d: death rate of the clone
    :param time: time in days
    :return: growth fraction
    """
    r = b - d
    return np.exp(r * time)


def get_growth_time(r, det_size, cur_size=1):
    """
    Calculate deterministic growth time to detection size starting from the given time
    :param r: growth rate
    :param det_size: detection size
    :param cur_size: current size of population
    :return: time to grow until it reached detection size
    """
    return np.log(det_size / cur_size) / r


def cond_growth_etime(b, d, det_size):
    """
    Calculate deterministic growth time to detection size starting with a single cell conditioned on survival
    (see Rick Durrett, Branching Process Models of Cancer, 2015)
    :param b: birth rate
    :param d: death rate
    :param det_size: detection size
    :return: time to grow until it reached detection size
    """
    return np.log((b - d) / b * det_size) / (b - d)


def cond_growth_etime_supexp(b, d, det_size):
    """
    Calculate deterministic growth time to detection size starting with a single cell conditioned on survival
    (see Rick Durrett, Branching Process Models of Cancer, 2015)
    Accounts for super exponential growth of subpopulations early on
    :param b: birth rate
    :param d: death rate
    :param det_size: detection size
    :return: time to grow until it reached detection size
    """
    return (np.log((b - d) / b * det_size) - EULERS_CONSTANT) / (b - d)