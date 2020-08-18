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
    return np.random.exponential(abs(b0 - d0) / shedding_rate, size=n_bms) * np.sign(b0 - d0)


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
    # NOTE: once the product of the tumor size the shedding rate becomes larger than 1e22, python reaches its
    # numerical precision and the returned time will often be zero
    growth = get_growth_to_biomarker(b0, d0, shedding_rate)
    return np.log(max(sc_sizes + growth, 1) / sc_sizes) / (b0 - d0), growth


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


def get_bms_at_size(tumor_size, shedding_rate, growth_rate, elimination_rate, sample_fraction=1.0):
    """
    Get the mean number of biomarkers for a tumor at a given size (Eq. 1, Avanzini et al, 2020)
    :param tumor_size: size of the tumor
    :param shedding_rate: biomarker shedding rate per unit of time [per day]
    :param growth_rate: tumor growth rate per day
    :param elimination_rate: biomarker elimination rate per day
    :param sample_fraction: fraction of blood stream that is sampled
    :return: mean number of biomarkers for a tumor at a given size
    """
    return tumor_size * shedding_rate / (growth_rate + elimination_rate) * sample_fraction


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


def get_rtime_to_sur_mut_supexp(clone_size, b0, d0, b1, d1, u, n_muts=None, det_size=1e8):
    """
    Get exponentially distributed time to the occurrence of n conditioned on survival mutations and the growth
    of the ancestral subclones in that time;
    Mutations are only acquired per cell division instead of seeding times occurring per unit of time
    Accounts for super exponential growth of originating subpopulation early on
    :param clone_size: size of subclone that generates the new mutations
    :param b0: birth rate of the wildtype
    :param d0: death rate of the wildtype
    :param b1: birth rate of the mutant
    :param d1: death rate of the mutant
    :param u: mutation rate
    :param n_muts: number of surviving mutant subclones if None then one is assumed
    :param det_size: some large size such that accounting for super exponential growth becomes possible
    :return: detection time, array of n random occurrence times, array of n random growth fractions to next n events
    """
    # growth rate of cells of original clone
    r0 = b0 - d0
    # assume variance in mutation arrival time is much larger than variance in time to exponential growth behavior
    det_time = cond_stochastic_growth_time(b0, d0, det_size)
    # account for stochastic super-exponential growth phase early on
    effective_time_correction = det_time - np.log(det_size) / r0
    if n_muts is None:
        growth = get_growth_to_sur_mut(b0, d0, b1, d1, u, n_muts=n_muts)
        return det_time, (np.log((clone_size + growth) / clone_size) / r0) + effective_time_correction, growth
    else:
        growths = get_growth_to_sur_mut(b0, d0, b1, d1, u, n_muts=n_muts)
        time_to_sur_muts = np.zeros(n_muts)
        cur_time = 0
        cur_size = clone_size
        for sc, gr_dr in enumerate(growths, 0):
            # amount of time passed from when the last subclone was generated to this one
            t = np.log((cur_size + gr_dr) / cur_size) / r0
            cur_time += t
            time_to_sur_muts[sc] = cur_time
            cur_size += gr_dr

        # time_to_sur_muts = [np.log((clone_size + growth) / clone_size) / (b0 - d0) for growth in growths]
        return det_time, time_to_sur_muts + effective_time_correction, growths


def cond_stochastic_growth_time(b, d, det_size, n_samples=None):
    """
    Calculate stochastic growth time to detection size starting with a single cell conditioned on survival
    accounting for the superexponential growth phase of a small population
    (see Rick Durrett, Branching Process Models of Cancer, 2015)
    :param b: birth rate
    :param d: death rate
    :param det_size: detection size
    :param n_samples: number of drawn samples
    :return: time to grow until it reached detection size
    """
    r = b - d
    location = np.log(r / b * det_size) / r
    scale = 1.0 / r
    return np.random.gumbel(loc=location, scale=scale, size=n_samples)


def get_growth_to_sur_mut(b0, d0, b1, d1, u, n_muts=None):
    """
    Get exponentially distributed growth to the occurrence of n conditioned on survival mutations;
    mutations are only acquired per cell division instead of seeding times occurring per unit of time
    :param b0: birth rate of the wildtype
    :param d0: death rate of the wildtype
    :param b1: birth rate of the mutant
    :param d1: death rate of the mutant
    :param u: mutation rate
    :param n_muts: number of drawn random samples
    :return: array of n drawn random numbers
    """
    return np.random.exponential((b0 - d0) * b1 / (b0 * (b1 - d1) * u), size=n_muts)


def cond_stochastic_growth(b, d, growth_time):
    """
    Calculate stochastic growth time to detection size starting with a single cell conditioned on survival
    accounting for the superexponential growth phase of a small population
    (see Rick Durrett, Branching Process Models of Cancer, 2015)
    WARNING: only works if r * growth time >> 1
    :param b: birth rate
    :param d: death rate
    :param growth_time: time for growth
    :return: time to grow until it reached detection size
    """
    r = b - d
    # correct for super-exponential growth when the PT is small
    t_to_large_size = cond_stochastic_growth_time(b, d, 1e10)
    # account for stochastic super-exponential growth phase early on
    effective_time_correction = t_to_large_size - (np.log(1e10) / r)
    # print(effective_time_correction)

    return get_growth_fraction_rate(r, growth_time - effective_time_correction)


def get_growth_to_dissemination(r, q, n_mets=None):
    """
    Get exponentially distributed growth to the dissemination of n cells
    :param r: growth rate of cells at original site
    :param q: dissemination rate
    :param n_mets: number of successfully seeded metastases
    :return: array of random growth fractions to next n metastases
    """
    return np.random.exponential(r / q, size=n_mets)


def get_growth_to_sur_dissemination(b_met, r_pri, r_met, q, n_mets=None):
    """
    Get exponentially distributed growth to the dissemination of n conditioned on survival cells
    :param b_met: birth rate of cells at new site
    :param r_pri: growth rate of cells at original site
    :param r_met: growth rate of cells at new site
    :param q: dissemination rate
    :param n_mets: number of successfully seeded metastases
    :return: array of random growth fractions to next n metastases
    """
    # sp.stats.expon.rvs(scale=b_met * r_pri / (r_met * q), size=n_mets)
    return np.random.exponential(b_met * r_pri / (r_met * q), size=n_mets)


def get_rtime_to_dissemination(sc_sizes, r, q, n_mets=None):
    """
    Get exponentially distributed time to the dissemination of n mets conditioned on survival cells and the growth of
    the ancestral subclones in that time;
    :param sc_sizes: size of subclone that generates the new mutationif n_mets is not None needs
                     to be an array of that size
    :param r: growth rate of cells at original site
    :param q: dissemination rate
    :param n_mets: number of successfully seeded metastases if not None then sc_sizes needs to be an array of that size
    :return: array of random growth fractions to next n metastases
    """
    if n_mets is None:
        growth = get_growth_to_dissemination(r, q, n_mets=n_mets)
        return np.log((sc_sizes + growth) / sc_sizes) / r, growth
    else:

        growths = get_growth_to_dissemination(r, q, n_mets=n_mets)
        times_to_dissemination = [np.log((sc_size + growth) / sc_size) / r
                                  for growth, sc_size in zip(growths, sc_sizes)]
        return times_to_dissemination, growths


def get_rtime_to_sur_dissemination(sc_sizes, b_met, r_pri, r_met, q, n_mets=None):
    """
    Get exponentially distributed time to the dissemination of n conditioned on survival cells and the growth of
    the ancestral subclones in that time;
    :param sc_sizes: size of subclone that generates the new mutation if n_mets is not None needs
                     to be an array of that size
    :param b_met: birth rate of cells at new site
    :param r_pri: growth rate of cells at original site
    :param r_met: growth rate of cells at new site
    :param q: dissemination rate
    :param n_mets: number of successfully seeded metastases if not None then sc_sizes needs to be an array of that size
    :return: array of random times to next n events, array of random growth fractions to next n dissemination events
    """
    if n_mets is None:
        growth = get_growth_to_sur_dissemination(b_met, r_pri, r_met, q, n_mets=n_mets)
        return np.log((sc_sizes + growth) / sc_sizes) / r_pri, growth
    else:

        growths = get_growth_to_sur_dissemination(b_met, r_pri, r_met, q, n_mets=n_mets)
        times_to_dissemination = [np.log((sc_size + growth) / sc_size) / r_pri
                                  for growth, sc_size in zip(growths, sc_sizes)]
        return times_to_dissemination, growths


def get_rtime_to_sur_dissemination_supexp(sc_sizes, b_pri, d_pri, b_met, r_met, q, n_mets=None):
    """
    Get exponentially distributed time to the dissemination of n mets conditioned on survival cells and the growth of
    the ancestral subclones in that time;
    Accounts for super exponential growth of originating subpopulation early on
    :param sc_sizes: size of subclone that generates the new mutation if n_mets is not None needs
                     to be an array of that size
    :param b_pri: birth rate of cells in primary tumor
    :param d_pri: death rate of cells in primary tumor
    :param b_met: birth rate of cells at new site
    :param r_met: growth rate of cells at new site
    :param q: dissemination rate
    :param n_mets: number of successfully seeded metastases if not None then sc_sizes needs to be an array of that size
    :return: array of random times to next n events, array of random growth fractions to next n events
    """
    # growth rate of cells at original site
    r_pri = b_pri - d_pri
    # assume variance in mutation arrival time is much larger than variance in time to exponential growth behavior
    t_to_large_size = cond_stochastic_growth_time(b_pri, d_pri, 1e8, n_samples=n_mets)
    # account for stochastic super-exponential growth phase early on
    effective_time_correction = t_to_large_size - np.log(1e8) / r_pri
    if n_mets is None:
        growth = get_growth_to_sur_dissemination(b_met, r_pri, r_met, q, n_mets=n_mets)
        return np.log((sc_sizes + growth) / sc_sizes) / r_pri + effective_time_correction, growth
    else:

        growths = get_growth_to_sur_dissemination(b_met, r_pri, r_met, q, n_mets=n_mets)
        times_to_dissemination = [np.log((sc_size + growth) / sc_size) / r_pri
                                  for growth, sc_size in zip(growths, sc_sizes)] + effective_time_correction
        return times_to_dissemination, growths


def get_etime_to_sur_dissemination(b_org, d_org, b_dist, d_dist, q):
    """
    Get expected time to the first successful colonization;
    Dissemination events happen per unit of time and not per cell division
    This does not account for super exponential growth of subpopulations early on
    :param b_org: birth rate at the original site
    :param d_org: death rate at the original site
    :param b_dist: birth rate at the distant site
    :param d_dist: death rate at the distant site
    :param q: dissemination rate
    :return: expected time to the seeding of the first surviving metastases
    """
    r_org = b_org - d_org
    r_dist = b_dist - d_dist

    return - (np.log(q / r_org * r_dist / b_dist) + EULERS_CONSTANT) / r_org


def get_etime_to_sur_dissemination_supexp(b_org, d_org, b_dist, d_dist, q):
    """
    Get expected time to the first successful colonization;
    Dissemination events happen per unit of time and not per cell division
    Accounts for super exponential growth of subpopulations early on
    :param b_org: birth rate at the original site
    :param d_org: death rate at the original site
    :param b_dist: birth rate at the distant site
    :param d_dist: death rate at the distant site
    :param q: dissemination rate
    :return: expected time to the seeding of the first surviving metastases
    """
    r_org = b_org - d_org
    r_dist = b_dist - d_dist

    return (np.log(r_org / q * r_org / r_dist * b_dist / b_org) - EULERS_CONSTANT) / r_org


def get_growth_to_mut(b0, d0, u, n_muts=None):
    """
    Get exponentially distributed growth/decline to the occurrence of 1 or n mutations;
    mutations are only acquired per cell division instead of seeding times occurring per unit of time
    :param b0: birth rate of the wildtype
    :param d0: death rate of the wildtype
    :param u: mutation probability
    :param n_muts: number of drawn random samples
    :return: array of n drawn random numbers
    """
    return np.random.exponential(abs(b0 - d0) / (b0 * u), size=n_muts)


def get_time_to_mut(sc_sizes, b0, d0, u, n_muts=None):
    """
    Get exponentially distributed time to the occurrence of 1 or n mutations and the growth of the ancestral subclones
    in that time;
    Mutations are only acquired per cell division instead of seeding times occurring per unit of time
    :param sc_sizes: size of subclone that generates the new mutation, if n_muts is not None needs
                     to be an array of that size
    :param b0: birth rate of the wildtype
    :param d0: death rate of the wildtype
    :param u: mutation probability
    :param n_muts: number of subclones if not None then sc_sizes needs to be an array of that size
    :return: array of n_muts times, array of n_muts growths
    """
    if n_muts is None:
        growth = get_growth_to_mut(b0, d0, u, n_muts=n_muts)
        return np.log((sc_sizes + growth) / sc_sizes) / (b0 - d0), growth
    else:
        growths = get_growth_to_mut(b0, d0, u, n_muts=n_muts)
        time_to_sur_muts = [np.log((sc_size + growth) / sc_size) / (b0 - d0)
                            for growth, sc_size in zip(growths, sc_sizes)]
        return time_to_sur_muts, growths
