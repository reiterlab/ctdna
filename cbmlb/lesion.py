"""Simulate growth dynamics of lesions and biomarker shedding"""
import logging
import math
import numpy as np
import pandas as pd
from heapq import heappush, heappop

from cbmlb.bp_formulas import get_time_to_biomarker, get_growth_fraction_rate, get_time_to_event_constant_pop

__date__ = 'October 19, 2018'
__author__ = 'Johannes REITER'

# get logger
logger = logging.getLogger(__name__)

DEATH_EVENT = 0
DIVISION_EVENT = 1
BIOMARKER_SHEDDING = 2
BIOMARKER_ELIMINATION = 3


class Lesion:

    def __init__(self, b, d, q_d, epsilon, n0=1, exact_th=1e4, q_b=0.0, lambda_1=0.0):
        """
        Initialize lesion with parameter values
        :param b: birth rate per day
        :param d: death rate per day
        :param q_d: biomarker shedding probability per cell death event
        :param epsilon: biomarker elimination rate
        :param n0: tumor start size
        :param exact_th: approximate growth of lesion after it reaches the threshold
        :param q_b: biomarker shedding probability per cell division (proliferation)
        :param lambda_1: biomarker shedding probability per unit of time (necrosis)
        """
        self.b = b  # birth rate
        self.d = d  # death rate
        self.q_d = q_d              # biomarker shedding probability per cell death (apoptosis)
        self.q_b = q_b              # biomarker shedding probability per cell division (proliferation)
        self.lambda_1 = lambda_1    # biomarker shedding probability per unit of time (necrosis)
        # calculate total biomarker shedding rate per time unit
        self.shedding_rate = self.d * self.q_d + self.b * self.q_b + self.lambda_1
        assert self.shedding_rate > 0, 'Shedding rate needs to be > 0: nu={:.3e}'.format(self.shedding_rate)
        self.epsilon = epsilon  # biomarker elimination rate
        self.size = n0  # initial lesion size
        self.bm = 0  # initial biomarker amount

        self.age = 0.0  # measured in days
        self.event_mean = self.b + self.d + self.lambda_1           # mean time to next event of cell
        self.inv_event_mean = 1.0 / self.event_mean  # inverse of mean time to next event of cell
        self.death_prob = self.d / self.event_mean  # death probability
        self.death_div_prob = (self.d + self.b) / self.event_mean  # death plus division probability
        self.inv_b = 1.0 / self.b  # inverse of the birth rate
        self.inv_d = 1.0 / self.d  # inverse of the death rate
        self.inv_epsilon = 1.0 / epsilon  # inverse of biomarker elimination rate
        self.r = self.b - self.d  # growth rate

        # event code: 0...die, 1...divide, 2...shedding (approximation), 3...biomarker is eliminated
        self.event_heap = list()

        # subpopulation size until an exact simulation has to be performed
        self.exact_th = exact_th  # approximate growth of lesion after it reaches the threshold
        if self.size > self.exact_th:
            self.approx_cells = True
            logger.info('Growth of lesion is deterministic from the beginning.'.format(self.size))
        else:
            self.approx_cells = False

        # calculate first cell event
        self._next_event_cell()

        # record history of lesion and biomarker dynamics for illustration
        self.history = np.ones([2, 1000]) * -1
        # initialize
        self.history[:, 0] = [self.size, self.bm]
        self.history_times = [0]

    def sim_to_size(self, size_th, day_resolution=0):
        """
        Simulate lesion until the given size is reached or it goes extinct
        :param size_th:
        :param day_resolution: number of decimals to round tumor age in days in logging files
        :return lesion size
        """

        if self.r == 0 and self.size > self.exact_th:
            raise RuntimeError('Lesion has a growth rate of zero and cannot grow in the deterministic regime!')

        # simulate lesion until it reaches the size threshold or the lesion goes extinct
        while self.size < size_th:
            if self._sim_events() == 0:
                return 0

            if round(self.age) != round(self.history_times[-1]):
                if int(math.log10(int(self.size))) > math.log10(int(self.history[0, len(self.history_times) - 1])):
                    self._output_sizes()

                # record dynamics of the lesion and the biomarker in CSV file
                # print(round(self.age))
                # print(round(self.history_times[-1]))
                self._log_history(day_resolution)

            elif round(self.age, day_resolution) != round(self.history_times[-1], day_resolution):
                # or (self.history[1, len(self.history_times)-1] != self.bm and self.bm < 1)):

                # record dynamics of the lesion and the biomarker in CSV file
                self._log_history(day_resolution)

        return self.size

    def sim_for_time(self, time, day_resolution=0):
        """
        Simulate lesion for a given amount of days or until it goes extinct
        :param time: in days
        :param day_resolution: number of decimals to round tumor age in days in logging files
        :return lesion size
        """

        end_time = self.age + time

        # check if tumor age after the next event would still be before the desired end time
        while self.event_heap[0][0] < end_time:

            if self._sim_events() == 0:
                return 0

            if int(self.age) != int(self.history_times[-1]):
                # record dynamics of the lesion and the biomarker in CSV file
                self._log_history(day_resolution)

                if int(self.age) % 500 == 0:
                    self._output_sizes()

        return self.size

    def export_history(self, output_fp):

        self.history[self.history == -1] = 0

        n_max_time = len(self.history_times)
        # have time in rows and subclones in columns
        history = np.zeros([n_max_time, len(self.history[:, 0]) + 1])

        # add logging times to array
        history[:, 0] = self.history_times
        # add evolutionary dynamics of the primary tumor
        history[:, 1] = self.history[0, 0:n_max_time]
        # add evolutionary dynamics of the biomarker
        history[:, 2] = self.history[1, 0:n_max_time]

        df = pd.DataFrame(history,  # index=history,
                          columns=['Time', 'LesionSize', 'BiomarkerLevel'])

        # Write results to CSV files
        df.to_csv(output_fp, index=False)
        logger.info('Exported evolutionary dynamics of lesion to {}.'.format(output_fp))

    def _sim_events(self):

        t, idx, event = heappop(self.event_heap)
        t_delta = t - self.age
        self.age = t
        # if event > 1:
        #   logger.debug(self.event_heap)
        #   logger.debug('Processing event: t_del={:.2e}, idx={}, ev={} at age {:.2e}, size {:.1e} with {} biomarkers.'
        #                  .format(t_delta, idx, event, self.age, self.size, self.cfdna))

        self._approximate_growth(t_delta)

        if event == DEATH_EVENT:  # cell dies

            self.size -= 1

            # is cfDNA shed into the blood stream?
            if self.q_d > 0.0 and np.random.rand() < self.q_d:
                # biomarker is shed during cell death
                self.bm += 1
                self._next_event_bm()

            if self.size == 0:
                # logger.debug('{:.1f}: Lesion went extinct!'.format(self.age))
                return 0

            # compute next event time
            if self.size > 0:
                self._next_event_cell()

        elif event == DIVISION_EVENT:  # cell divides

            self.size += 1

            # is cfDNA shed into the blood stream?
            if self.q_b > 0.0 and np.random.rand() < self.q_b:
                # biomarker is shed during cell death
                self.bm += 1
                self._next_event_bm()

            # compute next event time
            self._next_event_cell()

        elif event == BIOMARKER_SHEDDING:  # lesion shed biomarker (approximation)

            self.bm += 1
            self._next_event_bm()

            # compute next cell event time
            self._next_event_cell()

        elif event == BIOMARKER_ELIMINATION:  # biomarker is eliminated (exact)

            self.bm -= 1

            assert self.bm >= 0, 'Biomarker level cannot be negative!'
            # logger.debug('Biomarker was eliminated. {} remaining.'.format(self.cfdna))

        elif event == -1:
            pass

        else:
            raise RuntimeError('Not yet implemented')

        return self.size

    def _next_event_cell(self):
        """
        Calculate next event for cell
        :return time to next event
        """
        if self.size < self.exact_th:

            t = np.random.exponential(1.0 / (self.event_mean * self.size))
            rand = np.random.rand()
            # die
            if rand < self.death_prob:
                heappush(self.event_heap, (self.age + t, [0], DEATH_EVENT))

            # divide
            elif rand < self.death_div_prob:
                heappush(self.event_heap, (self.age + t, [0], DIVISION_EVENT))

            # shed biomarker without death or proliferation
            else:
                heappush(self.event_heap, (self.age + t, [0], BIOMARKER_SHEDDING))

            return self.age + t

        # lesion is too large for exact stochastic simulation => approximate
        else:
            if not self.approx_cells:
                self.approx_cells = True
                logger.debug('Growth of lesion is now deterministic. Size {:.2e}'.format(self.size))
            if self.r != 0:
                t_shed, gr = get_time_to_biomarker(self.size, self.b, self.d, self.shedding_rate)
            else:
                t_shed = get_time_to_event_constant_pop(self.shedding_rate, self.size)

            t_to_next_event = self.age + t_shed
            heappush(self.event_heap, (t_to_next_event, [0], BIOMARKER_SHEDDING))

            # check if now all subclones are approximated
            #             if all(self.is_sc_approx[sc_idx] for sc_idx, sc_size in enumerate(self.pt) if sc_size > 0):
            #                 # generate artificial events to generate accurate dynamics file for every day
            #                 heappush(self.event_heap, (self.age + 1, [], -1))

            return t_to_next_event

    def _next_event_bm(self):
        """
        Calculate next event for biomarkers (only elimination)
        """
        # implemented no approximated elimination of biomarkers yet, fully exact simulation
        # t_elim = np.random.exponential(1.0 / (self.epsilon * self.cfdna))
        t_elim = np.random.exponential(1.0 / self.epsilon)
        #         logger.debug('Time for next biomarker elimination: {:.3e}'.format(t))
        t_to_next_event = self.age + t_elim
        # eliminate biomarker
        heappush(self.event_heap, (t_to_next_event, [1], BIOMARKER_ELIMINATION))

        return t_to_next_event

    def _approximate_growth(self, t):
        """
        Perform deterministic growth for large lesion
        """
        if self.approx_cells:
            self.size *= get_growth_fraction_rate(self.r, t)

    def _log_history(self, day_resolution):

        self.history_times.append(round(self.age, day_resolution))
        # sizes for this day have not yet been recorded
        # check if history array is large enough
        if len(self.history_times) > len(self.history[0, :]):
            # add new free space to track all dynamics
            tmp = np.ones([2, 500]) * -1
            self.history = np.concatenate((self.history, tmp), axis=1)

        # log lesion size and biomarker amount
        self.history[:, len(self.history_times) - 1] = [round(self.size, 1), self.bm]

    def _output_sizes(self):

        if self.bm:
            logger.debug('t: {:.1f}: lesion size: {:.1e} with {} biomarkers.'.format(
                self.age, self.size, self.bm))
        else:
            logger.debug('t: {:.1f}: lesion size: {:.1e} with no biomarkers.'.format(
                self.age, self.size))


class PT(Lesion):
    def __init__(self, b, d, q_d, epsilon, exact_th, q_b=0.0, lambda_1=0.0):
        Lesion.__init__(self, b, d, q_d, epsilon, n0=1, exact_th=exact_th, q_b=q_b, lambda_1=lambda_1)

        logger.debug('{} initialized with r={:.2%}, b={}, d={}, nu={:.3e}, N0={:.1e}.'.format(
            self.__class__.__name__, self.r, self.b, self.d, self.q_d, self.size))


class Precursor(Lesion):
    def __init__(self, b, q_d, epsilon, n0, exact_th, q_b=0.0, lambda_1=0.0):
        Lesion.__init__(self, b, b, q_d, epsilon, n0=n0, exact_th=exact_th, q_b=q_b, lambda_1=lambda_1)

        logger.debug('{} initialized with r={:.2%}, b={}, d={}, nu={:.3e}, N0={:.1e}.'.format(
            self.__class__.__name__, self.r, self.b, self.d, self.q_d, self.size))
