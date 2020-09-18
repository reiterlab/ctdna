
import pytest
import numpy as np

from ctdna.bp_formulas import get_rtime_to_biomarker, get_growth_fraction_rate, get_rtime_to_event_constant_pop

__author__ = 'Johannes REITER'
__date__ = 'September 17, 2020'


@pytest.mark.parametrize('size, b, d, shedding_rate', [(1, 0.14, 0.136, 0.0001)])
def test_get_rtime_to_biomarker(size, b, d, shedding_rate):

    t_shed, growth = get_rtime_to_biomarker(size, b, d, shedding_rate)

    assert t_shed > 0
    assert growth > 0


@pytest.mark.parametrize('r, t, gr_frac_exp', [(0.1, 10, 2.718281), (0.1, 100, 22026.465)])
def test_get_growth_fraction_rate(r, t, gr_frac_exp):

    np.testing.assert_approx_equal(get_growth_fraction_rate(r, t), gr_frac_exp)


def test_get_time_to_event_constant_pop():

    rate = 0.01
    size = 100
    t = get_rtime_to_event_constant_pop(rate, size)

    assert t > 0

    ts = get_rtime_to_event_constant_pop(rate, size, n_events=5)

    assert len(ts) == 5
    assert all(ts > 0)
