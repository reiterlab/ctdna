
import pytest

import numpy as np
import scipy as sp

from ctdna.sampling import take_liquid_biopsies
import ctdna.settings as settings

__author__ = 'Johannes REITER'
__date__ = 'September 17, 2020'


@pytest.mark.parametrize('ctdna_hge_het, wt_hges_per_ml, tube_size',
                         [(np.array([100]), np.array([0]), settings.BLOOD_AMOUNT),
                          (np.array([100]), np.array([100]), settings.BLOOD_AMOUNT)])
def test_take_liquid_biopsies(ctdna_hge_het, wt_hges_per_ml, tube_size):

    plasma_ml = settings.BLOOD_AMOUNT * settings.PLASMA_FRACTION * 1000

    ctdna_vafs, ctdna_hges_sampled, ctdna_vafs_sampled = take_liquid_biopsies(
        ctdna_hge_het, wt_hges_per_ml, tube_size=tube_size)

    assert ctdna_vafs == ctdna_hge_het[0] / (2 * (ctdna_hge_het[0] + wt_hges_per_ml[0] * plasma_ml))
    assert np.all(0 <= ctdna_hges_sampled <= 2 * (ctdna_hge_het + wt_hges_per_ml * plasma_ml))
