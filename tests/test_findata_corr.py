"""Guard on the actual product: which ticker lands in which pack role.

A misidentified beta/epsilon/sigma/omega is silent -- the output still looks
like a plausible correlation -- so the identities are pinned against a
synthetic market whose correlations are exact by construction.
"""

from __future__ import annotations

from statistics import mean

import pytest

from conftest import EXPECTED_CORR, TRADING_DAYS
from findata_corr import PackCorrelation


def test_find_pack_correlation_identifies_pack_members(extractor):
    pack = PackCorrelation(extractor.data, extractor.ticker_dates)
    pack.define_alpha("ALPHA")
    pack.find_pack_correlation(plot_av=False)

    assert len(pack.corr_date) == len(TRADING_DAYS)

    row = pack.corr_date.iloc[0]
    assert row["Day"] == [3, 21, 2022]

    assert row["Beta"] == "TWIN"  # most correlated
    assert row["Epsilon"] == "MID"  # median correlated
    assert row["Sigma"] == "WEAK"  # least correlated by magnitude
    assert row["Omega"] == "MIRROR"  # most anti-correlated

    assert row["Beta Corr"] == pytest.approx(1.0)
    assert row["Omega Corr"] == pytest.approx(-1.0)
    assert row["Av Corr"] == pytest.approx(mean(EXPECTED_CORR.values()))

    # alpha closes the day up, so the directional correlation keeps its sign
    assert row["Alpha Gain"] == pytest.approx(1.05)
    assert row["Dir Corr"] == pytest.approx(mean(EXPECTED_CORR.values()))

    distribution = pack.dist_date[(2022, 3, 21)]
    assert sorted(distribution) == pytest.approx(sorted(EXPECTED_CORR.values()))
