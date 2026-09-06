"""Guard on the actual product: which ticker lands in which pack role.

A misidentified beta/epsilon/sigma/omega is silent -- the output still looks
like a plausible correlation -- so the identities are pinned against a
synthetic market whose correlations are exact by construction.
"""

from __future__ import annotations

import datetime as dt
from math import isnan
from statistics import mean

import pytest

from conftest import CLOSES, EXPECTED_CORR, TRADING_DAYS
from findata_corr import PackCorrelation


def _pack_for(extractor) -> PackCorrelation:
    pack = PackCorrelation(extractor.data)
    pack.define_alpha("ALPHA")
    pack.find_pack_correlation(plot_av=False)

    return pack


def test_find_pack_correlation_identifies_pack_members(extractor):
    pack = _pack_for(extractor)

    assert len(pack.corr_date) == len(TRADING_DAYS)
    assert list(pack.corr_date.index.date) == TRADING_DAYS

    row = pack.corr_date.loc["2022-03-21"]

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

    distribution = pack.dist_date[dt.date(2022, 3, 21)]
    assert sorted(distribution) == pytest.approx(sorted(EXPECTED_CORR.values()))


def test_correlation_aligns_on_timestamps_not_positions(gapped_extractor):
    """A missing mid-day bar must drop that minute, not shift the rest.

    GAPPED is exactly 2 * ALPHA on every bar it has, so its correlation is
    1.0. Aligning by position instead scores it at roughly 0.68.
    """
    pack = _pack_for(gapped_extractor)

    row = pack.corr_date.iloc[0]
    assert row["Beta"] == "GAPPED"
    assert row["Beta Corr"] == pytest.approx(1.0)

    # the pack holds a single ticker, so there is no spread to report
    assert isnan(row["Stdev Corr"])


def test_each_class_keeps_its_own_default_ticker(extractor):
    """The one difference the shared accessor has to preserve.

    Both classes inherit the same slice_data; picking the wrong default
    would quietly return a different ticker's prices.
    """
    pack = PackCorrelation(extractor.data)
    pack.define_alpha("MIRROR")

    assert extractor.default_ticker() == next(iter(extractor.data))
    assert pack.default_ticker() == "MIRROR"

    first_loaded = extractor.default_ticker()
    assert extractor.slice_data()["Close"].iloc[0] == CLOSES[first_loaded][0]
    assert pack.slice_data()["Close"].iloc[0] == CLOSES["MIRROR"][0]


def test_days_without_usable_correlations_are_skipped(disjoint_extractor):
    """One unusable day must not abort the whole run."""
    pack = _pack_for(disjoint_extractor)

    assert len(pack.corr_date) == 0
    assert pack.dist_date[dt.date(2022, 3, 21)] == []
