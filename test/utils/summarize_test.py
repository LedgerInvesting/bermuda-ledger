import datetime
from itertools import product

import numpy as np
import pytest

from bermuda import Cell, Metadata, Triangle, TriangleError, summarize
from bermuda.utils.summarize import _conforming_sum


def _make_industry_triangle():
    return Triangle(
        [
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2020, 12, 31),
                values={
                    "log_industry_lr": ind_lr,
                    "earned_premium": ep,
                },
                metadata=Metadata(details={"coverage": coverage}),
            )
            for coverage, ep, ind_lr in zip(
                ["BI", "PD"],
                [1.0, 5.0],
                [
                    np.log(np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
                    np.log(np.array([0.4, 0.4, 0.4, 0.4, 0.4])),
                ],
            )
        ]
    )


def test_loss_detail_summarization():
    """We should be able to intelligently combine premium over loss_details
    and per_occurrence_limit layers."""
    loss_detail_tri = Triangle(
        [
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2020, 12, 31),
                values={"reported_loss": 1, "earned_premium": 1},
                metadata=Metadata(
                    per_occurrence_limit=limit, loss_details={"coverage": coverage}
                ),
            )
            for limit, coverage in product([1_000, 2_000], ["BI", "PD"])
        ]
    )
    summarized = loss_detail_tri.summarize(summarize_premium=False)
    assert len(summarized) == 1
    assert summarized.cells[0]["reported_loss"] == 4
    assert summarized.cells[0]["earned_premium"] == 1


def test_atu_summarization():
    """We should be able to intelligently combine atus"""
    atu_tri = Triangle(
        [
            Cell(
                period_start=datetime.date(2020, 1, 1),
                period_end=datetime.date(2020, 12, 31),
                evaluation_date=datetime.date(2020, 12, 31),
                values={"reported_loss": 1, "earned_premium": 1, "implied_atu": atu},
                metadata=Metadata(details={"coverage": coverage}),
            )
            for coverage, atu in zip(["BI", "PD"], [2.0, 4.0])
        ]
    )
    summarized = atu_tri.summarize()
    assert len(summarized) == 1
    assert summarized.cells[0]["reported_loss"] == 2
    assert summarized.cells[0]["earned_premium"] == 2
    assert summarized.cells[0]["implied_atu"] == 3


def test_log_industry_lr_summarization():
    """When industry loss ratio is available, should be able to combine"""
    industry_tri = _make_industry_triangle()
    summarized = industry_tri.summarize()
    assert len(summarized) == 1
    # should be weighted average of arrays, where weighting done
    # by program premium volume
    assert np.mean(summarized.cells[0]["log_industry_lr"]) == np.log(
        0.2 * (1 / 6) + 0.4 * (5 / 6)
    )
    assert summarized.cells[0]["earned_premium"] == 6


def test_custom_field_summarization():
    """When custom field is summarized, must specify aggregating function"""
    industry_tri = _make_industry_triangle().replace(
        values=lambda ob: dict(custom_field=ob["log_industry_lr"])
    )
    # error if aggregating function not given
    with pytest.raises(TriangleError):
        industry_tri.summarize()

    # passes when function for custom field given
    summarized = industry_tri.summarize(
        summary_fns={"custom_field": lambda vd: _conforming_sum(vd["custom_field"])},
    )
    assert len(summarized) == 1
    assert np.array_equal(
        summarized.cells[0]["custom_field"],
        np.log(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        + np.log(np.array([0.4, 0.4, 0.4, 0.4, 0.4])),
    )
