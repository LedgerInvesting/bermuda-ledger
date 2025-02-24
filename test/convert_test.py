import datetime

import numpy as np
import pytest

from bermuda import Cell, IncrementalCell, Triangle
from bermuda.date_utils import resolution_delta
from bermuda.io import (
    matrix_to_triangle,
    rich_matrix_to_triangle,
    triangle_to_matrix,
    triangle_to_rich_matrix,
)

from .triangle_samples import (
    array_raw_obs,
    period_agg_obs,
    raw_incremental_obs,
    raw_obs,
    unequal_res_obs_dev,
)

base_tri = Triangle(raw_obs)
array_tri = Triangle(array_raw_obs)
tall_narrow_tri = Triangle(unequal_res_obs_dev)
short_wide_tri = Triangle(period_agg_obs)


@pytest.fixture
def triangle_incremental():
    return Triangle(raw_incremental_obs)


@pytest.fixture
def obs_no_values():
    return [
        Cell(
            period_start=datetime.date(2020, 4, 1),
            period_end=datetime.date(2020, 6, 30),
            evaluation_date=datetime.date(2020, 6, 30),
            values={},
        ),
        Cell(
            period_start=datetime.date(2020, 4, 1),
            period_end=datetime.date(2020, 6, 30),
            evaluation_date=datetime.date(2020, 9, 30),
            values={},
        ),
        Cell(
            period_start=datetime.date(2020, 7, 1),
            period_end=datetime.date(2020, 9, 30),
            evaluation_date=datetime.date(2020, 9, 30),
            values={},
        ),
    ]


@pytest.fixture
def triangle_no_values(obs_no_values):
    return Triangle(obs_no_values)


def test_triangle_requirements():
    # triangle is not monthly
    with pytest.raises(Exception):
        triangle_to_matrix(
            base_tri.replace(
                period_start=lambda cell: resolution_delta(
                    cell.period_start, (15, "day")
                )
            )
        )
    with pytest.raises(ValueError):
        triangle_to_rich_matrix(
            base_tri.replace(
                period_start=lambda cell: resolution_delta(
                    cell.period_start, (15, "day")
                )
            )
        )
    # triangle is not semi-regular
    with pytest.raises(Exception):
        triangle_to_matrix(
            Triangle(
                [raw_obs[0]]
                + [raw_obs[3].replace(period_end=datetime.date(2020, 5, 31))]
            )
        )

    # triangle cell(s) has inf dev lag
    #   FIXME: I don't think this code is reachable. This comment can be
    #   removed when the code is cleaned up.


def test_mat():
    rmat = triangle_to_matrix(base_tri)
    back = matrix_to_triangle(rmat)
    assert isinstance(back, Triangle)


def test_mat_cell_type(triangle_incremental):
    rmat = triangle_to_matrix(base_tri)
    back = matrix_to_triangle(rmat)
    assert base_tri.is_incremental == back.is_incremental

    # test with incremental cells
    mat_incr = triangle_to_matrix(triangle_incremental)
    incr_tri = matrix_to_triangle(mat_incr)

    assert mat_incr.incremental and incr_tri.is_incremental


def test_mat_fields(triangle_no_values):
    for field in ["paid_loss", "reported_loss"]:
        rmat = triangle_to_matrix(base_tri, fields=[field])
        back = matrix_to_triangle(rmat)
        assert isinstance(back, Triangle)
        assert back.fields == [field]

    # triangle has no `fields` and no `fields` arg provided
    with pytest.raises(Exception, match="Must specify fields"):
        triangle_to_matrix(triangle_no_values)


def test_mat_fields_empty():
    empty = base_tri.replace(values={})
    rmat = triangle_to_matrix(empty, fields=["paid_loss", "reported_loss"])
    assert rmat.data.shape[1] == 2


def test_mat_field_values():
    for field in ["paid_loss", "reported_loss"]:
        rmat = triangle_to_matrix(base_tri, fields=[field])
        back = matrix_to_triangle(rmat)
        assert (
            np.max([abs(ob1[field] - ob2[field]) for ob1, ob2 in zip(base_tri, back)])
            == 0
        )


def test_rich_mat_square_tri():
    rmat = triangle_to_rich_matrix(array_tri)
    back = rich_matrix_to_triangle(rmat)
    assert isinstance(back, Triangle)
    assert array_tri == back


def test_rich_mat_tall_narrow_tri():
    rmat = triangle_to_rich_matrix(tall_narrow_tri)
    back = rich_matrix_to_triangle(rmat)
    assert isinstance(back, Triangle)
    assert tall_narrow_tri == back


def test_rich_mat_short_wide_tri():
    rmat = triangle_to_rich_matrix(short_wide_tri)
    back = rich_matrix_to_triangle(rmat)
    assert isinstance(back, Triangle)
    assert short_wide_tri == back


def test_rich_mat_cell_type(triangle_incremental):
    rmat = triangle_to_rich_matrix(array_tri)
    back = rich_matrix_to_triangle(rmat)
    assert array_tri.is_incremental == back.is_incremental

    incr_mat = triangle_to_rich_matrix(triangle_incremental)
    incr_tri = rich_matrix_to_triangle(incr_mat)
    assert incr_tri.is_incremental
    assert all([type(c) is IncrementalCell for c in incr_tri.cells])


def test_rich_mat_fields():
    for field in ["paid_loss", "reported_loss"]:
        rmat = triangle_to_rich_matrix(array_tri, fields=[field])
        back = rich_matrix_to_triangle(rmat)
        assert isinstance(back, Triangle)
        assert back.fields == [field]


def test_rich_mat_field_values():
    for field in ["paid_loss", "reported_loss"]:
        rmat = triangle_to_rich_matrix(array_tri, fields=[field])
        back = rich_matrix_to_triangle(rmat)
        assert (
            np.max([abs(ob1[field] - ob2[field]) for ob1, ob2 in zip(array_tri, back)])
            == 0
        )


def test_rich_mat_fields_empty():
    empty = array_tri.replace(values={})
    rmat = triangle_to_rich_matrix(empty, fields=["paid_loss", "reported_loss"])
    assert rmat.data.shape[1] == 2
