import datetime

import pytest
from numpy.testing import assert_array_equal

from bermuda import Cell, Triangle, triangle_to_matrix
from bermuda.matrix import MatrixIndex

from .triangle_samples import raw_obs

obs_no_values = [
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 9, 30),
        values={},
    ),
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
def base_triangle():
    return Triangle(raw_obs)


@pytest.fixture
def triangle_no_values():
    return Triangle(obs_no_values)


@pytest.fixture
def base_matrix_index(base_triangle):
    return MatrixIndex.from_triangle(base_triangle)


@pytest.fixture
def base_matrix(base_triangle):
    return triangle_to_matrix(base_triangle)


def test_matrix_index_from_triangle_no_fields(triangle_no_values):
    with pytest.raises(Exception, match="Must provide fields"):
        MatrixIndex.from_triangle(triangle_no_values)

    matrix_idx = MatrixIndex.from_triangle(
        triangle_no_values, fields=["earned_premium"]
    )
    assert matrix_idx.fields == ["earned_premium"]


def test_matrix_index_from_triangle_resolution_issues(base_triangle):
    with pytest.raises(Exception, match="Must supply eval_resolution"):
        MatrixIndex.from_triangle(Triangle(raw_obs[:1]))

    with pytest.warns(UserWarning, match="Lowering resolution from"):
        matrix_index = MatrixIndex.from_triangle(
            base_triangle,
            # NOTE: this must be greater than the largest unit of resolution in the evaluation dates
            eval_resolution=10,
        )
    assert matrix_index.dev_resolution == 10


def test_matrix_index_subset(base_matrix_index):
    with pytest.raises(Exception, match="exactly 4 arguments"):
        base_matrix_index.subset((1, 2, 3, 4), [])

    # test when exp_ndxs and dev_ndxs indices are not slices
    subset = base_matrix_index.subset(
        (1, 2, 3, 4),
        0,
        0,
        1,
        2,
    )

    assert subset.exp_origin == (
        base_matrix_index.exp_origin + base_matrix_index.exp_resolution
    )

    assert subset.dev_origin == (
        base_matrix_index.dev_origin + 2 * base_matrix_index.dev_resolution
    )

    # test when exp_ndxs and dev_ndxs indices are slices
    subset = base_matrix_index.subset(
        (1, 2, 3, 4),
        0,
        [0],
        slice(0, 3),
        slice(1, 4),
    )

    assert subset.exp_origin == base_matrix_index.exp_origin
    assert subset.dev_origin == (
        base_matrix_index.dev_origin + base_matrix_index.dev_resolution
    )

    # test when indices are negative
    subset = base_matrix_index.subset(
        (1, 2, 3, 4),
        0,
        [0],
        -1,
        -2,
    )

    assert subset.exp_origin == (
        base_matrix_index.exp_origin + base_matrix_index.exp_resolution * 4
    )
    assert subset.dev_origin == (
        base_matrix_index.dev_origin + base_matrix_index.exp_resolution * 6
    )


def test_matrix_index_resolve_indices(base_matrix_index):
    with pytest.raises(Exception, match="exactly 4 arguments"):
        base_matrix_index.resolve_indices()

    # test with None indices
    res = base_matrix_index.resolve_indices(None, None, None, None)
    assert res == (None, None, None, None)

    # test with non-allowed list indices
    with pytest.raises(Exception, match="Indexing by an iterable"):
        res = base_matrix_index.resolve_indices(
            0,
            0,
            [0, 1],
            [0, 1],
        )

    # test with allowed list indices
    res = base_matrix_index.resolve_indices(
        [0, 1],
        [0, 1],
        0,
        0,
    )

    ###################
    # slice_ndx tests #
    ###################

    # test with valid slice-type slice_ndx
    res = base_matrix_index.resolve_indices(
        slice(0, 2, 1),
        None,
        None,
        None,
    )
    assert res[0] == slice(0, 2, 1)

    # test with None slice-type slice_ndx
    res = base_matrix_index.resolve_indices(
        slice(2),
        None,
        None,
        None,
    )
    assert res[0] == slice(None, 2, None)

    # test with negative slice-type slice_ndx
    with pytest.raises(Exception, match="out of range"):
        res = base_matrix_index.resolve_indices(
            slice(-1, -3, 1),
            None,
            None,
            None,
        )

    # test with Metadata-type slice_ndx
    #   Already covered in other tests???

    # test with unsupported type slice_ndx
    with pytest.raises(Exception, match="Unknown slice indexing type"):
        res = base_matrix_index.resolve_indices(
            "some string",
            None,
            None,
            None,
        )

    ###################
    # field_ndx tests #
    ###################

    # test with negative int field_ndx
    with pytest.raises(Exception, match=r"Field index .* out of range"):
        res = base_matrix_index.resolve_indices(
            None,
            -1,
            None,
            None,
        )

    # test with unsupported type field_ndx
    with pytest.raises(Exception, match="Unknown field indexing"):
        res = base_matrix_index.resolve_indices(
            None,
            1.0,  # NOTE: this float is just an aribitrary unsupported type
            None,
            None,
        )

    #################
    # exp_ndx tests #
    #################

    # test with negative int exp_ndx
    with pytest.raises(Exception, match=r"Experience index .* out of range"):
        res = base_matrix_index.resolve_indices(
            None,
            None,
            -1,
            None,
        )

    # test with date exp_ndx (in-range)
    res = base_matrix_index.resolve_indices(
        None,
        None,
        datetime.date(2022, 1, 1),
        None,
    )

    # test with date exp_ndx (out of range)
    with pytest.raises(Exception, match=r"Experience date .* out of range"):
        res = base_matrix_index.resolve_indices(
            None,
            None,
            datetime.date(2010, 1, 1),
            None,
        )

    # test with unsupported type exp_ndx
    with pytest.raises(Exception, match="Unknown experience period"):
        res = base_matrix_index.resolve_indices(
            None,
            None,
            1.0,  # NOTE: this float is just an aribitrary unsupported type
            None,
        )

    #################
    # dev_ndx tests #
    #################

    # test negative float dev_ndx
    with pytest.raises(Exception, match=r"Development lag .* out of range"):
        res = base_matrix_index.resolve_indices(
            None,
            None,
            None,
            -3.0,
        )

    # test negative int dev_ndx
    with pytest.raises(Exception, match=r"Development index .* out of range"):
        res = base_matrix_index.resolve_indices(
            None,
            None,
            None,
            -1,
        )

    # test unsupported dev_ndx
    with pytest.raises(Exception, match=r"Unknown dev lag indexing"):
        res = base_matrix_index.resolve_indices(
            None,
            None,
            None,
            "some string",
        )


def test_matrix_subset(base_matrix):
    res = base_matrix.subset([0, 0, slice(0, 2), 0])

    assert res.incremental == base_matrix.incremental
    assert_array_equal(res.data, [100, 150])


def test_matrix_setters(base_matrix):
    idx = (0, 1, 2, 0)
    assert base_matrix[idx] == 500
    base_matrix[idx] = 1000
    assert base_matrix[idx] == 1000


def test_matrix_getters(base_matrix):
    assert base_matrix[0, 0, 0, 0] == 100
    assert base_matrix[0, 1, 2, 0] == 500
