import datetime

import numpy as np
import pytest

from bermuda import Cell, Triangle, array_size, array_sizes

base_cells = [
    Cell(
        period_start=datetime.date(2019, 1, 1),
        period_end=datetime.date(2019, 12, 31),
        evaluation_date=datetime.date(2019, 12, 31),
        values={"paid_loss": np.array(500), "earned_premium": 1000},
    ),
    Cell(
        period_start=datetime.date(2019, 1, 1),
        period_end=datetime.date(2019, 12, 31),
        evaluation_date=datetime.date(2020, 12, 31),
        values={"paid_loss": np.zeros((500,)), "earned_premium": 1000},
    ),
    Cell(
        period_start=datetime.date(2019, 1, 1),
        period_end=datetime.date(2019, 12, 31),
        evaluation_date=datetime.date(2021, 12, 31),
        values={"paid_loss": np.zeros((1000,)), "earned_premium": 1000},
    ),
]

simple_tri = Triangle(base_cells[:1])
consistent_tri = Triangle(base_cells[:2])
mixed_tri = Triangle(base_cells)


def test_array_sizes():
    assert array_sizes(simple_tri) == []
    assert array_sizes(consistent_tri) == [500]
    assert array_sizes(mixed_tri) == [500, 1000]


def test_array_size():
    assert array_size(simple_tri) == 1
    assert array_size(consistent_tri) == 500
    with pytest.raises(ValueError):
        array_size(mixed_tri)
