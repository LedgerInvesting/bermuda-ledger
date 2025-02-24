import datetime

import numpy as np

from bermuda.base import Cell, CumulativeCell, IncrementalCell, Metadata

raw_obs = [
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": 160, "reported_loss": 270},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 190, "reported_loss": 310},
    ),
    Cell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 6, 30),
        metadata=Metadata(),
        values={"paid_loss": 150, "reported_loss": 400},
    ),
    Cell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 220, "reported_loss": 480},
    ),
    Cell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 9, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 120, "reported_loss": 500},
    ),
]

array_raw_obs = [
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": np.full(10, 100), "reported_loss": 200},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": np.full(10, 160), "reported_loss": 270},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": np.full(10, 190), "reported_loss": 310},
    ),
    Cell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 6, 30),
        metadata=Metadata(),
        values={"paid_loss": np.full(10, 150), "reported_loss": 400},
    ),
    Cell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": np.full(10, 220), "reported_loss": 480},
    ),
    Cell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 9, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": np.full(10, 120), "reported_loss": 500},
    ),
]

period_agg_obs = [
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": 310, "reported_loss": 670},
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 410, "reported_loss": 790},
    ),
    Cell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 12, 31),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 120, "reported_loss": 500},
    ),
]

premium_obs = [
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"earned_premium": 300},
    ),
    Cell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"earned_premium": 500},
    ),
    Cell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 9, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"earned_premium": 800},
    ),
]

raw_incremental_obs = [
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        prev_evaluation_date=datetime.date(2019, 12, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        prev_evaluation_date=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": 60, "reported_loss": 70},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        prev_evaluation_date=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 30, "reported_loss": 40},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        prev_evaluation_date=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": 150, "reported_loss": 400},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        prev_evaluation_date=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 70, "reported_loss": 80},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 9, 30),
        prev_evaluation_date=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 120, "reported_loss": 500},
    ),
]

period_agg_inc_obs = [
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 6, 30),
        prev_evaluation_date=datetime.date(2019, 12, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 6, 30),
        prev_evaluation_date=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": 210, "reported_loss": 470},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 6, 30),
        prev_evaluation_date=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 100, "reported_loss": 120},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 12, 31),
        prev_evaluation_date=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 120, "reported_loss": 500},
    ),
]

eval_agg_inc_obs = [
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        prev_evaluation_date=datetime.date(2019, 12, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        prev_evaluation_date=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 90, "reported_loss": 110},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        prev_evaluation_date=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 220, "reported_loss": 480},
    ),
    IncrementalCell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 9, 30),
        prev_evaluation_date=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 120, "reported_loss": 500},
    ),
]

raw_cumulative_obs = [
    CumulativeCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
    ),
    CumulativeCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": 160, "reported_loss": 270},
    ),
    CumulativeCell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 9, 30),
        values={"paid_loss": 190, "reported_loss": 310},
    ),
    CumulativeCell(
        period_start=datetime.date(2020, 4, 1),
        period_end=datetime.date(2020, 6, 30),
        evaluation_date=datetime.date(2020, 6, 30),
        metadata=Metadata(),
        values={"paid_loss": 150, "reported_loss": 400},
    ),
]

disordered_obs = [
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 6, 30),
        values={"paid_loss": 160, "reported_loss": 270},
        metadata=Metadata(details={"a": 1}),
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
        metadata=Metadata(details={"a": 2}),
    ),
    Cell(
        period_start=datetime.date(2020, 1, 1),
        period_end=datetime.date(2020, 3, 31),
        evaluation_date=datetime.date(2020, 3, 31),
        values={"paid_loss": 100, "reported_loss": 200},
        metadata=Metadata(details={"a": 1}),
    ),
]

experience_gap_obs = [
    Cell(
        period_start=datetime.date(2019, 1, 1),
        period_end=datetime.date(2019, 3, 31),
        evaluation_date=datetime.date(2020, 12, 31),
        values={},
    ),
    Cell(
        period_start=datetime.date(2019, 10, 1),
        period_end=datetime.date(2019, 12, 31),
        evaluation_date=datetime.date(2020, 12, 31),
        values={},
    ),
    Cell(
        period_start=datetime.date(2020, 7, 1),
        period_end=datetime.date(2020, 12, 31),
        evaluation_date=datetime.date(2020, 12, 31),
        values={},
    ),
]

unequal_res_obs_dev = [
    Cell(
        period_start=datetime.date(2022, 1, 1),
        period_end=datetime.date(2022, 3, 31),
        evaluation_date=datetime.date(2024, 9, 30),
        values={"reported_loss": 1},
    ),
    Cell(
        period_start=datetime.date(2022, 1, 1),
        period_end=datetime.date(2022, 3, 31),
        evaluation_date=datetime.date(2025, 9, 30),
        values={"reported_loss": 1},
    ),
    Cell(
        period_start=datetime.date(2022, 4, 1),
        period_end=datetime.date(2022, 6, 30),
        evaluation_date=datetime.date(2024, 9, 30),
        values={"reported_loss": 1},
    ),
    Cell(
        period_start=datetime.date(2022, 4, 1),
        period_end=datetime.date(2022, 6, 30),
        evaluation_date=datetime.date(2025, 9, 30),
        values={"reported_loss": 1},
    ),
]
