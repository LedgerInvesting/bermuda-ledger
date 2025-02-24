from datetime import date

import bermuda as tri


def test_off_diagonal_drop():
    starting_tri = tri.long_csv_to_triangle("test/test_data/meyers_long.csv").clip(
        max_eval=date(1997, 12, 31)
    )
    off_diagonal_end = starting_tri + starting_tri.right_edge.replace(
        evaluation_date=lambda cell: tri.date_utils.add_months(cell.evaluation_date, 1)
    )
    off_diagonal_middle = starting_tri + starting_tri.right_edge.replace(
        evaluation_date=lambda cell: tri.date_utils.add_months(cell.evaluation_date, -1)
    )

    off_diagonal_end_3 = starting_tri + starting_tri.right_edge.replace(
        evaluation_date=lambda cell: tri.date_utils.add_months(cell.evaluation_date, 3)
    )
    off_diagonal_middle_3 = starting_tri + starting_tri.right_edge.replace(
        evaluation_date=lambda cell: tri.date_utils.add_months(cell.evaluation_date, -3)
    )

    ragged_tri = tri.binary_to_triangle("test/test_data/ragged_aq_triangle.trib")
    off_diagonal_ragged = ragged_tri + ragged_tri.clip(
        max_eval=date(2020, 12, 31)
    ).right_edge.replace(
        evaluation_date=lambda cell: tri.date_utils.add_months(cell.evaluation_date, 1)
    )

    assert tri.drop_off_diagonals(off_diagonal_end) == starting_tri
    assert tri.drop_off_diagonals(off_diagonal_middle) == starting_tri
    assert tri.drop_off_diagonals(off_diagonal_end_3) == starting_tri
    assert tri.drop_off_diagonals(off_diagonal_middle_3) == starting_tri
    assert tri.drop_off_diagonals(off_diagonal_ragged) == ragged_tri
