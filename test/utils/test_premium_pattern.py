import numpy as np

from bermuda import program_earned_premium


def test_program_earned_premium():
    # 1 6 month policy  written in the middle of the first month
    pattern = program_earned_premium(
        premium_volume=600,
        writing_pattern=np.array([1]),
        writing_resolution=1,
        earning_pattern=np.array([1, 1, 1, 1, 1, 1]),
        earning_resolution=1,
        output_resolution=1,
    )
    written, earned = pattern
    np.testing.assert_array_equal(written, np.array([0, 600, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(earned, np.array([0, 50, 100, 100, 100, 100, 100, 50]))
