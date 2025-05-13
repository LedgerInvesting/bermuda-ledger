from typing import Tuple

import numpy as np

__all__ = [
    "program_earned_premium",
]


def program_earned_premium(
    premium_volume: float,
    writing_pattern: np.ndarray,
    writing_resolution: int,
    earning_pattern: np.ndarray,
    earning_resolution: int,
    output_resolution: int,
    output_offset: int = 0,
    continuous_writing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the aggregate premium earning and writing pattern for a treaty period, based
    on the expected pattern of written premium.

    Args:
        premium_volume: Total premium to be written into the program (in dollars).
        writing_pattern: An array with relative amounts of premium that will be written
            at each time step. For examples, np.array([1, 2, 3, 4]) would represent an
            increasing amount of premium written at each successive time step.
        writing_resolution: The number of months in each time step in the
            `writing_pattern` argument.
        earning_pattern: An array with relative amounts of premium that will be earned
            over the duration of a policy. The length of the policy is equal to
            `len(earning_pattern) * earning_resolution`.
        earning_resolution: The number of months in each time step in the
            `earning_pattern` argument.
        output_resolution: The number of months in each time step of the output
            aggregate earning pattern.
        output_offset: The number of months between the start of the first period of
            the output earning pattern and the effective date of the first written
            policy. For example, if `output_resolution` is `3` (implying a quarterly
            time series with calendar-quarter periods) and the program is effective on
            March 1st, then `output_offset` would be `2`.
        continuous_writing: If `True`, then it is assumed that policy effective dates
            are spread evenly throughout each time period in `writing_pattern`.
            Otherwise, it is assumed that all policy effective dates are on the first
            day of each time period in `writing_pattern`.
    """
    normalized_writing = writing_pattern / np.sum(writing_pattern)
    monthly_writing = np.repeat(
        premium_volume * normalized_writing / writing_resolution,
        writing_resolution,
    )
    normalized_earning = earning_pattern / np.sum(earning_pattern)
    raw_monthly_earning = np.repeat(
        normalized_earning / earning_resolution, earning_resolution
    )
    if continuous_writing:
        monthly_earning = np.sum(
            [
                np.concatenate([raw_monthly_earning / 2, [0]]),
                np.concatenate([[0], raw_monthly_earning / 2]),
            ],
            axis=0,
        )
    else:
        monthly_earning = raw_monthly_earning

    N = len(monthly_writing)
    monthly_combined = np.sum(
        [
            written_premium
            * np.concatenate([[0] * n, monthly_earning, [0] * (N - n - 1)])
            for n, written_premium in enumerate(monthly_writing)
        ],
        axis=0,
    )
    output_earning_pattern = [0.0]
    output_writing_pattern = [0.0]

    start = 0
    stop = output_offset if output_offset > 0 else output_resolution
    while start < monthly_combined.size:
        output_writing_pattern.append(np.sum(monthly_writing[start:stop]))
        output_earning_pattern.append(np.sum(monthly_combined[start:stop]))
        start, stop = stop, stop + output_resolution

    return np.array(output_writing_pattern), np.array(output_earning_pattern)
