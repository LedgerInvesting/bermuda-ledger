import numpy as np

import bermuda as tri
from bermuda import bootstrap
from bermuda.utils.bootstrap import maximum_entropy_ensemble

SEED = 1234
TRIANGLE = tri.io.json_to_triangle("test/test_data/meyers_triangle.json")


def test_basic_bootstrap():
    reps = sum(bootstrap(TRIANGLE, n=2, seed=SEED))
    assert len(reps.slices) == 2
    assert reps.periods == TRIANGLE.periods
    assert reps.evaluation_dates == TRIANGLE.evaluation_dates


def test_single_experience_period_bootstrap():
    clipped_triangle = TRIANGLE.clip(min_dev=0, max_dev=0)
    reps = sum(bootstrap(clipped_triangle, n=2, seed=SEED))

    assert len(reps.slices) == 2
    assert all(len(s) == len(clipped_triangle) for s in reps.slices.values())
    assert reps.periods == clipped_triangle.periods
    assert reps.evaluation_dates == clipped_triangle.evaluation_dates


def test_bootstrap_with_zero_fields():
    triangle = TRIANGLE.derive_fields(x=0)
    reps = sum(bootstrap(triangle, n=2, seed=SEED))
    assert len(reps.slices) == 2
    assert reps.periods == TRIANGLE.periods
    assert reps.evaluation_dates == TRIANGLE.evaluation_dates


def test_bootstrap_multislice_triangle():
    triangle = sum(
        TRIANGLE.derive_metadata(details=TRIANGLE.metadata[0].details | {"slice": i})
        for i in range(2)
    )
    reps = bootstrap(triangle, n=2, seed=SEED)

    assert len(reps) == 2
    assert all(len(rep.slices) == 2 for rep in reps)


def test_maximum_entropy_algorithm():
    """Testing the algorithm deterministically here to match
    other immplementations.

    The 'x' values are taken from the paper, Vinod (2006),
    and also appear in the the meboot R package JOSS paper (2009)
    and package vignette Vinod (2023).

    The `U` values are taken from Vinod (2009) and Vinod (2023).

    Note, the values of the replicate ensemble in the
    package paper (Table 1) and the vignette (Table 1)
    do not match. Our implementation matches the results
    in Vinod (2023). The most probable reason is that the
    package paper used slightly different random uniform
    variates to generate the sequence than is reported.

    The existing Python implementations of the meboot function
    algorithm also don't match the package's implementation.
    The Python implementations appear to miss a final sorting
    of the quantiles, resulting in values that don't match the
    rank ordering of the original values.

        * Vinod (2009):
            https://cran.r-project.org/web/packages/meboot/vignettes/meboot.pdf
        * Vinod (2023):
            https://cran.r-project.org/web/packages/meboot/vignettes/Toy_Example_Exposition.pdf
    """
    # x values reported in Vinod (2009, 2023)
    x = [4, 12, 36, 20, 8]
    # Random uniform draws reported in Vinod (2009, 2023)
    U = [0.12, 0.83, 0.53, 0.59, 0.11]
    ensemble = maximum_entropy_ensemble(x, U)

    # Values reported in Vinod (2023)
    assert all(np.round(ensemble, 2) == [5.85, 13.90, 23.95, 15.70, 6.70])
    # Sequence ranks should be the same as the data
    assert all(np.argsort(ensemble) == np.argsort(x))
