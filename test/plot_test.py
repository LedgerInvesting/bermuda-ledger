import datetime
import numpy as np

from bermuda import Triangle, meyers_tri, Metadata


def test_plot_data_completeness():
    test = meyers_tri.replace(
        values=lambda cell: (
            cell.values
            if cell.period_start < datetime.date(1990, 1, 1)
            else {
                "paid_loss": cell["paid_loss"],
                "earned_premium": cell["earned_premium"],
            }
            if cell.period_start < datetime.date(1994, 1, 1)
            else {"reported_loss": cell["reported_loss"]}
        ),
        metadata=Metadata(details={"id": 1}),
    )
    test2 = test.derive_metadata(id=2)
    test2.plot_data_completeness()
    (test + test2).plot_data_completeness()


def test_plot_data_completeness_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: cell["paid_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_data_completeness()


def test_plot_right_edge():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test2 = test.derive_metadata(id=2)
    test.plot_right_edge()
    (test + test2).plot_right_edge()


def test_plot_right_edge_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: cell["paid_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_right_edge(uncertainty=True, uncertainty_type="ribbon")
    test_predictions.plot_right_edge(uncertainty=True, uncertainty_type="segments")


def test_plot_heatmap():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    (test + test2).plot_heatmap(
        {
            "Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"],
            "Reported PR": lambda cell: cell["reported_loss"] / cell["earned_premium"],
        }
    )
    test.plot_heatmap(
        {
            "Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"],
            "Reported PR": lambda cell: cell["reported_loss"] / cell["earned_premium"],
            "Earned Premium": lambda cell: cell["earned_premium"] / 1e6,
            "Incurred LR": lambda cell: cell["incurred_loss"] / cell["earned_premium"],
            "Reported Claims": lambda cell: cell["reported_claims"] / 1e6,
        }
    )
    (test + test2 + test3).plot_heatmap(
        {
            "Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"],
            "Reported PR": lambda cell: cell["reported_loss"] / cell["earned_premium"],
            "Earned Premium": lambda cell: cell["earned_premium"] / 1e6,
        },
        ncols=3,
    )
    (test + test2).plot_heatmap()
    (test + test2 + test3 + test4 + test5).plot_heatmap()
    (test + test2 + test3 + test4 + test5).plot_heatmap(
        {"Earned Premium": lambda cell: cell["earned_premium"] / 1e6}
    )
    test.plot_heatmap(
        {
            "Paid ATAs": lambda cell, prev_cell: cell["paid_loss"]
            / prev_cell["paid_loss"]
        }
    )
    test.plot_heatmap(
        {
            "Reported ATAs": lambda cell, prev_cell: cell["reported_loss"]
            / prev_cell["reported_loss"]
        }
    )


def test_plot_heatmap_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: cell["paid_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_heatmap()
    test_predictions.plot_heatmap(
        {
            "Reported LR": lambda cell: 100
            * cell["reported_loss"]
            / cell["earned_premium"]
        }
    )


def test_plot_atas():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    (test + test2).plot_atas(["Paid ATA", "Reported ATA"])


def test_plot_growth_curve():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test.plot_growth_curve()
    (test + test2).plot_growth_curve()
    test.plot_growth_curve(
        {
            "Paid LR": lambda cell: 100 * cell["paid_loss"] / cell["earned_premium"],
            "Reported LR": lambda cell: 100
            * cell["reported_loss"]
            / cell["earned_premium"],
        }
    )


def test_plot_growth_curve_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: cell["paid_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_growth_curve()
    test_predictions.plot_growth_curve(uncertainty_type="segments")
    test_predictions.plot_growth_curve(uncertainty_type="spaghetti", n_lines=50)
    test_predictions.plot_growth_curve(
        {
            "Reported LR": lambda cell: 100
            * cell["reported_loss"]
            / cell["earned_premium"]
        }
    )


def test_plot_mountain():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test.plot_mountain()
    (test + test2).plot_mountain(
        {
            "Paid LR": lambda cell: 100 * cell["paid_loss"] / cell["earned_premium"],
            "Reported LR": lambda cell: 100
            * cell["reported_loss"]
            / cell["earned_premium"],
        }
    )


def test_plot_mountain_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.dev_lag() < 108
        else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: cell["paid_loss"]
        if cell.dev_lag() < 108
        else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_mountain()
    test_predictions.plot_mountain(uncertainty_type="segments")
    test_predictions.plot_mountain(
        {
            "Reported LR": lambda cell: 100
            * cell["reported_loss"]
            / cell["earned_premium"]
        }
    )


def test_plot_ballistic():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_ballistic()
    (test + test2 + test3 + test4 + test5).plot_ballistic(
        ncols=2, width=500, height=300
    )


def test_plot_ballistic_with_predictions():
    test_predictions = meyers_tri.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["reported_loss"], 1e6, 10_000),
    )

    test_predictions.plot_ballistic()


def test_plot_broom():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_broom()
    test.plot_broom(rule=None)
    (test + test2 + test3 + test4 + test5).plot_broom(ncols=2, width=500, height=300)


def test_plot_broom_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: cell["paid_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_broom()


def test_plot_drip():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=1000,
        open_claims=lambda cell: 1000 * np.exp(-0.1 * cell.dev_lag()),
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_drip()
    (test + test2 + test3 + test4 + test5).plot_drip()


def test_plot_drip_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8,
        reported_claims=1000,
        open_claims=lambda cell: 1000 * np.exp(-0.1 * cell.dev_lag()),
    )
    test_predictions = test.derive_fields(
        open_claims=lambda cell: cell["open_claims"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["open_claims"], 500, 10_000),
    )

    test_predictions.plot_drip()


def test_plot_hose():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss=lambda cell: cell["reported_loss"],
        reported_claims=1000,
        open_claims=lambda cell: 1000 * np.exp(-0.1 * cell.dev_lag()),
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_hose()
    (test + test2 + test3 + test4 + test5).plot_hose()


def test_plot_sunset():
    test = meyers_tri.derive_metadata(id=1)
    test2 = test.derive_metadata(id=2)
    test.plot_sunset()
    (test + test2).plot_sunset()


def test_plot_sunset_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss=lambda cell: cell["reported_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: cell["paid_loss"]
        if cell.period_start.year < 1995
        else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_sunset()
    test_predictions.plot_sunset(uncertainty_type="segments")


def test_plot_histogram():
    raw = meyers_tri.derive_metadata(id=1)[-2:].derive_fields(
        paid_loss=lambda cell: cell["paid_loss"] * 0.8
    )
    test = raw.derive_fields(
        reported_loss=lambda cell: np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss=lambda cell: np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )
    test2 = test.derive_metadata(id=2)
    test.plot_histogram(["Paid Loss Ratio", "Reported Loss Ratio"])
    (test + test2).plot_histogram(["Paid Loss", "Reported Loss"])
