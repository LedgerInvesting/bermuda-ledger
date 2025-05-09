import datetime
import numpy as np

from bermuda import Triangle, meyers_tri, Metadata

def test_plot_data_completeness():
    test = meyers_tri.replace(
        values = lambda cell: (
            cell.values if cell.period_start < datetime.date(1990, 1, 1) 
            else {"paid_loss": cell["paid_loss"], "earned_premium": cell["earned_premium"]} if cell.period_start < datetime.date(1994, 1, 1)
            else {"reported_loss": cell["reported_loss"]}
        ),
        metadata=Metadata(details={"id": 1})
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test2.plot_data_completeness().show()
    (test + test2).plot_data_completeness().show()
    (test + test2 + test3 + test4 + test5).plot_data_completeness().show()

def test_plot_data_completeness_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_data_completeness().show()

    
def test_plot_right_edge():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_right_edge().show()
    (test + test2).plot_right_edge().show()
    (test + test2 + test3 + test4 + test5).plot_right_edge().show()


def test_plot_right_edge_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_right_edge(show_uncertainty=True, uncertainty_type="ribbon").show()
    test_predictions.plot_right_edge(show_uncertainty=True, uncertainty_type="segments").show()

    
def test_plot_heatmap():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss = lambda cell: cell["reported_loss"],
        reported_claims = lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_heatmap({"Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"], "Reported PR": lambda cell: cell["reported_loss"] / cell["earned_premium"], "Earned Premium": lambda cell: cell["earned_premium"] / 1e6, "Incurred LR": lambda cell: cell["incurred_loss"] / cell["earned_premium"], "Reported Claims": lambda cell: cell["reported_claims"] / 1e6}).show()
    (test + test2 + test3).plot_heatmap({"Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"], "Reported PR": lambda cell: cell["reported_loss"] / cell["earned_premium"], "Earned Premium": lambda cell: cell["earned_premium"] / 1e6}, ncols=3).show()
    (test + test2).plot_heatmap().show()
    (test + test2 + test3 + test4 + test5).plot_heatmap().show()
    (test + test2 + test3 + test4 + test5).plot_heatmap({"Earned Premium": lambda cell: cell["earned_premium"] / 1e6}).show()

def test_plot_heatmap_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_heatmap().show()
    test_predictions.plot_heatmap({"Reported LR": lambda cell: 100 * cell["reported_loss"] / cell["earned_premium"]}).show()

def test_plot_growth_curve():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss = lambda cell: cell["reported_loss"],
        reported_claims = lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_growth_curve().show()
    test.plot_growth_curve({"Paid LR": lambda cell: 100 * cell["paid_loss"] / cell["earned_premium"], "Reported LR": lambda cell: 100 * cell["reported_loss"] / cell["earned_premium"]}).show()


def test_plot_growth_curve_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_growth_curve().show()
    test_predictions.plot_growth_curve(uncertainty_type = "segments").show()
    test_predictions.plot_growth_curve({"Reported LR": lambda cell: 100 * cell["reported_loss"] / cell["earned_premium"]}).show()


def test_plot_mountain():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss = lambda cell: cell["reported_loss"],
        reported_claims = lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_mountain().show()
    (test + test2 + test3 + test4 + test5).plot_mountain({"Paid LR": lambda cell: 100 * cell["paid_loss"] / cell["earned_premium"], "Reported LR": lambda cell: 100 * cell["reported_loss"] / cell["earned_premium"]}).show()
    test.plot_mountain({"Paid LR": lambda cell: 100 * cell["paid_loss"] / cell["earned_premium"], "Reported LR": lambda cell: 100 * cell["reported_loss"] / cell["earned_premium"]}).show()
    test.plot_mountain({"Paid/Reported LR": lambda cell: 100 * (cell["paid_loss"] / cell["earned_premium"]) / (cell["reported_loss"] / cell["earned_premium"])}).show()


def test_plot_mountain_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.dev_lag() < 108 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.dev_lag() < 108 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_mountain().show()
    test_predictions.plot_mountain(uncertainty_type = "segments").show()
    test_predictions.plot_mountain({"Reported LR": lambda cell: 100 * cell["reported_loss"] / cell["earned_premium"]}).show()

def test_plot_ballistic():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss = lambda cell: cell["reported_loss"],
        reported_claims = lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_ballistic().show()
    (test + test2 + test3 + test4 + test5).plot_ballistic(ncols=2, width=500, height=300).show()


def test_plot_ballistic_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_ballistic().show()

def test_plot_broom():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss = lambda cell: cell["reported_loss"],
        reported_claims = lambda cell: cell["reported_loss"],
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_broom().show()
    (test + test2 + test3 + test4 + test5).plot_broom(ncols=2, width=500, height=300).show()


def test_plot_broom_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_broom().show()

def test_plot_drip():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss = lambda cell: cell["reported_loss"],
        reported_claims = 1000,
        open_claims = lambda cell: 1000 * np.exp(-0.1 * cell.dev_lag()),
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_drip().show()
    (test + test2 + test3 + test4 + test5).plot_drip().show()


def test_plot_drip_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8,
        reported_claims = 1000,
        open_claims = lambda cell: 1000 * np.exp(-0.1 * cell.dev_lag()),
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_drip().show()

def test_plot_hose():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        incurred_loss = lambda cell: cell["reported_loss"],
        reported_claims = 1000,
        open_claims = lambda cell: 1000 * np.exp(-0.1 * cell.dev_lag()),
    )
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_hose().show()
    (test + test2 + test3 + test4 + test5).plot_hose().show()

def test_plot_sunset():
    test = meyers_tri.derive_metadata(id=1)
    test2 = test.derive_metadata(id=2)
    test3 = test.derive_metadata(id=3)
    test4 = test.derive_metadata(id=4)
    test5 = test.derive_metadata(id=5)
    test.plot_sunset().show()
    (test + test2 + test3 + test4 + test5).plot_sunset().show()

def test_plot_sunset_with_predictions():
    test = meyers_tri.derive_metadata(id=1).derive_fields(
        paid_loss = lambda cell: cell["paid_loss"] * 0.8
    )
    test_predictions = test.derive_fields(
        reported_loss = lambda cell: cell["reported_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["reported_loss"], 1e5, 10_000),
        paid_loss = lambda cell: cell["paid_loss"] if cell.period_start.year < 1995 else np.random.normal(cell["paid_loss"], 1e5, 10_000),
    )

    test_predictions.plot_sunset().show()
    test_predictions.plot_sunset(uncertainty_type = "segments").show()

