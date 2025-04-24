import datetime

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
    
