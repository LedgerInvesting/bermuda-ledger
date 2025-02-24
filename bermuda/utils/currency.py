from dataclasses import replace

from ..base import Cell
from ..triangle import Triangle

__all__ = [
    "convert_to_dollars",
    "convert_currency",
]


# Define the list of fields that are currency-denominated and that therefore need to be
# converted when changing currencies
CURRENCY_FIELDS = [
    "earned_premium",
    "used_earned_premium",
    "written_premium",
    "paid_loss",
    "reported_loss",
    "incurred_loss",
]


# These are rough defaults and should almost always be overridden for a real analysis
DEFAULT_EXCHANGE_RATES = {
    "EUR": 1.10,
    "GBP": 1.31,
}


def convert_to_dollars(
    triangle: Triangle, exchange_rates: dict[str, float] | None = None
) -> Triangle:
    """Convert all currency-denominated fields in a Triangle to USD.

    See `convert_currency` for more details on the conversion process.

    Arguments:
        triangle: A triangle to convert to dollars.
        exchange_rates: A dictionary that maps currencies to currency-to-dollar exchange rates.
    Returns:
        A triangle with all cells denominated in dollars.
    """
    if exchange_rates is None:
        exchange_rates = DEFAULT_EXCHANGE_RATES
    return convert_currency(triangle, "USD", exchange_rates)


def convert_currency(
    triangle: Triangle, target_currency: str, exchange_rates: dict[str, float]
) -> Triangle:
    """Convert all currency-denominated fields in a Triangle to a target currency.

    Non-currency fields (e.g., claim counts, policy counts) will not be altered.

    Arguments:
        triangle: A triangle to convert to a single currency.
        target_currency: The currency that all values will be expressed in, e.g. `USD` or `EUR`.
        exchange_rates: A dictionary with exchange rates from other currencies to the target
            currency. For example, to convert a triangle with GBP-denominated losses to USD
            with an assumed exchange rate of 1.0 GBP = 1.4 USD, then `exchange_rates` would
            be `{"GBP": 1.4}`.
    Returns:
        A triangle with all cells denominated in the target currency.
    """
    converted_cells = []
    for metadata, tri_slice in triangle.slices.items():
        if metadata.currency is None:
            raise ValueError(
                "Every slice in the input triangle must have a defined currency"
            )
        elif metadata.currency == target_currency:
            converted_cells += tri_slice.cells
        elif metadata.currency not in exchange_rates:
            raise ValueError(
                f"Must supply an exchange rate for currency `{metadata.currency}`"
            )
        else:
            exchange_rate = exchange_rates[metadata.currency]
            for cell in tri_slice:
                converted_cells.append(
                    _convert_cell_currency(cell, exchange_rate, target_currency)
                )

    return Triangle(converted_cells)


def _convert_cell_currency(
    cell: Cell, exchange_rate: float, target_currency: str
) -> Cell:
    converted_values = {
        k: v * exchange_rate if k in CURRENCY_FIELDS else v
        for k, v in cell.values.items()
    }

    return cell.replace(
        values=converted_values,
        metadata=replace(cell.metadata, currency=target_currency),
    )
