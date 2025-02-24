import copy
import datetime
import math
from dataclasses import dataclass, field
from typing import Union, get_args

import numpy as np

__all__ = [
    "Metadata",
    "MetadataValue",
    "common_metadata",
    "metadata_diff",
]


METADATA_COMMON_ATTRIBUTES = [
    "risk_basis",
    "country",
    "currency",
    "reinsurance_basis",
    "loss_definition",
    "per_occurrence_limit",
    "details",
    "loss_details",
]


MetadataValue = Union[str, float, int, bool, datetime.date, np.ndarray, None]


@dataclass(frozen=True, eq=True)
class Metadata(object):
    # noinspection PyUnresolvedReferences
    """Metadata about the information stored in `Cell`s.

    Attributes:
        risk_basis: The basis on which risks are assigned to experience periods.
            Common values are "Accident" and "Policy".
        country: The country where the risks are domiciled, as an ISO 3166 alpha-2 code
            (e.g., "US", "ES", "DE").
        currency: The currency in which premium and losses are denominated, as an ISO 4217
            code (e.g., "USD", "GBP", "JPY").
        reinsurance_basis: Whether the losses are net of reinsurance.
            Typical values are "Gross" and "Net"; the field is typed as a `str` and not a
            `bool` to allow for more nuanced gradations of value.
        loss_definition: The portions of loss costs and expenses that are included in
            loss amounts. Typical values are "Loss", "Loss+DCC", and "Loss+LAE".
        per_occurrence_limit: The per-occurrence limit on individual losses.
        details: Other information about the data represented in the `Cell`. For example,
            `details` could plausibly include coverage, rated state, or product code.
        loss_details: Information about the data in the `Cell` that only pertains to the losses.
            For example, you may have a triangle where losses are segmented by cause of loss,
            but premium is not.
    """

    risk_basis: str = "Accident"
    country: str | None = None
    currency: str | None = None
    reinsurance_basis: str | None = None
    loss_definition: str | None = None
    per_occurrence_limit: float | None = None
    details: dict[str, MetadataValue] = field(default_factory=dict)
    loss_details: dict[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self):
        # Perform validations on string attributes
        for attr in [
            "risk_basis",
            "country",
            "currency",
            "reinsurance_basis",
            "loss_definition",
        ]:
            if not isinstance(getattr(self, attr), (str, type(None))):
                raise TypeError(f"{attr} should be a string.")

        # Perform validation on per_occurrence_limit
        if not isinstance(self.per_occurrence_limit, (int, float, type(None))):
            raise TypeError("per_occurrence_limit should be a number.")

        # Perform validations on details and loss_details objects
        for attr in ["details", "loss_details"]:
            test_attr = getattr(self, attr)

            # Test type
            if not isinstance(test_attr, dict):
                raise TypeError("{attr} should be of type dict.")

            # Test dict structure
            for k, v in test_attr.items():
                if not isinstance(k, str):
                    raise TypeError(f"{attr} has non-string dict key.")
                if not isinstance(v, get_args(MetadataValue)):
                    raise TypeError(f"{attr} has dict value of invalid type.")

    def as_dict(self):
        return {
            "currency": self.currency,
            "country": self.country,
            "risk_basis": self.risk_basis,
            "reinsurance_basis": self.reinsurance_basis,
            "loss_definition": self.loss_definition,
            "per_occurrence_limit": self.per_occurrence_limit,
            "details": copy.deepcopy(self.details),
            "loss_details": copy.deepcopy(self.loss_details),
        }

    def as_flat_dict(self):
        """All metadata associated with the cell (currency, country, details, etc.)"""
        return {
            "currency": self.currency,
            "country": self.country,
            "risk_basis": self.risk_basis,
            "reinsurance_basis": self.reinsurance_basis,
            "loss_definition": self.loss_definition,
            "per_occurrence_limit": self.per_occurrence_limit,
            **self.details,
            **self.loss_details,
        }

    def __hash__(self):
        return hash(
            (
                self.risk_basis,
                self.country,
                self.currency,
                self.reinsurance_basis,
                self.loss_definition,
                self.per_occurrence_limit,
                frozenset(self.details.items()),
                frozenset(self.loss_details.items()),
            )
        )

    def __lt__(self, other):
        """Required so metadata can be sorted. This allows cells sharing metadata to be grouped
        together when sorting lists of cells.
        Comparison is based on fields in the canonical order
        """
        # noinspection DuplicatedCode
        return (
            "" if self.risk_basis is None else self.risk_basis,
            "" if self.country is None else self.country,
            "" if self.currency is None else self.currency,
            "" if self.reinsurance_basis is None else self.reinsurance_basis,
            "" if self.loss_definition is None else self.loss_definition,
            (
                math.inf
                if self.per_occurrence_limit is None
                else self.per_occurrence_limit
            ),
            tuple(sorted(self.details.items())),
            tuple(sorted(self.loss_details.items())),
        ) < (
            "" if other.risk_basis is None else other.risk_basis,
            "" if other.country is None else other.country,
            "" if other.currency is None else other.currency,
            "" if other.reinsurance_basis is None else other.reinsurance_basis,
            "" if other.loss_definition is None else other.loss_definition,
            (
                math.inf
                if other.per_occurrence_limit is None
                else other.per_occurrence_limit
            ),
            tuple(sorted(other.details.items())),
            tuple(sorted(self.loss_details.items())),
        )


def common_metadata(meta1: Metadata, meta2: Metadata) -> Metadata:
    """Return a Metadata object that has all elements that are identical in meta1 and meta2."""
    risk_basis = _first_if_equal(meta1.risk_basis, meta2.risk_basis)
    country = _first_if_equal(meta1.country, meta2.country)
    currency = _first_if_equal(meta1.currency, meta2.currency)
    reinsurance_basis = _first_if_equal(
        meta1.reinsurance_basis, meta2.reinsurance_basis
    )
    loss_definition = _first_if_equal(meta1.loss_definition, meta2.loss_definition)
    per_occurrence_limit = _first_if_equal(
        meta1.per_occurrence_limit, meta2.per_occurrence_limit
    )

    common_detail_keys = set(meta1.details.keys()) & set(meta2.details.keys())
    details = {
        k: meta1.details[k]
        for k in common_detail_keys
        if meta1.details[k] == meta2.details[k]
    }
    common_loss_detail_keys = set(meta1.loss_details.keys()) & set(
        meta2.loss_details.keys()
    )
    loss_details = {
        k: meta2.loss_details[k]
        for k in common_loss_detail_keys
        if meta1.loss_details[k] == meta2.loss_details[k]
    }

    return Metadata(
        risk_basis=risk_basis,
        country=country,
        currency=currency,
        reinsurance_basis=reinsurance_basis,
        loss_definition=loss_definition,
        per_occurrence_limit=per_occurrence_limit,
        details=details,
        loss_details=loss_details,
    )


def _first_if_equal(arg1, arg2):
    """Return arg1 if arg1 == arg2, otherise None."""
    if arg1 == arg2:
        return arg1
    else:
        return None


def metadata_diff(meta_core: Metadata, meta_diff: Metadata) -> Metadata:
    """Return a Metadata object with all of the items in `meta_diff` that are not in `meta_core`."""

    risk_basis = meta_diff.risk_basis if meta_core.risk_basis is None else None
    country = meta_diff.country if meta_core.country is None else None
    currency = meta_diff.currency if meta_core.currency is None else None
    reinsurance_basis = (
        meta_diff.reinsurance_basis if meta_core.reinsurance_basis is None else None
    )
    loss_definition = (
        meta_diff.loss_definition if meta_core.loss_definition is None else None
    )
    per_occurrence_limit = (
        meta_diff.per_occurrence_limit
        if meta_core.per_occurrence_limit is None
        else None
    )
    details = {k: v for k, v in meta_diff.details.items() if k not in meta_core.details}
    loss_details = {
        k: v
        for k, v in meta_diff.loss_details.items()
        if k not in meta_core.loss_details
    }

    return Metadata(
        risk_basis=risk_basis,
        country=country,
        currency=currency,
        reinsurance_basis=reinsurance_basis,
        loss_definition=loss_definition,
        per_occurrence_limit=per_occurrence_limit,
        details=details,
        loss_details=loss_details,
    )
