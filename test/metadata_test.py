import pytest

from bermuda.base import Metadata, common_metadata, metadata_diff

ay = Metadata()
py = Metadata(risk_basis="Policy")


def test_metadata_diff():
    # No difference because risk_basis is in both
    basis_diff = metadata_diff(ay, py)

    # Add currency and it should show up in diff
    usd = Metadata(currency="USD")
    usd_diff = metadata_diff(ay, usd)

    assert not any(vars(basis_diff).values())
    assert usd_diff.currency == "USD"


def test_common_metadata():
    meta_1 = Metadata(
        country="US", details={"coverage": "BI", "line_of_business": "PA"}
    )
    meta_2 = Metadata(
        country="EU", details={"coverage": "COLL", "line_of_business": "PA"}
    )

    common = common_metadata(meta_1, meta_2)

    assert common.risk_basis == "Accident"
    assert common.country is None
    assert set(common.details.keys()) == {"line_of_business"}


# Create a bunch of examples for what the str fields shouldn't look like
non_str_types = [0, 0.0, True, [], (), {}]


@pytest.mark.parametrize("non_str_types", non_str_types)
def test_metadata_argument_constraints(non_str_types):
    with pytest.raises(TypeError):
        Metadata(risk_basis=non_str_types)
        Metadata(country=non_str_types)
        Metadata(currency=non_str_types)
        Metadata(reinsurance_basis=non_str_types)
        Metadata(loss_definition=non_str_types)


# Create a bunch of examples for what the per_occurrence field shouldn't look like
non_float_types = ["0", "0.0", True, [], (), {}]


@pytest.mark.parametrize("non_float_types", non_float_types)
def test_metadata_per_occurrence_constraint(non_float_types):
    with pytest.raises(TypeError):
        Metadata(per_occurrence=0.0)


# Create a bunch of examples for what the details and loss_details fields shouldn't look like
invalid_details_examples = [
    # Test Type
    ["Generic List"],
    ("Generic Tuple",),
    "Simple String",
    0,
    True,
    None,
    # Test dict structure
    {0: "dict with non-str key"},
    # Test MetadataValue type
    {"key": ("Tuple value",)},
    {"key": ["List value"]},
    {"key": {"dict": "value"}},
]


@pytest.mark.parametrize("invalid_details_examples", invalid_details_examples)
def test_metadata_detail_constraints(invalid_details_examples):
    with pytest.raises(TypeError):
        Metadata(
            details=invalid_details_examples, loss_details=invalid_details_examples
        )
