from types import SimpleNamespace

import pytest

from response_parser import parse_account_type_response


def response(account_type="Supplies"):
    return SimpleNamespace(
        output_text=(
            '{"line_description":"paper","account_type":"%s",'
            '"inferred_account_type":null,"reason":"office consumable",'
            '"confidence":0.91}' % account_type
        )
    )


def test_parser_enforces_allowed_taxonomy():
    parsed = parse_account_type_response(response(), "paper", ["Supplies"])
    assert parsed["account_type"] == "Supplies"
    assert parsed["line_description"] == "paper"


def test_parser_rejects_unknown_account_type():
    with pytest.raises(ValueError, match="allowed taxonomy"):
        parse_account_type_response(response("Airfare"), "paper", ["Supplies"])


def test_parser_allows_unknown_without_inferred_type():
    parsed = parse_account_type_response(response("Unknown"), "paper", ["Supplies"])
    assert parsed["account_type"] == "Unknown"
    assert parsed["inferred_account_type"] is None


def test_parser_reads_native_oci_chat_content_list():
    native = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=[SimpleNamespace(text=response().output_text)])
            )
        ]
    )
    parsed = parse_account_type_response(native, "paper", ["Supplies"])
    assert parsed["account_type"] == "Supplies"
