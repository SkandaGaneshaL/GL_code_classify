import json

from coa_validation import COAValidator, validate_mapping_with_optional_artifact


def test_coa_validator_rejects_unknown_segment3(tmp_path):
    values = tmp_path / "coa.json"
    values.write_text(json.dumps({"coa_version": "coa-v2", "segment3_values": ["60520"]}), encoding="utf-8")
    validator = COAValidator.from_files(values, coa_version="coa-v2")
    assert validator is not None
    result = validator.validate({"segment1": "101", "segment2": "10", "segment3": "99999"})
    assert result["status"] == "invalid"


def test_missing_coa_artifact_is_unavailable_not_valid():
    result = validate_mapping_with_optional_artifact({"segment3": "60520"}, None)
    assert result["status"] == "unavailable"
