"""Active COA value-set and combination-rule validation for Segment 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class COAValidator:
    """Validate candidate combinations against Finance-provided artifacts.

    The files are intentionally simple JSON so Finance can version them with
    the model artifact. A missing artifact means validation is unavailable, not
    that a suggestion is valid.
    """

    coa_version: str
    segment_values: frozenset[str]
    combinations: tuple[Mapping[str, str], ...] = ()

    @classmethod
    def from_files(
        cls,
        values_path: Path | None,
        combinations_path: Path | None = None,
        *,
        coa_version: str = "",
    ) -> "COAValidator | None":
        if values_path is None or not values_path.is_file():
            return None
        values = json.loads(values_path.read_text(encoding="utf-8"))
        if isinstance(values, Mapping):
            raw_values = values.get("segment3_values") or values.get("values") or []
            version = str(values.get("coa_version") or coa_version)
        else:
            raw_values = values
            version = coa_version
        if not isinstance(raw_values, (list, tuple, set)):
            raise ValueError("COA value-set artifact must contain a list of Segment 3 values")
        combinations: tuple[Mapping[str, str], ...] = ()
        if combinations_path is not None and combinations_path.is_file():
            raw_rules = json.loads(combinations_path.read_text(encoding="utf-8"))
            if isinstance(raw_rules, Mapping):
                raw_rules = raw_rules.get("combinations") or raw_rules.get("rules") or []
            combinations = tuple(dict(rule) for rule in raw_rules or [] if isinstance(rule, Mapping))
        return cls(
            coa_version=version,
            segment_values=frozenset(str(value).strip() for value in raw_values if str(value).strip()),
            combinations=combinations,
        )

    def validate(self, mapping: Mapping[str, Any]) -> dict[str, Any]:
        segment3 = str(mapping.get("segment3") or "").strip()
        if not segment3 or segment3 not in self.segment_values:
            return {
                "status": "invalid",
                "reason": "segment3_not_in_active_coa",
                "coa_version": self.coa_version,
            }
        if self.combinations:
            matches = [
                rule
                for rule in self.combinations
                if all(value in (None, "", "*") or str(mapping.get(key) or "") == str(value) for key, value in rule.items())
            ]
            if not matches:
                return {
                    "status": "invalid",
                    "reason": "combination_rule_rejected",
                    "coa_version": self.coa_version,
                }
        return {"status": "valid", "coa_version": self.coa_version}


def validate_mapping_with_optional_artifact(
    mapping: Mapping[str, Any], validator: COAValidator | None
) -> dict[str, Any]:
    if validator is None:
        return {"status": "unavailable", "reason": "active_coa_artifact_not_configured"}
    return validator.validate(mapping)
