"""Constrained candidate ranking with optional, versioned probability calibration."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .segment3_taxonomy import ACCOUNT_TYPE_TO_SEGMENT3, normalize_account_type
except ImportError:  # Supports direct execution from this folder.
    from segment3_taxonomy import ACCOUNT_TYPE_TO_SEGMENT3, normalize_account_type


@dataclass(frozen=True)
class RankerCalibrationArtifact:
    """Metadata and learned coefficients for a Segment 3 probability artifact."""

    ranker_version: str = "unversioned"
    feature_schema_version: str = "segment3-features-v1"
    coa_version: str = ""
    dataset_version: str = ""
    calibration_version: str = ""
    intercept: float = 0.0
    retrieval_weight: float = 0.0
    feature_weights: dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0
    sample_count: int = 0
    accepted_precision_lower_bound: float | None = None
    finance_approved: bool = False
    model_route: str = ""
    random_seed: int = 42
    calibrated: bool = True
    auto_threshold: float = 0.90
    artifact_sha256: str = field(default="", compare=False)

    def is_compatible(self, *, coa_version: str) -> bool:
        """Require exact COA version match before displaying a probability."""
        return bool(self.coa_version) and self.coa_version == str(coa_version or "")

    def is_runtime_compatible(
        self,
        *,
        coa_version: str,
        feature_schema_version: str | None = None,
        dataset_version: str | None = None,
        calibration_version: str | None = None,
        model_route: str | None = None,
    ) -> bool:
        """Validate all supplied runtime compatibility dimensions.

        Older POC callers only supplied a COA version, so ``is_compatible`` is
        retained for backwards compatibility.  Production callers should use
        this method and pass every version they know.
        """
        checks = (
            (coa_version, self.coa_version),
            (feature_schema_version, self.feature_schema_version),
            (dataset_version, self.dataset_version),
            (calibration_version, self.calibration_version),
            (model_route, self.model_route),
        )
        return all(expected is None or not str(expected) or str(expected) == str(actual) for expected, actual in checks)

    def is_strictly_runtime_compatible(
        self,
        *,
        coa_version: str,
        feature_schema_version: str,
        dataset_version: str,
        calibration_version: str,
        model_route: str,
    ) -> bool:
        """Require every manifest dimension before exposing live probability."""
        required = (coa_version, feature_schema_version, dataset_version, calibration_version, model_route)
        artifact_values = (
            self.coa_version,
            self.feature_schema_version,
            self.dataset_version,
            self.calibration_version,
            self.model_route,
        )
        return all(str(expected).strip() and str(expected) == str(actual) for expected, actual in zip(required, artifact_values))

    def probability(self, retrieval_score: float) -> float:
        """Return learned logistic probability for a normalized ranker score."""
        logit = self.intercept + self.retrieval_weight * retrieval_score
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))

    def probability_from_features(self, features: Mapping[str, float]) -> float:
        """Score a candidate using the learned feature schema.

        ``retrieval_score`` remains a supported fallback for artifacts created
        by the original POC.  New artifacts include the full feature vector and
        temperature learned from Finance-labelled calibration data.
        """
        if not self.feature_weights:
            return self.probability(float(features.get("retrieval_score", 0.0)))
        logit = self.intercept + sum(
            float(weight) * float(features.get(name, 0.0))
            for name, weight in self.feature_weights.items()
        )
        temperature = max(1e-6, float(self.temperature or 1.0))
        logit /= temperature
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))

    def release_ready(self, *, minimum_samples: int = 200, target_precision: float = 0.98) -> bool:
        """Return whether the artifact may participate in a future auto gate."""
        return bool(
            self.finance_approved
            and self.sample_count >= minimum_samples
            and self.accepted_precision_lower_bound is not None
            and self.accepted_precision_lower_bound >= target_precision
        )

    def to_dict(self) -> dict[str, str | float]:
        """Serialize every compatibility field with learned coefficients."""
        return {
            "ranker_version": self.ranker_version,
            "feature_schema_version": self.feature_schema_version,
            "coa_version": self.coa_version,
            "dataset_version": self.dataset_version,
            "calibration_version": self.calibration_version,
            "intercept": self.intercept,
            "retrieval_weight": self.retrieval_weight,
            "feature_weights": dict(self.feature_weights),
            "temperature": self.temperature,
            "sample_count": self.sample_count,
            "accepted_precision_lower_bound": self.accepted_precision_lower_bound,
            "finance_approved": self.finance_approved,
            "model_route": self.model_route,
            "random_seed": self.random_seed,
            "calibrated": self.calibrated,
            "auto_threshold": self.auto_threshold,
            "artifact_sha256": self.artifact_sha256,
        }

    def save(self, path: Path) -> None:
        """Persist a portable, reviewable calibration artifact."""
        payload = self.to_dict()
        payload["artifact_sha256"] = ""
        digest = __import__("hashlib").sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload["artifact_sha256"] = digest
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RankerCalibrationArtifact":
        """Load an artifact only when all compatibility metadata is present."""
        required = {"ranker_version", "feature_schema_version", "coa_version", "dataset_version", "calibration_version"}
        missing = sorted(field for field in required if not str(values.get(field) or "").strip())
        if missing:
            raise ValueError(f"Ranker calibration artifact missing metadata: {missing}")
        artifact = cls(
            ranker_version=str(values["ranker_version"]),
            feature_schema_version=str(values["feature_schema_version"]),
            coa_version=str(values["coa_version"]),
            dataset_version=str(values["dataset_version"]),
            calibration_version=str(values["calibration_version"]),
            intercept=_score(values.get("intercept")),
            retrieval_weight=_score(values.get("retrieval_weight")),
            feature_weights={
                str(name): _score(weight)
                for name, weight in (values.get("feature_weights") or {}).items()
            },
            temperature=max(1e-6, _score(values.get("temperature", 1.0)) or 1.0),
            sample_count=max(0, int(values.get("sample_count") or 0)),
            accepted_precision_lower_bound=(
                None
                if values.get("accepted_precision_lower_bound") is None
                else _score(values.get("accepted_precision_lower_bound"))
            ),
            finance_approved=(
                str(values.get("finance_approved")).strip().upper() in {"Y", "YES", "TRUE", "1"}
                if isinstance(values.get("finance_approved"), str)
                else bool(values.get("finance_approved", False))
            ),
            model_route=str(values.get("model_route") or ""),
            random_seed=int(values.get("random_seed") or 42),
            calibrated=(
                str(values.get("calibrated")).strip().upper() in {"Y", "YES", "TRUE", "1"}
                if isinstance(values.get("calibrated"), str)
                else bool(values.get("calibrated", True))
            ),
            auto_threshold=max(0.0, min(1.0, _score(values.get("auto_threshold", 0.90)) or 0.90)),
            artifact_sha256=str(values.get("artifact_sha256") or ""),
        )
        recorded_hash = artifact.artifact_sha256
        if recorded_hash:
            payload = dict(values)
            payload["artifact_sha256"] = ""
            expected_hash = __import__("hashlib").sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if recorded_hash != expected_hash:
                raise ValueError("Ranker calibration artifact hash mismatch")
        return artifact


def load_ranker_calibration(path: Path | None) -> RankerCalibrationArtifact | None:
    """Load a ranker artifact without treating a missing path as calibration."""
    if path is None or not path.is_file():
        return None
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read ranker calibration artifact: {path}") from exc
    if not isinstance(values, Mapping):
        raise ValueError("Ranker calibration artifact must be a JSON object")
    return RankerCalibrationArtifact.from_dict(values)


def _score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def rank_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    artifact: RankerCalibrationArtifact | None,
    coa_version: str,
    limit: int = 3,
    feature_schema_version: str | None = None,
    dataset_version: str | None = None,
    calibration_version: str | None = None,
    model_route: str | None = None,
    taxonomy: Mapping[str, str] | None = None,
    require_strict_compatibility: bool = False,
) -> dict[str, Any]:
    """Rank valid taxonomy candidates and expose probability only when compatible."""
    if limit < 1:
        raise ValueError("limit must be positive")
    active_taxonomy = taxonomy or ACCOUNT_TYPE_TO_SEGMENT3
    best_by_type: dict[str, tuple[float, Mapping[str, Any]]] = {}
    for raw in candidates:
        account_type = normalize_account_type(str(raw.get("account_type") or ""))
        if account_type not in active_taxonomy:
            continue
        score = _score(raw.get("retrieval_score", raw.get("rrf_score")))
        current = best_by_type.get(account_type)
        if current is None or score > current[0]:
            best_by_type[account_type] = (score, raw)
    if require_strict_compatibility:
        calibrated = bool(
            artifact is not None
            and artifact.calibrated
            and artifact.is_strictly_runtime_compatible(
                coa_version=coa_version,
                feature_schema_version=str(feature_schema_version or ""),
                dataset_version=str(dataset_version or ""),
                calibration_version=str(calibration_version or ""),
                model_route=str(model_route or ""),
            )
        )
    else:
        calibrated = artifact is not None and artifact.calibrated and artifact.is_runtime_compatible(
            coa_version=coa_version,
            feature_schema_version=feature_schema_version,
            dataset_version=dataset_version,
            calibration_version=calibration_version,
            model_route=model_route,
        )
    ranked = [
        {
            "rank": index,
            "account_type": account_type,
            "segment3": active_taxonomy[account_type],
            "retrieval_score": retrieval_score,
            "dense_rank": raw.get("dense_rank"),
            "sparse_rank": raw.get("sparse_rank"),
            "evidence": "hybrid_retrieval",
            "features": _candidate_features(raw, retrieval_score),
            "calibrated_probability": (
                artifact.probability_from_features(_candidate_features(raw, retrieval_score))
                if calibrated and artifact
                else None
            ),
        }
        for index, (account_type, (retrieval_score, raw)) in enumerate(sorted(
            best_by_type.items(), key=lambda item: (-item[1][0], item[0])
        )[:limit], start=1)
    ]
    return {
        "candidates": ranked,
        "selected": ranked[0] if ranked else None,
        "confidence_status": "calibrated" if calibrated and ranked else "unavailable",
        "artifact": None
        if artifact is None
        else {
            "ranker_version": artifact.ranker_version,
            "feature_schema_version": artifact.feature_schema_version,
            "coa_version": artifact.coa_version,
            "dataset_version": artifact.dataset_version,
            "calibration_version": artifact.calibration_version,
            "model_route": artifact.model_route,
            "artifact_sha256": artifact.artifact_sha256,
            "sample_count": artifact.sample_count,
            "finance_approved": artifact.finance_approved,
        },
    }


FEATURE_NAMES = (
    "retrieval_score",
    "vendor_match",
    "line_type_match",
    "amount_similarity",
    "lexical_score",
    "dense_score",
    "ambiguity_penalty",
)


def _candidate_features(raw: Mapping[str, Any], retrieval_score: float) -> dict[str, float]:
    """Normalize optional feature values supplied by retrieval/ranker callers."""
    values = {name: _score(raw.get(name)) for name in FEATURE_NAMES}
    values["retrieval_score"] = retrieval_score
    return values


def fit_ranker_artifact(
    candidates_by_row: Sequence[Sequence[Mapping[str, Any]]],
    gold_account_types: Sequence[str],
    *,
    ranker_version: str,
    feature_schema_version: str,
    coa_version: str,
    dataset_version: str,
    calibration_version: str,
    model_route: str = "segment3_ranker",
    epochs: int = 300,
    learning_rate: float = 0.08,
    random_seed: int = 42,
    taxonomy: Mapping[str, str] | None = None,
    is_synthetic: Sequence[bool] | None = None,
) -> RankerCalibrationArtifact:
    """Fit a deterministic one-vs-rest candidate correctness ranker.

    Each retrieved candidate becomes a binary training example: positive when
    its account type matches the Finance-posted gold type.  This deliberately
    avoids an LLM-derived confidence score and is small enough to run in the
    repository's offline evaluation job without external ML libraries.
    """
    if len(candidates_by_row) != len(gold_account_types):
        raise ValueError("candidates_by_row and gold_account_types must have equal length")
    if is_synthetic is not None and len(is_synthetic) != len(gold_account_types):
        raise ValueError("is_synthetic must match the training rows")
    active_taxonomy = taxonomy or ACCOUNT_TYPE_TO_SEGMENT3
    examples: list[tuple[dict[str, float], float]] = []
    for index, (candidates, gold) in enumerate(zip(candidates_by_row, gold_account_types)):
        if is_synthetic is not None and bool(is_synthetic[index]):
            continue
        normalized_gold = normalize_account_type(str(gold or ""))
        for candidate in candidates:
            account_type = normalize_account_type(str(candidate.get("account_type") or ""))
            if account_type not in active_taxonomy:
                continue
            retrieval_score = _score(candidate.get("retrieval_score", candidate.get("rrf_score")))
            examples.append((_candidate_features(candidate, retrieval_score), float(account_type == normalized_gold)))
    if not examples:
        raise ValueError("At least one valid retrieved candidate is required to fit the ranker")

    weights = {name: 0.0 for name in FEATURE_NAMES}
    bias = 0.0
    for _ in range(max(1, int(epochs))):
        gradient = {name: 0.0 for name in FEATURE_NAMES}
        bias_gradient = 0.0
        for features, target in examples:
            logit = bias + sum(weights[name] * features[name] for name in FEATURE_NAMES)
            probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
            error = probability - target
            bias_gradient += error
            for name in FEATURE_NAMES:
                gradient[name] += error * features[name]
        scale = 1.0 / len(examples)
        bias -= learning_rate * bias_gradient * scale
        for name in FEATURE_NAMES:
            weights[name] -= learning_rate * gradient[name] * scale

    return RankerCalibrationArtifact(
        ranker_version=ranker_version,
        feature_schema_version=feature_schema_version,
        coa_version=coa_version,
        dataset_version=dataset_version,
        calibration_version=calibration_version,
        intercept=bias,
        retrieval_weight=weights["retrieval_score"],
        feature_weights=weights,
        sample_count=len(examples),
        model_route=model_route,
        random_seed=random_seed,
        calibrated=False,
    )


def calibrate_ranker_artifact(
    artifact: RankerCalibrationArtifact,
    candidates_by_row: Sequence[Sequence[Mapping[str, Any]]],
    gold_account_types: Sequence[str],
    *,
    calibration_version: str | None = None,
    epochs: int = 250,
    learning_rate: float = 0.05,
    taxonomy: Mapping[str, str] | None = None,
    is_synthetic: Sequence[bool] | None = None,
) -> RankerCalibrationArtifact:
    """Fit an intercept/temperature calibration layer on held-out real rows."""
    if len(candidates_by_row) != len(gold_account_types):
        raise ValueError("candidates_by_row and gold_account_types must have equal length")
    if is_synthetic is not None and len(is_synthetic) != len(gold_account_types):
        raise ValueError("is_synthetic must match the calibration rows")
    active_taxonomy = taxonomy or ACCOUNT_TYPE_TO_SEGMENT3
    observations: list[tuple[float, float]] = []
    for index, (candidates, gold) in enumerate(zip(candidates_by_row, gold_account_types)):
        if is_synthetic is not None and bool(is_synthetic[index]):
            raise ValueError("Synthetic rows cannot be used for ranker calibration")
        gold_type = normalize_account_type(str(gold or ""))
        valid = [
            candidate
            for candidate in candidates
            if normalize_account_type(str(candidate.get("account_type") or "")) in active_taxonomy
        ]
        if not valid:
            continue
        selected = max(valid, key=lambda candidate: _score(candidate.get("retrieval_score", candidate.get("rrf_score"))))
        features = _candidate_features(
            selected,
            _score(selected.get("retrieval_score", selected.get("rrf_score"))),
        )
        raw_probability = artifact.probability_from_features(features)
        clipped_probability = max(1e-6, min(1.0 - 1e-6, raw_probability))
        raw_logit = math.log(clipped_probability / (1.0 - clipped_probability))
        observations.append((raw_logit, float(normalize_account_type(str(selected.get("account_type") or "")) == gold_type)))
    if not observations:
        raise ValueError("At least one held-out real row is required for calibration")
    intercept = 0.0
    temperature = 1.0
    for _ in range(max(1, int(epochs))):
        grad_intercept = 0.0
        grad_temperature = 0.0
        for raw_logit, target in observations:
            scaled = (raw_logit + intercept) / max(1e-6, temperature)
            probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, scaled))))
            error = probability - target
            grad_intercept += error / max(1e-6, temperature)
            grad_temperature += error * -(raw_logit + intercept) / max(1e-6, temperature) ** 2
        scale = 1.0 / len(observations)
        intercept -= learning_rate * grad_intercept * scale
        temperature = max(0.05, temperature - learning_rate * grad_temperature * scale)
    correct = sum(int(target) for _, target in observations)
    count = len(observations)
    proportion = correct / count
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) / count) + z * z / (4.0 * count * count)) / denominator
    lower_bound = max(0.0, centre - margin)
    calibrated = RankerCalibrationArtifact(
        **{
            **artifact.__dict__,
            "intercept": artifact.intercept + intercept,
            "temperature": temperature,
            "sample_count": len(observations),
            "accepted_precision_lower_bound": lower_bound,
            "calibration_version": calibration_version or artifact.calibration_version,
            "calibrated": True,
        }
    )
    return calibrated
