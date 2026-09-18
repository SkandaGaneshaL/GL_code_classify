"""Small, dependency-free calibration layer for logprob signals."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .release_gate import evaluate_release_gate
except ImportError:
    from release_gate import evaluate_release_gate


@dataclass(frozen=True)
class LogprobCalibrator:
    """Apply a saved logistic calibration model.

    Parameters are deliberately explicit so the artifact can be produced by
    an offline Finance-approved calibration job without importing a training
    framework into the Streamlit runtime.
    """

    intercept: float = 0.0
    selected_weight: float = 1.0
    margin_weight: float = 1.0
    temperature: float = 1.0
    model: str | None = None
    endpoint_route: str | None = None
    request_mode: str | None = None
    code_map_version: str | None = None
    sample_count: int | None = None
    validation_metrics: dict[str, Any] | None = None
    finance_approved: bool = False
    artifact_version: str = "segment3-logprob-calibration-v2"
    dataset_version: str | None = None
    created_at: str | None = None
    auto_threshold: float | None = None
    selected_center: float = 0.0
    selected_scale: float = 1.0
    margin_center: float = 0.0
    margin_scale: float = 1.0

    @classmethod
    def from_json(cls, path: str | Path) -> "LogprobCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Logprob calibration artifact must be a JSON object")
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        temperature = float(data.get("temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Logprob calibration temperature must be positive")
        return cls(
            intercept=float(data.get("intercept", 0.0)),
            selected_weight=float(data.get("selected_weight", 1.0)),
            margin_weight=float(data.get("margin_weight", 1.0)),
            temperature=temperature,
            model=str(data.get("model") or data.get("model_id") or metadata.get("model") or "").strip() or None,
            endpoint_route=str(
                data.get("endpoint_route") or data.get("route") or metadata.get("endpoint_route") or ""
            ).strip() or None,
            request_mode=str(data.get("request_mode") or metadata.get("request_mode") or "").strip() or None,
            code_map_version=str(data.get("code_map_version") or metadata.get("code_map_version") or "").strip() or None,
            sample_count=(
                int(data.get("sample_count", metadata.get("sample_count")))
                if data.get("sample_count", metadata.get("sample_count")) is not None
                else None
            ),
            validation_metrics=(
                dict(data.get("validation_metrics", metadata.get("validation_metrics")))
                if isinstance(data.get("validation_metrics", metadata.get("validation_metrics")), dict)
                else None
            ),
            finance_approved=bool(data.get("finance_approved", metadata.get("finance_approved", False))),
            artifact_version=str(data.get("artifact_version") or metadata.get("artifact_version") or "segment3-logprob-calibration-v2"),
            dataset_version=str(data.get("dataset_version") or metadata.get("dataset_version") or "").strip() or None,
            created_at=str(data.get("created_at") or metadata.get("created_at") or "").strip() or None,
            auto_threshold=(
                float(data.get("auto_threshold", metadata.get("auto_threshold")))
                if data.get("auto_threshold", metadata.get("auto_threshold")) is not None
                else None
            ),
            selected_center=float(data.get("selected_center", metadata.get("selected_center", 0.0))),
            selected_scale=max(1e-12, float(data.get("selected_scale", metadata.get("selected_scale", 1.0)))),
            margin_center=float(data.get("margin_center", metadata.get("margin_center", 0.0))),
            margin_scale=max(1e-12, float(data.get("margin_scale", metadata.get("margin_scale", 1.0)))),
        )

    def is_compatible(
        self,
        *,
        model: str | None,
        route: str | None,
        request_mode: str | None,
        code_map_version: str,
    ) -> bool:
        """Require explicit artifact metadata before using calibrated evidence."""
        if not self.model or not model or self.model != model:
            return False
        if not self.endpoint_route or not route or self.endpoint_route != route:
            return False
        if self.request_mode and self.request_mode != (request_mode or ""):
            return False
        if not self.code_map_version or self.code_map_version != code_map_version:
            return False
        if self.sample_count is None or self.sample_count <= 0:
            return False
        if not self.validation_metrics:
            return False
        return True

    def score(self, selected_mean_logprob: float | None, margin: float | None) -> float | None:
        if selected_mean_logprob is None:
            return None
        selected = float(selected_mean_logprob)
        gap = float(margin or 0.0)
        selected = (selected - self.selected_center) / self.selected_scale
        gap = (gap - self.margin_center) / self.margin_scale
        value = (self.intercept + self.selected_weight * selected + self.margin_weight * gap) / self.temperature
        value = max(-60.0, min(60.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def is_auto_ready(self, *, target_precision: float, minimum_samples: int) -> bool:
        """Return whether this artifact is safe to use for automatic posting."""
        if not self.finance_approved or self.sample_count is None or self.sample_count < minimum_samples:
            return False
        if self.auto_threshold is None or not 0.0 < self.auto_threshold < 1.0:
            return False
        metrics = self.validation_metrics or {}
        accepted_real_count = metrics.get("accepted_real_count", self.sample_count)
        try:
            if int(accepted_real_count) < minimum_samples:
                return False
        except (TypeError, ValueError):
            return False
        lower_bound = metrics.get("accepted_precision_lower_bound")
        if lower_bound is None:
            lower_bound = metrics.get("auto_precision_lower_bound")
        try:
            return float(lower_bound) >= float(target_precision)
        except (TypeError, ValueError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "intercept": self.intercept,
            "selected_weight": self.selected_weight,
            "margin_weight": self.margin_weight,
            "temperature": self.temperature,
            "model": self.model,
            "endpoint_route": self.endpoint_route,
            "request_mode": self.request_mode,
            "code_map_version": self.code_map_version,
            "sample_count": self.sample_count,
            "validation_metrics": self.validation_metrics or {},
            "finance_approved": self.finance_approved,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "auto_threshold": self.auto_threshold,
            "selected_center": self.selected_center,
            "selected_scale": self.selected_scale,
            "margin_center": self.margin_center,
            "margin_scale": self.margin_scale,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def fit_logprob_calibrator(
    records: list[dict[str, Any]],
    *,
    model: str,
    endpoint_route: str,
    request_mode: str,
    code_map_version: str,
    dataset_version: str,
    auto_threshold: float = 0.90,
    finance_approved: bool = False,
    iterations: int = 2500,
    learning_rate: float = 0.05,
) -> LogprobCalibrator:
    """Fit a dependency-free logistic calibrator from held-out correctness rows.

    Each record must contain ``selected_mean_logprob``, ``logprob_margin`` and
    ``correct``. Records without complete candidate evidence are ignored.
    """
    usable: list[tuple[float, float, int]] = []
    for record in records:
        if record.get("evidence_scope") not in {None, "candidate_set", "full_taxonomy"}:
            continue
        try:
            selected = float(record["selected_mean_logprob"])
            margin = float(record["logprob_margin"])
            correct = int(bool(record["correct"]))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(selected) and math.isfinite(margin):
            usable.append((selected, margin, correct))
    if not usable:
        raise ValueError("No complete held-out logprob records are available for calibration")

    selected_center = sum(row[0] for row in usable) / len(usable)
    margin_center = sum(row[1] for row in usable) / len(usable)
    selected_scale = max(1e-12, math.sqrt(sum((row[0] - selected_center) ** 2 for row in usable) / len(usable)))
    margin_scale = max(1e-12, math.sqrt(sum((row[1] - margin_center) ** 2 for row in usable) / len(usable)))
    mean_label = sum(row[2] for row in usable) / len(usable)
    intercept = math.log(max(1e-6, min(1.0 - 1e-6, mean_label)) / max(1e-6, 1.0 - min(1.0 - 1e-6, mean_label)))
    selected_weight = 0.0
    margin_weight = 0.0
    for _ in range(max(1, iterations)):
        gradient_intercept = 0.0
        gradient_selected = 0.0
        gradient_margin = 0.0
        for selected, margin, correct in usable:
            selected = (selected - selected_center) / selected_scale
            margin = (margin - margin_center) / margin_scale
            probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, intercept + selected_weight * selected + margin_weight * margin))))
            error = probability - correct
            gradient_intercept += error
            gradient_selected += error * selected
            gradient_margin += error * margin
        denominator = len(usable)
        intercept -= learning_rate * gradient_intercept / denominator
        selected_weight -= learning_rate * gradient_selected / denominator
        margin_weight -= learning_rate * gradient_margin / denominator

    calibrator = LogprobCalibrator(
        intercept=intercept,
        selected_weight=selected_weight,
        margin_weight=margin_weight,
        model=model,
        endpoint_route=endpoint_route,
        request_mode=request_mode,
        code_map_version=code_map_version,
        sample_count=len(usable),
        validation_metrics={},
        finance_approved=finance_approved,
        dataset_version=dataset_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        auto_threshold=auto_threshold,
        selected_center=selected_center,
        selected_scale=selected_scale,
        margin_center=margin_center,
        margin_scale=margin_scale,
    )
    probabilities = [calibrator.score(row[0], row[1]) or 0.0 for row in usable]
    accepted = [index for index, probability in enumerate(probabilities) if probability >= auto_threshold]
    release_metrics = evaluate_release_gate(
        (
            {
                "accepted": index in accepted,
                "correct": correct,
                "is_synthetic": False,
            }
            for index, (_, _, correct) in enumerate(usable)
        ),
        finance_approved=finance_approved,
        review_first=False,
        minimum_real_accepted=0,
    )
    return LogprobCalibrator(
        **{
            **calibrator.to_dict(),
            "validation_metrics": {
                "accepted_precision": release_metrics["accepted_precision"],
                "accepted_precision_lower_bound": release_metrics["accepted_precision_lower_bound"],
                "accepted_count": release_metrics["accepted_real_count"],
                "accepted_real_count": release_metrics["accepted_real_count"],
                "accepted_real_correct": release_metrics["accepted_real_correct"],
                "brier_score": sum((probability - usable[index][2]) ** 2 for index, probability in enumerate(probabilities)) / len(probabilities),
            },
        }
    )


def _wilson_lower_bound(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) / total) + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - margin)


def load_calibrator(path: str | Path | None) -> LogprobCalibrator | None:
    if path is None or not str(path).strip():
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return LogprobCalibrator.from_json(candidate)
