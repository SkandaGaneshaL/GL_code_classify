import json

from calibration import LogprobCalibrator, fit_logprob_calibrator, load_calibrator


def test_calibrator_returns_bounded_signal(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"intercept": 0, "selected_weight": 1, "margin_weight": 2, "temperature": 1}),
        encoding="utf-8",
    )
    calibrator = load_calibrator(path)
    assert calibrator is not None
    assert 0.0 < calibrator.score(-0.1, 1.0) < 1.0


def test_missing_calibration_artifact_disables_calibrated_signal(tmp_path):
    assert load_calibrator(tmp_path / "missing.json") is None
    assert LogprobCalibrator().score(None, None) is None


def test_calibration_requires_explicit_runtime_metadata(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "intercept": 0,
                "model": "openai.gpt-oss-20b",
                "endpoint_route": "oci_native_chat",
                "request_mode": "structured_json",
                "code_map_version": "compact-code-v1",
                "sample_count": 40,
                "validation_metrics": {"top1_accuracy": 0.9, "calibration_error": 0.05},
                "finance_approved": False,
            }
        ),
        encoding="utf-8",
    )
    calibrator = load_calibrator(path)
    assert calibrator is not None
    assert calibrator.is_compatible(
        model="openai.gpt-oss-20b",
        route="oci_native_chat",
        request_mode="structured_json",
        code_map_version="compact-code-v1",
    )
    assert not calibrator.is_compatible(
        model="openai.gpt-4o",
        route="oci_native_chat",
        request_mode="structured_json",
        code_map_version="compact-code-v1",
    )


def test_auto_readiness_requires_finance_approval_sample_support_and_lower_bound():
    calibrator = LogprobCalibrator(
        model="model",
        endpoint_route="route",
        request_mode="structured_json",
        code_map_version="compact-code-v1",
        sample_count=100,
        validation_metrics={"accepted_precision_lower_bound": 0.981},
        finance_approved=True,
        auto_threshold=0.90,
    )
    assert calibrator.is_auto_ready(target_precision=0.98, minimum_samples=50)
    assert not calibrator.is_auto_ready(target_precision=0.99, minimum_samples=50)


def test_fit_logprob_calibrator_produces_versioned_artifact(tmp_path):
    records = [
        {
            "selected_mean_logprob": -0.1 + index * 0.001,
            "logprob_margin": 1.0 + index * 0.01,
            "correct": index % 5 != 0,
            "evidence_scope": "full_taxonomy",
        }
        for index in range(20)
    ]
    calibrator = fit_logprob_calibrator(
        records,
        model="model",
        endpoint_route="route",
        request_mode="structured_json",
        code_map_version="compact-code-v1",
        dataset_version="test-v1",
    )
    path = tmp_path / "calibration.json"
    calibrator.save(path)
    loaded = load_calibrator(path)
    assert loaded is not None
    assert loaded.artifact_version == "segment3-logprob-calibration-v2"
    assert loaded.dataset_version == "test-v1"
    assert loaded.sample_count == 20
    assert "accepted_precision_lower_bound" in loaded.validation_metrics
