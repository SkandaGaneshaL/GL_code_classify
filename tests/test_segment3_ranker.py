from segment3_ranker import (
    RankerCalibrationArtifact,
    calibrate_ranker_artifact,
    fit_ranker_artifact,
    load_ranker_calibration,
    rank_candidates,
)


def test_ranker_only_exposes_probability_from_compatible_artifact():
    candidates = [
        {"account_type": "Supplies", "retrieval_score": 0.9},
        {"account_type": "Meals", "retrieval_score": 0.1},
        {"account_type": "Invented", "retrieval_score": 1.0},
    ]
    artifact = RankerCalibrationArtifact(
        ranker_version="ranker-v1",
        feature_schema_version="segment3-features-v1",
        coa_version="coa-v1",
        dataset_version="data-v1",
        calibration_version="cal-v1",
        intercept=0.0,
        retrieval_weight=1.0,
    )

    result = rank_candidates(candidates, artifact=artifact, coa_version="coa-v1")

    assert [item["account_type"] for item in result["candidates"]] == ["Supplies", "Meals"]
    assert result["selected"]["account_type"] == "Supplies"
    assert result["selected"]["calibrated_probability"] is not None
    assert result["confidence_status"] == "calibrated"


def test_ranker_keeps_probability_unavailable_when_artifact_is_incompatible():
    artifact = RankerCalibrationArtifact(coa_version="old-coa")

    result = rank_candidates(
        [{"account_type": "Supplies", "retrieval_score": 0.9}], artifact=artifact, coa_version="new-coa"
    )

    assert result["selected"]["calibrated_probability"] is None
    assert result["confidence_status"] == "unavailable"


def test_ranker_artifact_round_trips_required_version_metadata(tmp_path):
    artifact = RankerCalibrationArtifact(
        ranker_version="ranker-v1",
        feature_schema_version="segment3-features-v1",
        coa_version="coa-v1",
        dataset_version="finance-2026-01",
        calibration_version="cal-v2",
        intercept=-1.0,
        retrieval_weight=2.0,
    )
    path = tmp_path / "ranker-calibration.json"
    artifact.save(path)

    loaded = load_ranker_calibration(path)

    assert loaded == artifact


def test_training_artifact_requires_separate_held_out_calibration():
    candidates = [[{"account_type": "Supplies", "retrieval_score": 0.9}], [{"account_type": "Meals", "retrieval_score": 0.8}]]
    artifact = fit_ranker_artifact(
        candidates,
        ["Supplies", "Meals"],
        ranker_version="ranker-v2",
        feature_schema_version="segment3-features-v2",
        coa_version="coa-v1",
        dataset_version="data-v1",
        calibration_version="cal-v1",
    )
    assert rank_candidates(candidates[0], artifact=artifact, coa_version="coa-v1")["confidence_status"] == "unavailable"
    calibrated = calibrate_ranker_artifact(artifact, candidates, ["Supplies", "Meals"])
    assert rank_candidates(candidates[0], artifact=calibrated, coa_version="coa-v1")["confidence_status"] == "calibrated"


def test_strict_runtime_compatibility_requires_all_manifest_dimensions():
    artifact = RankerCalibrationArtifact(
        ranker_version="ranker-v1",
        feature_schema_version="features-v1",
        coa_version="coa-v1",
        dataset_version="data-v1",
        calibration_version="cal-v1",
        model_route="segment3_ranker",
    )
    result = rank_candidates(
        [{"account_type": "Supplies", "retrieval_score": 0.9}],
        artifact=artifact,
        coa_version="coa-v1",
        feature_schema_version="features-v1",
        dataset_version="data-v1",
        calibration_version="cal-v1",
        model_route="wrong-route",
        require_strict_compatibility=True,
    )
    assert result["confidence_status"] == "unavailable"
