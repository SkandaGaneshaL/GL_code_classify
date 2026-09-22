from types import SimpleNamespace

from classification import classify_invoice, classify_line
from segment3_ranker import RankerCalibrationArtifact


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        return [0.0] * 1024


class FakeDB:
    def __init__(self, cases=None, identity=None, prior=None):
        self.cases = cases or []
        self.identity = identity
        self.prior = prior

    def get_identity_match(self, key):
        return self.identity

    def get_vendor_prior(self, vendor):
        return self.prior

    def search_hybrid_history(self, feature, embedding, **kwargs):
        return self.cases


class FakeLLM:
    def __init__(self, account_type="Supplies"):
        self.account_type = account_type
        self.calls = 0

    def call(self, prompt, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            output_text=(
                '{"line_description":"paper","account_type":"%s",'
                '"inferred_account_type":null,"reason":"consumable office item",'
                '"confidence":0.92}' % self.account_type
            )
        )


class FailingLLM:
    def call(self, prompt, **kwargs):
        raise RuntimeError("provider unavailable")


def test_identity_cache_is_evidence_and_llm_still_classifies():
    embedder = FakeEmbedder()
    llm = FakeLLM("Supplies")
    db = FakeDB(
        identity={
            "real_post_count": 2,
            "conflict": False,
            "account_type": "Supplies",
            "account_types": ["Supplies"],
        }
    )
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "paper", "LineType": "ITEM"}]},
        embedder,
        db_module=db,
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert result["account_type"] == "Supplies"
    assert result["segment3"] == "60520"
    assert embedder.calls
    assert result["llm_called"] is True
    assert result["llm_route"] == "llm"
    assert llm.calls == 1


def test_software_is_unknown_even_when_retrieval_has_a_mapped_case():
    llm = FakeLLM("Supplies")
    db = FakeDB(cases=[{"id": 1, "account_type": "Supplies", "line_description": "paper"}])
    result = classify_invoice(
        {
            "VendorName": "Lee Supplies",
            "LineItems": [{"LineDescription": "RHEL enterprise Linux license", "LineType": "ITEM"}],
        },
        FakeEmbedder(),
        db_module=db,
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert result["account_type"] == "Unknown"
    assert result["segment3"] is None
    assert "unmapped_software" in result["conflict_flags"]
    assert llm.calls == 1


def test_tax_line_is_review_only_until_policy_is_approved():
    llm = FakeLLM("Withholding Tax Payable")
    db = FakeDB(cases=[{"id": 1, "account_type": "Withholding Tax Payable", "line_description": "tax"}])
    result = classify_invoice(
        {"LineItems": [{"LineDescription": "GST tax charge", "LineType": "TAX"}]},
        FakeEmbedder(),
        db_module=db,
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert result["account_type"] == "Unknown"
    assert result["segment3"] is None
    assert "unsupported_tax" in result["conflict_flags"]


def test_confidence_breakdown_exposes_contributions_and_penalties():
    llm = FakeLLM("Supplies")
    db = FakeDB(
        cases=[
            {"id": 1, "account_type": "Supplies", "line_description": "office paper", "similarity_score": 0.92},
            {"id": 2, "account_type": "Meals", "line_description": "office paper", "similarity_score": 0.80},
        ]
    )
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=db,
        llm_client_factory=lambda: llm,
    )["lines"][0]
    breakdown = result["confidence_breakdown"]
    assert breakdown["retrieval_weight"] == 0.40
    assert breakdown["llm_weight"] == 0.15
    assert breakdown["llm_signal"] == 0.0
    assert result["self_reported_confidence"] == 0.92
    assert result["confidence_source"] == "heuristic_composite"
    assert result["logprob_confidence_source"] == "diagnostic_only"
    assert breakdown["penalties"][0]["flag"] == "interleaved_account_types"
    assert breakdown["final_confidence"] == result["confidence"]
    assert "clamp(0.40R" in breakdown["formula"]


def test_blank_line_has_not_called_logprob_explanation():
    result = classify_invoice(
        {"LineItems": [{"LineDescription": ""}]},
        FakeEmbedder(),
        db_module=FakeDB(),
    )["lines"][0]
    assert result["logprob_explanation"]["status"] == "not_called"
    assert result["confidence_breakdown"]["final_confidence"] == 0.0


def test_compound_extracted_line_is_routed_to_review_without_llm_call():
    llm = FakeLLM("Asset Clearing")
    result = classify_invoice(
        {
            "LineItems": [
                {
                    "LineDescription": (
                        "Office workstation capital asset acquisition, equipment lease charges, "
                        "and withholding tax adjustment."
                    ),
                    "LineType": "ITEM",
                }
            ]
        },
        FakeEmbedder(),
        db_module=FakeDB(),
        llm_client_factory=lambda: llm,
    )["lines"][0]

    assert result["account_type"] == "Unknown"
    assert result["decision"] == "REVIEW_REQUIRED"
    assert "compound_account_nature" in result["review_reasons"]
    assert result["llm_called"] is False
    assert result["split_suggestions"]["status"] == "split_suggested"
    assert [child["segment3"] for child in result["split_suggestions"]["children"]] == [None, None, None]


def test_retrieval_candidates_are_limited_to_valid_top_three_review_suggestions():
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(
            cases=[
                {"id": 1, "account_type": "Supplies", "rrf_score": 0.9},
                {"id": 2, "account_type": "Meals", "rrf_score": 0.8},
                {"id": 3, "account_type": "Invented Account", "rrf_score": 1.0},
            ]
        ),
        llm_client_factory=lambda: FakeLLM("Supplies"),
    )["lines"][0]

    assert [candidate["account_type"] for candidate in result["ranked_candidates"]] == ["Supplies", "Meals"]
    assert result["ranker_confidence_status"] == "unavailable"


def test_compatible_ranker_artifact_is_only_source_of_live_percentage():
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(cases=[{"id": 1, "account_type": "Supplies", "rrf_score": 0.9}]),
        classification_context={
            "ranker_calibration_artifact": RankerCalibrationArtifact(
                ranker_version="ranker-v1",
                feature_schema_version="segment3-features-v1",
                coa_version="segment3-demo-16-v1",
                dataset_version="finance-v1",
                calibration_version="cal-v1",
                intercept=0.0,
                retrieval_weight=1.0,
            )
        },
        llm_client_factory=lambda: FakeLLM("Supplies"),
    )["lines"][0]

    assert result["confidence_source"] == "calibrated_ranker"
    assert result["confidence_status"] == "calibrated"
    assert result["calibrated_correctness_probability"] == result["ranked_candidates"][0]["calibrated_probability"]
    assert result["decision"] == "REVIEW_REQUIRED"


def test_strict_finance_mode_requires_active_coa_artifacts():
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(cases=[{"id": 1, "account_type": "Supplies", "rrf_score": 0.9}]),
        classification_context={"strict_finance": True},
        llm_client_factory=lambda: FakeLLM("Supplies"),
    )["lines"][0]
    assert result["segment3"] is None
    assert result["coa_validation"]["reason"] == "finance_coa_artifact_required"
    assert result["decision"] == "REVIEW_REQUIRED"


def test_missing_vendor_still_calls_llm_and_requires_review():
    llm = FakeLLM("Supplies")
    result = classify_invoice(
        {"LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(cases=[{"account_type": "Supplies", "rrf_score": 0.9}]),
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert llm.calls == 1
    assert result["llm_route"] == "llm"
    assert result["decision_band"] == "REVIEW_REQUIRED"
    assert "missing_vendor" in result["review_reasons"]
    assert result["segment3"] == "60520"


def test_missing_line_type_still_calls_llm_and_requires_review():
    llm = FakeLLM("Supplies")
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper"}]},
        FakeEmbedder(),
        db_module=FakeDB(cases=[{"account_type": "Supplies", "rrf_score": 0.9}]),
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert llm.calls == 1
    assert result["llm_called"] is True
    assert "missing_line_type" in result["review_reasons"]
    assert result["decision_band"] == "REVIEW_REQUIRED"


def test_no_history_uses_full_taxonomy_for_constrained_llm():
    llm = FakeLLM("Supplies")
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(),
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert llm.calls == 1
    assert result["llm_route"] == "llm"
    assert result["segment3"] == "60520"


def test_uncertain_only_preserves_retrieval_fast_path():
    llm = FakeLLM("Supplies")
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(cases=[
            {"account_type": "Supplies", "rrf_score": 0.9},
            {"account_type": "Supplies", "rrf_score": 0.8},
            {"account_type": "Supplies", "rrf_score": 0.7},
        ]),
        classification_context={"llm_routing_mode": "uncertain_only"},
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert llm.calls == 0
    assert result["llm_called"] is False
    assert result["llm_route"] == "retrieval_evidence"


def test_missing_description_does_not_call_llm():
    llm = FakeLLM("Supplies")
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(),
        llm_client_factory=lambda: llm,
    )["lines"][0]
    assert llm.calls == 0
    assert result["llm_route"] == "quality_block"
    assert result["llm_skip_reason"] == "missing_line_description"


def test_llm_failure_is_explicitly_reported():
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "office paper", "LineType": "ITEM"}]},
        FakeEmbedder(),
        db_module=FakeDB(cases=[{"account_type": "Supplies", "rrf_score": 0.9}]),
        llm_client_factory=lambda: FailingLLM(),
    )["lines"][0]
    assert result["llm_called"] is True
    assert result["llm_route"] == "llm_error"
    assert "provider unavailable" in result["llm_failure_reason"]
    assert result["segment3"] is None


def test_description_only_single_line_wrapper_calls_llm():
    llm = FakeLLM("Supplies")
    result = classify_line(
        "Printer toner",
        FakeEmbedder(),
        settings=None,
        db_module=FakeDB(cases=[{"account_type": "Supplies", "rrf_score": 0.9}]),
        llm_client_factory=lambda: llm,
    )
    assert llm.calls == 1
    assert result["llm_called"] is True
    assert result["llm_route"] == "llm"
    assert result["decision_band"] == "REVIEW_REQUIRED"
    assert result["segment3"] == "60520"
