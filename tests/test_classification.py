from types import SimpleNamespace

from classification import classify_invoice


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


def test_identity_cache_skips_embedding_and_maps_segment3():
    embedder = FakeEmbedder()
    db = FakeDB(
        identity={
            "real_post_count": 2,
            "conflict": False,
            "account_type": "Supplies",
            "account_types": ["Supplies"],
        }
    )
    result = classify_invoice(
        {"VendorName": "Acme", "LineItems": [{"LineDescription": "paper"}]},
        embedder,
        db_module=db,
    )["lines"][0]
    assert result["account_type"] == "Supplies"
    assert result["segment3"] == "60520"
    assert embedder.calls == []
    assert result["llm_skipped"] is True


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
        {"LineItems": [{"LineDescription": "office paper"}]},
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
