from retrieval import reciprocal_rank_fusion, retrieval_summary
from db_utils import _history_filters


def test_rrf_merges_dense_and_sparse_by_case_id():
    dense = [{"id": 1, "account_type": "Supplies"}, {"id": 2, "account_type": "Meals"}]
    sparse = [{"id": 2, "account_type": "Meals"}, {"id": 1, "account_type": "Supplies"}]
    merged = reciprocal_rank_fusion(dense, sparse, k=60)
    assert {row["id"] for row in merged} == {1, 2}
    assert all("rrf_score" in row for row in merged)


def test_retrieval_summary_detects_unanimity():
    summary = retrieval_summary(
        [
            {"account_type": "Supplies", "rrf_score": 0.03},
            {"account_type": "Supplies", "rrf_score": 0.02},
            {"account_type": "Supplies", "rrf_score": 0.01},
        ]
    )
    assert summary["top3_unanimous"] is True
    assert summary["top_account_type"] == "Supplies"


def test_retrieval_summary_reports_actual_top_three_types():
    summary = retrieval_summary(
        [
            {"account_type": "Meals", "rrf_score": 0.04},
            {"account_type": "Meals", "rrf_score": 0.03},
            {"account_type": "Meals", "rrf_score": 0.02},
            {"account_type": "Leases", "rrf_score": 0.01},
        ]
    )
    assert summary["top3_types"] == ["Meals", "Meals", "Meals"]
    assert summary["top3_unanimous"] is True


def test_history_filters_default_to_history_and_exclude_evaluation_rows():
    clauses, params = _history_filters(
        dataset_types=("HISTORY",),
        exclude_ids=(12,),
        exclude_invoice_distribution_ids=("DIST-1",),
        exclude_source_invoice_distribution_ids=("DIST-0",),
    )
    rendered = " ".join(clauses)
    assert "DATASET_TYPE" in rendered
    assert "HISTORY" in params.values()
    assert "exclude_id_0" in params
    assert "exclude_distribution_0" in params
    assert "exclude_source_distribution_0" in params
