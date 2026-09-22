import pytest

from split_review import accept_split_suggestions, merge_split_suggestions, persist_split_decision


def test_accept_split_never_assigns_segment3():
    result = accept_split_suggestions(
        {"status": "split_suggested", "children": [{"description": "paper"}, {"description": "meals"}]},
        [{"description": "paper", "account_nature": "supplies"}],
    )
    assert result["status"] == "split_accepted_for_coding"
    assert result["children"][0]["segment3"] is None


def test_merge_requires_multiple_children():
    with pytest.raises(ValueError):
        merge_split_suggestions([{"description": "paper"}])


def test_persist_split_decision_calls_audit_sink():
    calls = []

    class FakeDb:
        @staticmethod
        def record_split_review_decision(**kwargs):
            calls.append(kwargs)

    decision = {"status": "split_accepted_for_coding", "children": [{"description": "paper"}]}
    assert persist_split_decision(FakeDb, source_line_id="line-1", decision=decision) is True
    assert calls[0]["source_line_id"] == "line-1"
