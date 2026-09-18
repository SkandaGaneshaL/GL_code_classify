from line_atomizer import atomize_line_description


def test_atomizer_returns_review_only_children_for_compound_line():
    result = atomize_line_description(
        "Office workstation asset acquisition, equipment lease charges, and withholding tax adjustment."
    )

    assert result["status"] == "split_suggested"
    assert result["requires_review"] is True
    assert [child["account_nature"] for child in result["children"]] == ["asset", "lease", "tax"]
    assert all(child["segment3"] is None for child in result["children"])


def test_atomizer_leaves_atomic_line_unsplit():
    result = atomize_line_description("Printer toner cartridges")

    assert result == {"status": "not_compound", "requires_review": False, "children": []}
