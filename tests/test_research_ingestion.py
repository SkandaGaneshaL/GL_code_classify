from pathlib import Path

from research_ingestion import build_url_inventory, summarize_audit


def test_inventory_deduplicates_urls_and_preserves_invalid_entries(tmp_path: Path):
    first = tmp_path / "first.md"
    first.write_text("[one](https://example.com/a) https://example.com/a https://<pod", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("https://example.com/b", encoding="utf-8")

    inventory = build_url_inventory([first, second])

    assert len(inventory) == 3
    duplicate = next(item for item in inventory if item["url"] == "https://example.com/a")
    assert duplicate["source_count"] == 2
    invalid = next(item for item in inventory if item["url"] == "https://<pod")
    assert invalid["status"] == "invalid_url"


def test_audit_summary_does_not_count_failed_or_invalid_urls_as_reviewed():
    summary = summarize_audit(
        [
            {"status": "fetched", "http_status": 200, "evidence_grade": "primary_vendor"},
            {"status": "http_error", "http_status": 403, "evidence_grade": "unavailable"},
            {"status": "invalid_url", "evidence_grade": "unavailable"},
        ]
    )

    assert summary["total_urls"] == 3
    assert summary["reviewed_urls"] == 1
    assert summary["unreviewed_urls"] == 2
    assert summary["status_counts"] == {"fetched": 1, "http_error": 1, "invalid_url": 1}
    assert summary["evidence_reviewed_only"] is True


def test_inventory_joins_wrapped_urls_and_deduplicates_invalid_template(tmp_path: Path):
    source = tmp_path / "wrapped.md"
    source.write_text(
        "https://example.com/a/very-long-\n"
        "  path.html\n"
        "https://<pod`, https://<pod",
        encoding="utf-8",
    )
    inventory = build_url_inventory([source])
    assert any(item["canonical_url"] == "https://example.com/a/very-long-path.html" for item in inventory)
    invalid = [item for item in inventory if item["status"] == "invalid_url"]
    assert len(invalid) == 1
    assert invalid[0]["source_count"] == 2
