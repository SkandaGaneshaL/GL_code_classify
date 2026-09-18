from pathlib import Path

from research_ingestion import build_url_inventory


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
