from shadow_monitor import summarize_shadow_week


def test_shadow_week_is_review_only_and_reports_rates():
    report = summarize_shadow_week(
        [
            {"segment3": "60520", "correct": True, "override": False, "vendor_name": "Acme", "line_type": "ITEM"},
            {"segment3": "60521", "correct": False, "override": True, "vendor_name": "Acme", "line_type": "ITEM"},
        ]
    )
    assert report["auto_post_enabled"] is False
    assert report["coverage"]["percent"] == "100.00%"
    assert report["override_rate"]["percent"] == "50.00%"
