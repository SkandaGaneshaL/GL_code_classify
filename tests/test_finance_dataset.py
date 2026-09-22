import pytest

from finance_dataset import validate_finance_rows


class _ActiveCoa:
    segment_values = {"60520"}


class _EmptyActiveCoa:
    segment_values = set()


def test_real_finance_row_requires_governed_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_finance_rows(
            [
                {
                    "invoice_distribution_id": "d-1",
                    "invoice_date": "2026-01-01",
                    "source_group_id": "invoice-1",
                    "invoice_source": "ERP",
                    "line_source": "AP_INVOICE_LINE",
                    "requester": "finance@example.com",
                    "is_synthetic": False,
                }
            ]
        )


def test_synthetic_row_is_allowed_but_excluded_from_certification():
    report = validate_finance_rows(
        [
            {
                "invoice_distribution_id": "d-1",
                "invoice_date": "2026-01-01",
                "source_group_id": "invoice-1",
                "invoice_source": "ERP",
                "line_source": "AP_INVOICE_LINE",
                "requester": "finance@example.com",
                "vendor_id": "v-1",
                "vendor_name": "Acme",
                "vendor_site_id": "site-1",
                "business_unit": "bu-1",
                "legal_entity": "le-1",
                "ledger": "ledger-1",
                "chart_of_accounts": "coa-1",
                "line_type": "ITEM",
                "line_description": "Printer toner",
                "line_amount": "100.00",
                "currency": "USD",
                "segment3": "60520",
                "final_posted": True,
                "is_synthetic": True,
            }
        ]
    )

    assert report == {"row_count": 1, "real_row_count": 0, "synthetic_row_count": 1}


def test_excel_style_n_flag_is_counted_as_real_not_truthy_synthetic():
    row = {
        "invoice_distribution_id": "d-1",
        "invoice_date": "2026-01-01",
        "source_group_id": "invoice-1",
        "vendor_id": "v-1",
        "vendor_name": "Acme",
        "vendor_site_id": "site-1",
        "business_unit": "bu-1",
        "legal_entity": "le-1",
        "ledger": "ledger-1",
        "chart_of_accounts": "coa-1",
        "invoice_source": "EMAIL",
        "line_source": "INVOICE",
        "requester": "requester-1",
        "line_type": "ITEM",
        "line_description": "Printer toner",
        "line_amount": "100.00",
        "currency": "USD",
        "segment3": "60520",
        "final_posted": True,
        "is_synthetic": "N",
    }

    report = validate_finance_rows([row])

    assert report["real_row_count"] == 1


def test_strict_contract_rejects_non_posted_or_invalid_segment():
    row = {
        "invoice_distribution_id": "d-1",
        "invoice_date": "2026-01-01",
        "source_group_id": "invoice-1",
        "vendor_id": "v-1",
        "vendor_name": "Acme",
        "vendor_site_id": "site-1",
        "business_unit": "bu-1",
        "legal_entity": "le-1",
        "ledger": "ledger-1",
        "chart_of_accounts": "coa-1",
        "invoice_source": "EMAIL",
        "line_source": "INVOICE",
        "requester": "requester-1",
        "line_type": "ITEM",
        "line_description": "Printer toner",
        "natural_account_description": "Supplies",
        "line_amount": "100.00",
        "currency": "USD",
        "segment1": "101",
        "segment2": "10",
        "segment3": "99999",
        "segment4": "000",
        "segment5": "000",
        "segment6": "000",
        "source_invoice_distribution_id": "d-1",
        "final_posted": "N",
        "is_synthetic": "N",
    }
    with pytest.raises(ValueError, match="final posted"):
        validate_finance_rows([row], strict=True)


def test_certification_requires_active_coa_artifact():
    with pytest.raises(ValueError, match="active Finance COA"):
        validate_finance_rows([], strict=True, certification=True)


def test_certification_uses_active_coa_values():
    row = {
        "invoice_distribution_id": "d-1",
        "invoice_date": "2026-01-01",
        "source_group_id": "invoice-1",
        "invoice_source": "ERP",
        "line_source": "AP_INVOICE_LINE",
        "requester": "finance@example.com",
        "vendor_id": "v-1",
        "vendor_name": "Acme",
        "vendor_site_id": "site-1",
        "business_unit": "bu-1",
        "legal_entity": "le-1",
        "ledger": "ledger-1",
        "chart_of_accounts": "coa-1",
        "line_type": "ITEM",
        "line_description": "Printer toner",
        "natural_account_description": "Supplies",
        "line_amount": "100.00",
        "currency": "USD",
        "segment1": "101",
        "segment2": "10",
        "segment3": "60520",
        "segment4": "000",
        "segment5": "000",
        "segment6": "000",
        "final_posted": "Y",
        "is_synthetic": "N",
    }
    report = validate_finance_rows([row], strict=True, active_coa=_ActiveCoa(), certification=True)
    assert report["real_row_count"] == 1


def test_certification_rejects_empty_active_coa_value_set():
    row = {
        "invoice_distribution_id": "d-empty-coa",
        "invoice_date": "2026-01-01",
        "source_group_id": "invoice-empty-coa",
        "invoice_source": "ERP",
        "line_source": "AP_INVOICE_LINE",
        "requester": "finance@example.com",
        "vendor_id": "v-1",
        "vendor_name": "Acme",
        "vendor_site_id": "site-1",
        "business_unit": "bu-1",
        "legal_entity": "le-1",
        "ledger": "ledger-1",
        "chart_of_accounts": "coa-1",
        "line_type": "ITEM",
        "line_description": "Printer toner",
        "natural_account_description": "Supplies",
        "line_amount": "100.00",
        "currency": "USD",
        "segment1": "101",
        "segment2": "10",
        "segment3": "60520",
        "segment4": "000",
        "segment5": "000",
        "segment6": "000",
        "final_posted": "Y",
        "is_synthetic": "N",
    }
    with pytest.raises(ValueError, match="non-empty active Finance Segment 3"):
        validate_finance_rows(
            [row],
            strict=True,
            active_coa=_EmptyActiveCoa(),
            certification=True,
        )
