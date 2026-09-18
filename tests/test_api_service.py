import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api_service import create_app


def test_classification_service_exposes_versioned_contract():
    def classifier(payload):
        assert payload["LineItems"][0]["LineDescription"] == "Printer toner"
        return {
            "lines": [
                {
                    "account_type": "Supplies",
                    "segment3": "60520",
                    "decision_band": "REVIEW",
                    "confidence_source": "none",
                }
            ]
        }

    client = TestClient(create_app(classifier))
    response = client.post(
        "/v1/classifications",
        json={"VendorName": "Acme", "LineItems": [{"LineDescription": "Printer toner", "LineType": "ITEM"}]},
    )

    assert response.status_code == 200
    line = response.json()["lines"][0]
    assert line["prediction"]["account_type"] == "Supplies"
    assert line["decision"] == "REVIEW_REQUIRED"


def test_classification_service_rejects_unapproved_input_fields():
    client = TestClient(create_app(lambda payload: {"lines": []}))
    response = client.post(
        "/v1/classifications",
        json={"LineItems": [{"LineDescription": "Printer toner", "Currency": "USD"}]},
    )

    assert response.status_code == 422
