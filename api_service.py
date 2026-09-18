"""Versioned review-first HTTP boundary for Segment 3 classification."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

try:
    from .result_contract import apply_result_contract
except ImportError:
    from result_contract import apply_result_contract


class InvoiceLineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    LineDescription: str | None = None
    LineType: str | None = None
    LineAmount: str | float | int | None = None
    UnitPrice: str | float | int | None = None
    QuantityInvoiced: str | float | int | None = None


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    VendorName: str | None = None
    PayeeName: str | None = None
    LineItems: list[InvoiceLineRequest] = Field(min_length=1)


def create_app(classifier: Callable[[dict[str, Any]], dict[str, Any]]) -> FastAPI:
    app = FastAPI(title="GL Segment 3 classification", version="v1")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "policy": "review-first-v1"}

    @app.post("/v1/classifications")
    def classify(request: ClassificationRequest) -> dict[str, Any]:
        result = classifier(request.model_dump(exclude_none=False))
        lines = [apply_result_contract(line) for line in result.get("lines") or []]
        return {**result, "lines": lines, "api_version": "v1", "policy_version": "review-first-v1"}

    return app
