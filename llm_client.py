from __future__ import annotations

import os
from typing import Any

from oci_openai import OciOpenAI, OciUserPrincipalAuth


class OCIResponsesClient:
    """OCI OpenAI-compatible Responses API client."""

    def __init__(self) -> None:
        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise ValueError("LLM_MODEL must be set in GL_code_classify/.env")
        # No config path is passed: OciUserPrincipalAuth uses the user's
        # default OCI config location and DEFAULT profile automatically.
        auth = OciUserPrincipalAuth()
        self.model = model
        self.client = OciOpenAI(
            auth=auth,
            service_endpoint=os.getenv("OCI_SERVICE_ENDPOINT"),
            compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        )

    def call(self, prompt: str) -> Any:
        return self.client.responses.create(model=self.model, input=prompt, temperature=0)
