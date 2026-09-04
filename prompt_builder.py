from __future__ import annotations

import json
from typing import Any


def _historical_evidence(cases: list[dict[str, Any]]) -> str:
    """Serialize only classification evidence needed by the LLM.

    Similarity scores are intentionally excluded: they control retrieval in the
    backend, but must not become a proxy for the LLM's accounting decision.
    """
    evidence = [
        {
            "line_description": case.get("line_description"),
            "approved_account_type": case.get("account_type"),
        }
        for case in cases
    ]
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def _build_prompt(
    line_description: str,
    historical_cases: list[dict[str, Any]],
    evidence_instructions: str,
) -> str:
    line_json = json.dumps(line_description, ensure_ascii=False)
    return f"""You are an intelligent enterprise accounts-payable account-type classifier.

Your task is to classify a new invoice line description into its account type. An account type is the business or natural expense/balance-sheet category represented by the accounting treatment. It identifies what was purchased or incurred and supports consistent GL mapping, reporting, and downstream accounting.

Current invoice line description:
{line_description}

Historical evidence:
{_historical_evidence(historical_cases)}

{evidence_instructions}

Common classification instructions:
- Historical rows are past approved and classified accounting decisions covering different types of invoice lines. They are semantic examples, not exact string matches.
- Understand the current description semantically and lexically when useful: identify the good or service, business activity, and accounting nature.
- For each historical row, understand both what its description means and why it was assigned its approved account type. Then decide whether that reasoning applies to the current line.
- Do not copy an account type merely because words overlap or because a row was retrieved. The current line must have the same or sufficiently close business meaning.
- If a historical account type genuinely fits, return it and explain the semantic/accounting fit in `reason`.
- If no historical account type fits, return `account_type` as "Unknown", infer the most appropriate general account type in `inferred_account_type`, and explain why the retrieved cases do not apply.
- Do not choose or invent a GL code or segment value. A deterministic backend mapping is performed after your response.
- Confidence must be a number from 0.0 to 1.0 and reflect classification certainty and semantic fit, not retrieval score.

Return ONLY valid JSON with exactly these fields:
{{
  "line_description": {line_json},
  "account_type": "<matched account type or Unknown>",
  "inferred_account_type": null,
  "reason": "<concise, accounting-focused explanation>",
  "confidence": 0.0
}}

For a matching line, `inferred_account_type` must be null. For a non-matching line, `account_type` must be "Unknown" and `inferred_account_type` must contain the inferred general account type. Keep the reason concise but specific. No Markdown, code fences, or text outside the JSON object.
"""


def build_qualified_cases_prompt(
    line_description: str,
    historical_cases: list[dict[str, Any]],
) -> str:
    """Prompt for one or more cases that passed the similarity threshold."""
    line_json = json.dumps(line_description, ensure_ascii=False)
    return f"""You are an enterprise Accounts Payable account-type classifier.

Classify the CURRENT invoice line using its business and accounting meaning. First identify what was purchased or received, such as a product, service, fee, subscription, travel item, or other transaction. Use semantic meaning and, when useful, meaningful lexical clues.

The HISTORICAL CASES are approved accounting precedents. They passed the backend retrieval threshold and are strong hints, but retrieval similarity does not prove that their account types apply.

Similarity score and retrieval rank are metadata only. Do not classify based on the score, rank, or word overlap alone. Compare the current line with each historical description and understand why its approved account type may or may not apply.

If one historical account type clearly matches the current line's business and accounting nature:
- Return that approved account type in account_type.
- Copy its wording exactly. Do not rename, shorten, paraphrase, translate, or change capitalization, spacing, punctuation, or hyphen characters.
- Set inferred_account_type to null.
- Explain the semantic and accounting fit.

If the historical cases do not clearly apply:
- Return account_type as exactly "Unknown".
- Set inferred_account_type to the best possible account type.
- If a plausible historical account type exists, copy its wording exactly from the historical cases.
- If no historical account type is meaningfully related, provide a concise general account type based on the current line and explain that no reliable historical precedent applies.
- Explain why the historical cases do not fit and why the inference was made.

If historical cases have different account types, select the one with the closest business and accounting meaning. Do not use majority voting. If multiple account types remain equally plausible, return Unknown.

Do not return GL codes, segment values, mappings, or accounting entries. Confidence must reflect accounting certainty only, between 0.0 and 1.0. Return only valid JSON with exactly these fields:

{{
  "line_description": {line_json},
  "account_type": "<exact historical account type or Unknown>",
  "inferred_account_type": "<exact historical account type, general inferred type, or null>",
  "reason": "<1–3 concise accounting-focused sentences>",
  "confidence": 0.0
}}

CURRENT INVOICE LINE:
{line_description}

HISTORICAL CASES:
{_historical_evidence(historical_cases)}

Return no Markdown, code fences, or text outside the JSON object.
"""


def build_weak_cases_prompt(
    line_description: str,
    historical_cases: list[dict[str, Any]],
) -> str:
    """Prompt for the nearest cases when no case passed the threshold."""
    """Prompt for the nearest cases when no case passed the threshold."""
    line_json = json.dumps(line_description, ensure_ascii=False)
    return f"""You are an enterprise Accounts Payable account-type classifier.

Classify the CURRENT invoice line independently using its business and accounting meaning. First identify what was purchased or received, such as a product, service, fee, subscription, travel item, or other transaction. Use semantic meaning and, when useful, meaningful lexical clues.

The HISTORICAL CASES are approved accounting decisions, but none passed the backend similarity threshold. They are the nearest available references and may be weak, unrelated, or misleading. Treat them only as possible accounting hints.

Similarity score and retrieval rank are metadata only. Never classify based on score, rank, proximity, or word overlap alone. Do not force a classification just because five historical cases are provided.

Compare the current line with each historical description and understand why its approved account type may or may not apply. Use a historical account type only when the current line genuinely has the same business and accounting nature.

If one historical account type clearly matches:
- Return that approved account type in account_type.
- Copy its wording exactly. Do not rename, shorten, paraphrase, translate, or change capitalization, spacing, punctuation, or hyphen characters.
- Set inferred_account_type to null.
- Explain the semantic and accounting fit.

If no historical account type clearly matches:
- Return account_type as exactly "Unknown".
- Set inferred_account_type to the best possible account type.
- If a weak historical account type is still the closest reasonable candidate, copy its wording exactly from the historical cases.
- If none is meaningfully related, provide a concise general account type based on the current line and explain that it is a general inference rather than a reliable historical precedent.
- Explain why the historical cases do not fit and why the inference was made.

If historical cases have different account types, select the one with the closest business and accounting meaning. Do not use majority voting. If multiple account types remain equally plausible, return Unknown.

Do not return GL codes, segment values, mappings, or accounting entries. Confidence must reflect accounting certainty only, between 0.0 and 1.0. Return only valid JSON with exactly these fields:

{{
  "line_description": {line_json},
  "account_type": "<exact historical account type or Unknown>",
  "inferred_account_type": "<exact historical account type, general inferred type, or null>",
  "reason": "<1–3 concise accounting-focused sentences>",
  "confidence": 0.0
}}

CURRENT INVOICE LINE:
{line_description}

HISTORICAL CASES:
{_historical_evidence(historical_cases)}

Return no Markdown, code fences, or text outside the JSON object.
"""


def build_account_type_prompt(
    line_description: str,
    historical_cases: list[dict[str, Any]],
) -> str:
    """Backward-compatible alias for the qualified-evidence prompt."""
    return build_qualified_cases_prompt(line_description, historical_cases)
