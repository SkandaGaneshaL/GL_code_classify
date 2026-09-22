from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from contextlib import nullcontext
from typing import Any

from dotenv import load_dotenv
import streamlit as st

try:
    from . import db_utils
    from .classification import map_account_type_to_segments
    from .config import (
        AP_EXTRACTION_MAX_BYTES,
        BASE_DIR,
        CLASSIFIER_MODEL_ID,
        CLASSIFIER_PROVIDER,
        LLM_NATIVE_REASONING_EFFORT,
        REQUIRED_CLASSIFIER_MODEL,
        validate_classifier_configuration,
    )
    from .embeddings import create_document_embedder, embed_documents, load_oci_settings
    from .evaluation import load_evaluation_rows, run_evaluation
    from .feature_pack import build_line_feature
    from .split_review import accept_split_suggestions, merge_split_suggestions, persist_split_decision
    from .invoice_pipeline import InvoiceClassificationPipeline, InvoicePipelineError
except ImportError:  # Supports streamlit run streamlit_app.py from this folder.
    import db_utils
    from classification import map_account_type_to_segments
    from config import (
        AP_EXTRACTION_MAX_BYTES,
        BASE_DIR,
        CLASSIFIER_MODEL_ID,
        CLASSIFIER_PROVIDER,
        LLM_NATIVE_REASONING_EFFORT,
        REQUIRED_CLASSIFIER_MODEL,
        validate_classifier_configuration,
    )
    from embeddings import create_document_embedder, embed_documents, load_oci_settings
    from evaluation import load_evaluation_rows, run_evaluation
    from feature_pack import build_line_feature
    from split_review import accept_split_suggestions, merge_split_suggestions, persist_split_decision
    from invoice_pipeline import InvoiceClassificationPipeline, InvoicePipelineError


MAX_UPLOAD_MB = max(1, AP_EXTRACTION_MAX_BYTES // (1024 * 1024))


@st.cache_resource(show_spinner=False)
def get_runtime(config_fingerprint: str) -> tuple[InvoiceClassificationPipeline, Any]:
    del config_fingerprint
    _refresh_local_classifier_environment()
    validate_classifier_configuration()
    settings = load_oci_settings()
    pipeline = InvoiceClassificationPipeline(settings=settings)
    document_embedder = create_document_embedder(settings)
    return pipeline, document_embedder


def _refresh_local_classifier_environment() -> None:
    env_path = BASE_DIR / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)


def _runtime_config_fingerprint() -> str:
    _refresh_local_classifier_environment()
    env_path = BASE_DIR / ".env"
    env_stamp = str(env_path.stat().st_mtime_ns) if env_path.is_file() else "missing"
    values = "|".join(
        [
            env_stamp,
            *(
                os.getenv(key, "")
                for key in (
                    "LLM_MODEL",
                    "GL_REQUIRED_CLASSIFIER_MODEL",
                    "OCI_SERVICE_ENDPOINT",
                    "OCI_COMPARTMENT_ID",
                    "OCI_PROFILE",
                    "GL_CAPABILITY_PROBE_CACHE_VERSION",
                    "GL_LLM_ROUTING_MODE",
                )
            ),
        ]
    )
    return hashlib.sha256(values.encode("utf-8")).hexdigest()[:16]


def _render_rate_metric(label: str, metric: dict[str, Any]) -> None:
    value = metric.get("value")
    display = "Unavailable" if value is None else str(metric.get("percent") or f"{float(value) * 100:.2f}%")
    st.metric(label, display)
    denominator = f"{metric.get('correct', 0)}/{metric.get('total', 0)}"
    interval = metric.get("ci95") or {}
    interval_text = interval.get("percent") if interval else "not available"
    st.caption(f"{denominator} · 95% CI: {interval_text}")


def _render_evaluation_metrics(evaluation: dict[str, Any]) -> None:
    metrics = evaluation.get("metrics") or {}
    st.write(
        f"Rows scored: **{metrics.get('row_count', 0)}** · "
        f"real: **{metrics.get('real_row_count', 0)}** · "
        f"synthetic: **{metrics.get('synthetic_row_count', 0)}**"
    )
    st.caption("Certification metrics use real held-out rows only. Synthetic rows are regression evidence, not production evidence.")
    first, second, third = st.columns(3)
    with first:
        _render_rate_metric("Real filled-cell accuracy", metrics.get("real_filled_cell_accuracy_metric") or {})
    with second:
        _render_rate_metric("Coverage", metrics.get("coverage") or {})
    with third:
        _render_rate_metric("Real strict accuracy", metrics.get("real_strict_accuracy") or {})
    fourth, fifth, sixth = st.columns(3)
    with fourth:
        _render_rate_metric("Review rate", metrics.get("review_rate") or {})
    with fifth:
        _render_rate_metric("Abstain rate", metrics.get("abstain_rate") or {})
    with sixth:
        _render_rate_metric("Invalid Segment 3 rate", metrics.get("invalid_code_rate") or {})
    seventh, eighth, ninth = st.columns(3)
    with seventh:
        _render_rate_metric("Override rate", metrics.get("override_rate") or {})
    with eighth:
        _render_rate_metric("Auto-post precision", metrics.get("auto_post_precision") or {})
    with ninth:
        _render_rate_metric("Auto-post recall", metrics.get("auto_post_recall") or {})
    _render_rate_metric("Invoice-level exact accuracy", metrics.get("invoice_level_accuracy") or {})

    classification_metrics = metrics.get("real_classification_metrics") or {}
    st.write(
        "Macro F1: **"
        f"{classification_metrics.get('macro_f1_percent', 'Unavailable')}** · "
        "Balanced accuracy: **"
        f"{classification_metrics.get('balanced_accuracy_percent', 'Unavailable')}**"
    )
    calibration_metrics = metrics.get("calibration_metrics") or {}
    st.caption(
        "Calibrated evidence rows: "
        f"{calibration_metrics.get('evidence_count', 0)} · "
        f"ECE: {calibration_metrics.get('ece') if calibration_metrics.get('ece') is not None else 'Unavailable'} · "
        f"Brier: {calibration_metrics.get('brier_score') if calibration_metrics.get('brier_score') is not None else 'Unavailable'}"
    )
    quality_metrics = metrics.get("input_quality_metrics") or {}
    st.caption(
        "Input-quality review rows: "
        f"{quality_metrics.get('insufficient_count', 0)} · "
        f"Recall@1: {metrics.get('retrieval_type_recall_at_1_metric', {}).get('percent', 'Unavailable')} · "
        f"Recall@3: {metrics.get('retrieval_type_recall_at_3_metric', {}).get('percent', 'Unavailable')} · "
        f"Recall@10: {metrics.get('retrieval_type_recall_at_10_metric', {}).get('percent', 'Unavailable')} · "
        f"MRR: {metrics.get('real_retrieval_mrr_percent', metrics.get('retrieval_mrr_percent', 'Unavailable'))} · "
        f"nDCG@3: {metrics.get('real_retrieval_ndcg_at_3_percent', metrics.get('retrieval_ndcg_at_3_percent', 'Unavailable'))}"
    )
    baseline_comparison = metrics.get("baseline_comparison") or {}
    if baseline_comparison:
        st.write("Baseline comparison")
        st.dataframe(
            [
                {
                    "Route": name,
                    "Filled-cell accuracy": (values.get("filled_cell_accuracy_metric") or {}).get("percent", "Unavailable"),
                    "Coverage": (values.get("coverage") or {}).get("percent", "Unavailable"),
                    "Strict accuracy": (values.get("strict_accuracy") or {}).get("percent", "Unavailable"),
                }
                for name, values in baseline_comparison.items()
            ],
            hide_index=True,
        )
    per_class = classification_metrics.get("per_class") or {}
    if per_class:
        st.dataframe(
            [
                {
                    "Account type": label,
                    "Support": values.get("support", 0),
                    "Precision": values.get("precision_percent", "Unavailable"),
                    "Recall": values.get("recall_percent", "Unavailable"),
                    "F1": values.get("f1_percent", "Unavailable"),
                }
                for label, values in per_class.items()
                if values.get("support", 0)
            ],
            hide_index=True,
        )


def _oci_sdk_version() -> str:
    try:
        return version("oci-openai")
    except PackageNotFoundError:
        return "unknown"


def _render_token_usage(
    title: str,
    usage: dict[str, Any] | None,
    *,
    records: list[dict[str, Any]] | None = None,
    not_called: bool = False,
) -> None:
    if not usage or not usage.get("calls"):
        st.caption(f"{title}: {'Not applicable' if not_called else 'Not reported by OCI'}")
        return
    st.write(f"**{title}**")
    columns = st.columns(3)
    values = [
        ("Calls", usage.get("calls")),
        ("Input tokens", usage.get("input_tokens")),
        ("Output tokens", usage.get("output_tokens")),
        ("Cached tokens", usage.get("cached_tokens")),
        ("Reasoning tokens", usage.get("reasoning_tokens")),
        ("Total tokens", usage.get("total_tokens")),
    ]
    for index, (label, value) in enumerate(values):
        display = "Unavailable from OCI" if value is None else f"{int(value):,}"
        columns[index % 3].metric(label, display)
    if usage.get("output_tokens_semantics") == "may_include_reasoning_tokens":
        st.caption("Output tokens may include hidden reasoning tokens.")
    if usage.get("model") or usage.get("route"):
        st.caption(
            f"Model: `{usage.get('model') or 'unavailable'}` · "
            f"Route: `{usage.get('route') or 'unavailable'}` · "
            f"Latency: {usage.get('latency') if usage.get('latency') is not None else 'unavailable'} ms"
        )
    if usage.get("reasoning_tokens") is None and usage.get("reasoning_tokens_status"):
        st.caption(f"Reasoning-token status: {usage['reasoning_tokens_status']}")
    missing = usage.get("missing_categories") or {}
    if missing:
        st.caption("Some token categories were unavailable from the provider; no values were estimated.")
    if records:
        attempts = max(int(record.get("attempt") or 1) for record in records)
        finish_reasons = list(dict.fromkeys(
            str(record.get("finish_reason"))
            for record in records
            if record.get("finish_reason")
        ))
        st.caption(
            f"Attempts: {attempts} · Finish reason: {', '.join(finish_reasons) if finish_reasons else 'unavailable'}"
        )
    with st.expander(f"{title} diagnostics", expanded=False):
        diagnostics = {
            key: usage.get(key)
            for key in (
                "reported_calls",
                "unknown_calls",
                "missing_categories",
                "reasoning_tokens_reported",
                "reasoning_tokens_status",
                "output_tokens_semantics",
                "total_latency_ms",
            )
            if key in usage
        }
        if records:
            diagnostics["attempts"] = [record.get("attempt") for record in records]
            diagnostics["finish_reasons"] = [record.get("finish_reason") for record in records]
            diagnostics["request_ids"] = [record.get("request_id") for record in records]
            diagnostics["routes"] = [record.get("route") for record in records]
        st.json(diagnostics)


def _record_correction(result: dict[str, Any], selected_type: str, document_embedder: Any) -> None:
    feature = build_line_feature(
        {
            "LineDescription": result.get("line_description"),
            "LineType": result.get("line_type_norm"),
            "LineAmount": result.get("line_amount"),
            "UnitPrice": result.get("unit_price"),
            "QuantityInvoiced": result.get("quantity_invoiced"),
        },
        result.get("vendor_name_norm") or "",
        int(result.get("line_index", 0)),
    )
    db_utils.record_override(
        identity_key=feature["identity_key"],
        line_description=feature["line_description"],
        vendor_name_norm=feature["vendor_name_norm"] or None,
        line_type_norm=feature["line_type_norm"],
        predicted_account_type=result.get("account_type"),
        corrected_account_type=selected_type,
        confidence=result.get("confidence"),
        reason="Human correction from invoice review UI",
    )
    mapping = map_account_type_to_segments(selected_type)
    if mapping:
        vector = embed_documents([{"retrieval_text": feature["retrieval_text"]}], document_embedder)[0]
        db_utils.promote_correction_to_history(feature, selected_type, vector, mapping)


def _render_case(case: dict[str, Any], rank: int | None = None) -> None:
    st.caption(
        f"Rank {rank if rank is not None else '—'} · {case.get('account_type') or 'Unknown'} · "
        f"{case.get('line_description') or ''}"
    )


def _render_confidence_breakdown(result: dict[str, Any], *, nested: bool = False) -> None:
    breakdown = result.get("confidence_breakdown") or {}
    with (nullcontext() if nested else st.expander("Evidence and confidence diagnostics")):
        metric_left, metric_right = st.columns(2)
        with metric_left:
            probability = result.get("calibrated_correctness_probability")
            display = "Unavailable" if probability is None else f"{float(probability) * 100:.2f}%"
            st.metric("Calibrated correctness probability", display)
        with metric_right:
            st.metric("Decision", str(result.get("decision") or "REVIEW_REQUIRED"))
        st.caption(
            "The values below are diagnostic evidence, not measured accuracy. "
            + breakdown.get("formula", "")
        )
        rows = [
            {
                "Signal": "Retrieval margin",
                "Value": breakdown.get("retrieval_margin", 0.0),
                "Weight": breakdown.get("retrieval_weight", 0.40),
                "Contribution": breakdown.get("retrieval_contribution", 0.0),
            },
            {
                "Signal": "Identity strength",
                "Value": breakdown.get("identity_strength", 0.0),
                "Weight": breakdown.get("identity_weight", 0.25),
                "Contribution": breakdown.get("identity_contribution", 0.0),
            },
            {
                "Signal": "Vendor agreement",
                "Value": breakdown.get("vendor_agreement", 0.0),
                "Weight": breakdown.get("vendor_weight", 0.20),
                "Contribution": breakdown.get("vendor_contribution", 0.0),
            },
            {
                "Signal": "LLM account-type signal",
                "Value": breakdown.get("llm_signal", 0.0),
                "Weight": breakdown.get("llm_weight", 0.15),
                "Contribution": breakdown.get("llm_contribution", 0.0),
            },
        ]
        st.table(rows)
        penalties = breakdown.get("penalties") or []
        if penalties:
            st.write("Penalties")
            st.table(
                [
                    {
                        "Flag": item.get("flag"),
                        "Weight": -float(item.get("weight") or 0.0),
                        "Description": item.get("description", ""),
                    }
                    for item in penalties
                ]
            )
        st.write(f"Penalty total: **{float(breakdown.get('penalty_total') or 0.0):.3f}**")
        st.write(f"Unclamped score: **{float(breakdown.get('unclamped_score') or 0.0):.3f}**")
        st.write(f"Heuristic evidence score: **{float(breakdown.get('final_confidence') or 0.0):.3f}**")
        if breakdown.get("calculation_mode") == "identity_cache_override":
            st.info(breakdown.get("override_reason", "Identity cache supplied a deterministic result."))


def _render_logprob_explanation(result: dict[str, Any], *, nested: bool = False) -> None:
    explanation = result.get("logprob_explanation") or {}
    status = explanation.get("status", "not_called")
    evidence_scope = explanation.get("evidence_scope", "selected_only")

    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    with (nullcontext() if nested else st.expander("Account-type token logprobs")):
        if status == "not_called":
            st.info("LLM was skipped by the retrieval cascade. Account-type logprobs are not applicable to this line.")
            return
        if status in {"missing", "invalid"}:
            st.warning("No usable account-type token logprobs were returned; the line remains review-safe.")
        elif status == "saturated":
            st.warning(
                "The provider returned saturated or rounded token probabilities. "
                "This evidence is not reliable enough for automatic defaulting."
            )
        elif status == "available" and evidence_scope == "selected_only":
            st.warning(
                "Selected-token logprobs are available, but complete candidate evidence was not returned. "
                "Only the selected account-type evidence is shown; it is diagnostic and cannot authorize "
                "automatic defaulting."
            )
        elif status == "available":
            st.warning(
                "Logprobs are next-token diagnostics, not a correctness probability. The line "
                "stays review-safe."
            )

        st.write(f"Status: **{status}**")
        st.write(f"Evidence scope: **{evidence_scope}**")
        if explanation.get("provider"):
            st.write(f"Provider: `{explanation['provider']}`")
        if explanation.get("request_mode"):
            st.write(f"Request mode: `{explanation['request_mode']}`")
        if explanation.get("model"):
            st.write(f"Model: `{explanation['model']}`")
        if explanation.get("finish_reason"):
            st.write(f"Finish reason: `{explanation['finish_reason']}`")
        coverage = explanation.get("candidate_coverage") or {}
        if coverage:
            st.write(
                "Candidate coverage: **"
                f"{coverage.get('observed', 0)}/{coverage.get('requested', 0)}**"
            )
        st.write(
            "Calibration artifact: **"
            f"{explanation.get('calibration_status', 'not_configured')}**"
        )
        st.write(f"Used in composite: **{'yes' if explanation.get('used_in_composite') else 'no'}**")
        st.write(f"Selected account type: **{explanation.get('selected_account_type') or 'n/a'}**")
        st.write(f"Selected token count: **{explanation.get('selected_token_count') or 0}**")
        if explanation.get("selected_loglikelihood") is not None:
            st.write(f"Selected log-likelihood: **{fmt(explanation['selected_loglikelihood'])}**")
        if explanation.get("selected_mean_logprob") is not None:
            st.write(f"Mean logprob: **{fmt(explanation['selected_mean_logprob'])}**")
        if explanation.get("selected_probability_proxy") is not None:
            st.write(
                "Probability proxy, exp(mean logprob): **"
                f"{fmt(explanation['selected_probability_proxy'])}**"
            )
            st.caption("This is a token-likelihood proxy, not a calibrated class probability.")
        if explanation.get("runner_up_account_type"):
            st.write(f"Runner-up: **{explanation['runner_up_account_type']}**")
            st.write(f"Runner-up mean logprob: **{fmt(explanation['runner_up_mean_logprob'])}**")
            st.write(f"Selected-vs-runner-up margin: **{fmt(explanation['logprob_margin'])}**")
        else:
            st.caption(
                "Runner-up comparison unavailable. The provider did not return complete candidate evidence "
                "for the competing taxonomy labels."
            )

        candidate_rows = list(explanation.get("candidate_scores") or [])
        observed_names = {str(row.get("account_type")) for row in candidate_rows}
        selected_name = str(explanation.get("selected_account_type") or "")
        if selected_name and selected_name not in observed_names:
            candidate_rows.insert(
                0,
                {
                    "account_type": selected_name,
                    "mean_logprob": explanation.get("selected_mean_logprob"),
                    "probability_proxy": explanation.get("selected_probability_proxy"),
                    "status": "selected",
                },
            )
            observed_names.add(selected_name)
        requested_candidates = explanation.get("requested_candidate_types") or result.get("candidate_types") or []
        for candidate in requested_candidates:
            if candidate not in observed_names:
                candidate_rows.append(
                    {
                        "account_type": candidate,
                        "mean_logprob": None,
                        "probability_proxy": None,
                        "status": "not_returned",
                    }
                )
        if candidate_rows:
            st.write("Candidate evidence")
            st.table(
                [
                    {
                        "Account type": row.get("account_type"),
                        "Mean logprob": fmt(row.get("mean_logprob")),
                        "Probability proxy": fmt(row.get("probability_proxy")),
                        "Normalized share": fmt(row.get("normalized_share")),
                        "Evidence": row.get("status", "not_returned"),
                    }
                    for row in candidate_rows
                ]
            )

        tokens = explanation.get("tokens") or []
        if tokens:
            st.write("Selected account-type tokens")
            st.table(
                [
                    {
                        "Token": token.get("token", ""),
                        "Logprob": fmt(token.get("logprob")),
                        "Probability proxy": fmt(token.get("probability")),
                        "Selected": "Yes",
                        "Top allowed alternatives": ", ".join(
                            f"{item.get('token')}: {fmt(item.get('probability'))}"
                            for item in token.get("top_alternatives", [])
                        ) or "none",
                    }
                    for token in tokens
                ]
            )


def _render_line(result: dict[str, Any], document_embedder: Any) -> None:
    index = int(result.get("line_index", 0))
    description = result.get("line_description") or "(missing description)"
    with st.container(border=True):
        st.subheader(f"Line {index + 1}: {description}")
        left, right = st.columns(2)
        with left:
            st.write(f"Account type: **{result.get('account_type', 'Unknown')}**")
            st.write(f"Segment 3: **{result.get('segment3') or ''}**")
            st.write(f"Decision: **{result.get('decision', 'REVIEW_REQUIRED')}**")
        if result.get("calibrated_correctness_probability") is not None:
            st.write(
                "Calibrated correctness probability: **"
                f"{float(result['calibrated_correctness_probability']) * 100:.2f}%**"
            )
        else:
            st.caption("No calibrated correctness probability is available for this line.")
        st.caption(f"Confidence status: `{result.get('confidence_status') or 'unavailable'}`")
        if result.get("review_reasons"):
            st.warning("Review reasons: " + ", ".join(result["review_reasons"]))
        split_suggestions = result.get("split_suggestions") or {}
        if split_suggestions.get("status") == "split_suggested":
            st.warning("Split required before Segment 3 assignment. Suggestions are account natures, not GL codes.")
            st.dataframe(
                [
                    {
                        "Child description": child.get("description"),
                        "Account nature": child.get("account_nature"),
                        "Decision": child.get("decision"),
                    }
                    for child in split_suggestions.get("children") or []
                ],
                hide_index=True,
            )
            split_left, split_right = st.columns(2)
            with split_left:
                if st.button("Accept split for coding review", key=f"accept_split_{index}"):
                    decision = accept_split_suggestions(
                        split_suggestions, split_suggestions.get("children") or []
                    )
                    st.session_state[f"split_decision_{index}"] = decision
                    try:
                        persist_split_decision(
                            db_utils,
                            source_line_id=str(result.get("line_index", index)),
                            decision=decision,
                        )
                    except Exception as exc:
                        st.warning(f"Split review is visible but could not be persisted: {exc}")
            with split_right:
                if st.button("Merge split back", key=f"merge_split_{index}"):
                    decision = merge_split_suggestions(
                        split_suggestions.get("children") or []
                    )
                    st.session_state[f"split_decision_{index}"] = decision
                    try:
                        persist_split_decision(
                            db_utils,
                            source_line_id=str(result.get("line_index", index)),
                            decision=decision,
                        )
                    except Exception as exc:
                        st.warning(f"Split review is visible but could not be persisted: {exc}")
            if st.session_state.get(f"split_decision_{index}"):
                st.info("Split decision recorded for review; no Segment 3 was assigned.")
        with right:
            st.write(f"LLM: {'called' if result.get('llm_called') else 'skipped'}")
            st.write(f"LLM route: `{result.get('llm_route') or 'unknown'}`")
            if result.get("llm_skip_reason"):
                st.caption(f"LLM skip reason: {result['llm_skip_reason']}")
            if result.get("llm_failure_reason"):
                st.error(f"LLM failure: {result['llm_failure_reason']}")
            if result.get("llm_called"):
                st.write(
                    f"Provider route: `{result.get('llm_provider') or 'unknown'}` · "
                    f"`{result.get('llm_request_mode') or 'unknown'}`"
                )
            logprob_status = result.get("logprob_status") or "not_called"
            st.write(f"Line type: `{result.get('line_type_norm') or 'UNKNOWN'}`")
            top3_types = result.get("top3_candidate_types") or result.get("candidate_types") or []
            st.write(f"Top review types: {', '.join(top3_types[:3]) or 'none'}")
            st.write(f"Conflict flags: {', '.join(result.get('conflict_flags') or []) or 'none'}")
        st.info(result.get("reason") or "No reason returned.")
        ranked_candidates = result.get("ranked_candidates") or []
        if ranked_candidates:
            with st.expander("Top review candidates"):
                st.caption(
                    "Candidate ranking is evidence only. A correctness percentage remains unavailable until a compatible, held-out ranker calibration artifact is loaded."
                )
                st.dataframe(
                    [
                        {
                            "Rank": index + 1,
                            "Account type": candidate.get("account_type"),
                            "Segment 3": candidate.get("segment3"),
                            "Evidence": "hybrid retrieval candidate",
                            "Calibrated probability": (
                                "Unavailable"
                                if candidate.get("calibrated_probability") is None
                                else f"{float(candidate['calibrated_probability']) * 100:.2f}%"
                            ),
                        }
                        for index, candidate in enumerate(ranked_candidates)
                    ],
                    hide_index=True,
                )
        with st.expander("Engineering diagnostics", expanded=False):
            st.write(f"Logprob confidence source: `{result.get('logprob_confidence_source') or 'none'}`")
            if result.get("llm_called"):
                _render_token_usage(
                    "Classification token usage",
                    result.get("llm_usage_summary"),
                    records=list(result.get("llm_usage_records") or []),
                )
            else:
                _render_token_usage("Classification token usage", None, not_called=True)
            _render_confidence_breakdown(result, nested=True)
            _render_logprob_explanation(result, nested=True)
        cases = result.get("retrieved_cases") or []
        if cases:
            with st.expander("Historical evidence"):
                for rank, case in enumerate(cases[:3], start=1):
                    _render_case(case, rank)
        options = list(
            dict.fromkeys(
                [
                    "Unknown",
                    result.get("account_type") or "Unknown",
                    *(candidate.get("account_type") for candidate in ranked_candidates),
                    *(result.get("candidate_types") or []),
                ]
            )
        )
        selected = st.selectbox(
            "Correction / confirmation",
            options,
            index=options.index(result.get("account_type")) if result.get("account_type") in options else 0,
            key=f"correction_{index}",
        )
        if st.button("Save correction", key=f"save_correction_{index}", width="stretch"):
            if selected == result.get("account_type") and result.get("account_type") != "Unknown":
                st.success("Current classification confirmed.")
            else:
                try:
                    _record_correction(result, selected, document_embedder)
                    st.success("Correction saved to audit/history.")
                except Exception as exc:
                    st.error(f"Correction could not be saved: {exc}")


st.set_page_config(page_title="Segment 3 Natural Account Classifier", layout="wide")
st.title("Segment 3 Natural Account Classification")
st.caption("AP_INVOICE extraction → strict six-field projection → per-line cascade classification")
st.warning("Shadow mode — auto-post disabled. Percentages appear only after Finance-approved calibration.")
_refresh_local_classifier_environment()
active_model = os.getenv("LLM_MODEL", CLASSIFIER_MODEL_ID).strip()
active_provider = os.getenv("CLASSIFIER_PROVIDER", CLASSIFIER_PROVIDER).strip() or CLASSIFIER_PROVIDER
active_route = "oci_native_chat" if "gpt-oss" in active_model.lower() else active_provider
configuration_state = "valid"
try:
    validate_classifier_configuration()
except RuntimeError:
    configuration_state = "invalid"
st.caption(
    f"Classifier model: `{active_model or 'not configured'}` · "
    f"active route: `{active_route}` · OCI SDK: `{_oci_sdk_version()}` · "
    f"configuration: `{configuration_state}` · native reasoning: `{LLM_NATIVE_REASONING_EFFORT}` · "
    f"LLM routing: `{os.getenv('GL_LLM_ROUTING_MODE', 'always')}`"
)
required_model = os.getenv("GL_REQUIRED_CLASSIFIER_MODEL", REQUIRED_CLASSIFIER_MODEL).strip()
if active_model != required_model:
    st.error(
        f"Classifier configuration is invalid: `{active_model or 'missing'}` is configured, "
        f"but `{required_model}` is required."
    )

st.session_state.setdefault("invoice_signature", None)
st.session_state.setdefault("pipeline_result", None)
st.session_state.setdefault("evaluation_result", None)

uploaded = st.file_uploader(
    "Upload invoice PDF",
    type=["pdf"],
    max_upload_size=MAX_UPLOAD_MB,
    key="invoice_upload",
)
document_bytes = uploaded.getvalue() if uploaded is not None else None
if document_bytes is not None:
    signature = hashlib.sha256(document_bytes).hexdigest()
    if signature != st.session_state["invoice_signature"]:
        st.session_state["invoice_signature"] = signature
        st.session_state["pipeline_result"] = None

with st.form("invoice_processing_form", clear_on_submit=False):
    submitted = st.form_submit_button(
        "Extract and classify invoice",
        type="primary",
        disabled=document_bytes is None,
        width="stretch",
    )

if submitted and document_bytes is not None and uploaded is not None:
    filename = uploaded.name or "invoice.pdf"
    try:
        pipeline, document_embedder = get_runtime(_runtime_config_fingerprint())
        with st.status("Processing invoice", expanded=True) as status:
            def report_stage(stage: str) -> None:
                labels = {
                    "extracting": "Extracting with AP_INVOICE",
                    "projecting": "Parsed six-field JSON",
                    "classifying": "Classifying lines",
                    "completed": "Completed",
                }
                st.write(labels.get(stage, stage))

            pipeline.stage_callback = report_stage
            st.write("Uploaded")
            result = pipeline.extract_and_classify(document_bytes, filename=filename)
            if not result["line_count"]:
                st.warning("No line items were extracted; classification was not called.")
            status.update(label="Invoice processing completed", state="complete")
        st.session_state["pipeline_result"] = result
    except InvoicePipelineError as exc:
        st.error(f"{exc.stage.title()} failed [{exc.code}]: {exc}")
    except RuntimeError as exc:
        st.error(f"Classifier configuration failed: {exc}")
    except Exception as exc:
        st.error(f"Unexpected processing failure: {exc}")

result = st.session_state.get("pipeline_result")
if result:
    st.subheader("Sanitized extracted JSON")
    st.json(result["sanitized_extracted_json"])
    diagnostics = result.get("extraction_diagnostics") or {}
    if diagnostics:
        with st.expander("Extraction diagnostics"):
            st.json(diagnostics)
    st.subheader(f"Classification results ({result['line_count']} lines)")
    if result["lines"]:
        _, document_embedder = get_runtime(_runtime_config_fingerprint())
        for line_result in result["lines"]:
            _render_line(line_result, document_embedder)
    else:
        st.info("No classifications available.")
    with st.expander("Pipeline diagnostics"):
        st.json(
            {
                "filename": result["filename"],
                "page_count": result["page_count"],
                "extraction_model": result["extraction_model"],
                "projection_diagnostics": result["projection_diagnostics"],
                "classifier_version": result["classifier_version"],
                "classifier_capability": result.get("classifier_capability"),
                "classification_usage": result.get("classification_usage"),
                "timing_seconds": result["timing_seconds"],
            }
        )

st.divider()
with st.expander("Evaluation dashboard", expanded=False):
    st.caption(
        "Runs the frozen TEST split from gl_account_history_poc_120.xlsx. "
        "Only HISTORY rows are available to retrieval, and the evaluated row is excluded from its own lookup."
    )
    if st.button("Run frozen TEST evaluation", key="run_test_evaluation"):
        try:
            workbook_path = BASE_DIR / "gl_account_history_poc_120.xlsx"
            evaluation_rows = load_evaluation_rows(workbook_path)
            history_rows = load_evaluation_rows(workbook_path, dataset_types=("HISTORY",))
            evaluation_pipeline, evaluation_embedder = get_runtime(_runtime_config_fingerprint())

            def classify_evaluation_row(row: Any, classification_context: dict[str, Any] | None = None) -> dict[str, Any]:
                context = {"header": {}, "line_items": [{}]}
                context.update(classification_context or {})
                classification = evaluation_pipeline.classifier(
                    {
                        "VendorName": row.vendor_name or "",
                        "LineItems": [{"LineDescription": row.description, "LineType": row.line_type or "UNKNOWN"}],
                    },
                    evaluation_embedder,
                    settings=evaluation_pipeline.settings,
                    classification_context=context,
                )
                return classification.get("lines", [{}])[0]

            st.session_state["evaluation_result"] = run_evaluation(
                evaluation_rows,
                classify_evaluation_row,
                "frozen-test",
                history_rows=history_rows,
            )
        except Exception as exc:
            st.error(f"Evaluation could not be completed: {exc}")

    if st.session_state.get("evaluation_result"):
        _render_evaluation_metrics(st.session_state["evaluation_result"])
