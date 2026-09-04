from __future__ import annotations

import json
import logging
import time

import streamlit as st

try:
    from .classification import classify_line
    from .config import DEFAULT_TOP_K, MIN_SIMILARITY_SCORE
    from .embeddings import create_document_embedder, load_oci_settings
except ImportError:  # Supports running the file directly from this folder.
    from classification import classify_line
    from config import DEFAULT_TOP_K, MIN_SIMILARITY_SCORE
    from embeddings import create_document_embedder, load_oci_settings


LOGGER = logging.getLogger(__name__)


@st.cache_resource
def get_runtime():
    settings = load_oci_settings()
    return settings, create_document_embedder(settings)


def status_pill(label: str, state: str, placeholder: object, elapsed: float | None = None) -> None:
    colors = {"ready": "#64748b", "running": "#d97706", "success": "#15803d", "error": "#b91c1c"}
    state_text = {"ready": "Ready", "running": "Running", "success": "Succeeded", "error": "Failed"}.get(state, "Ready")
    timing = f"<span style='color:#cbd5e1;font-size:.78rem'>{elapsed:.2f}s</span>" if elapsed is not None else ""
    placeholder.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;padding:12px 12px;margin:6px 0;border:1px solid #64748b;border-radius:7px'>"
        f"<span style='font-weight:600;color:#f8fafc'>{label}</span>{timing}"
        f"<span style='color:white;background:{colors.get(state, colors['ready'])};padding:3px 10px;border-radius:999px;font-size:.78rem'>{state_text}</span></div>",
        unsafe_allow_html=True,
    )

def render_cases(cases: list[dict[str, object]]) -> None:
    st.subheader("Retrieved GL history cases")
    for rank, case in enumerate(cases, start=1):
        score = float(case.get("similarity_score") or 0.0)
        with st.expander(f"{rank}. {case.get('line_description')} — similarity {score:.4f}", expanded=rank == 1):
            if case.get("lower_similarity"):
                st.markdown(
                    "<span style='display:inline-block;color:white;background:#b91c1c;padding:4px 10px;border-radius:999px;font-size:.78rem;font-weight:600'>Lower similarity</span>",
                    unsafe_allow_html=True,
                )
            left, right = st.columns(2)
            with left:
                st.write(f"**Account type:** {case.get('account_type')}")
                st.write(f"**GL code:** {case.get('gl_code')}")
                st.write(f"**Invoice:** {case.get('invoice_num')}")
            with right:
                st.write(f"**Similarity:** {score:.6f}")
                st.write(f"**Segments:** {'.'.join(str(case.get(f'segment{i}') or '') for i in range(1, 7))}")
            st.write(f"**Historical line:** {case.get('line_description')}")


def main() -> None:
    st.set_page_config(page_title="Invoice GL Account Classification POC", page_icon="🧾", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #ffffff !important;
        }
        input:disabled,
        textarea:disabled {
            -webkit-text-fill-color: #ffffff !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }
        /* Keep the input and pipeline areas aligned with the same vertical size. */
        div[data-testid="stHorizontalBlock"]:has(textarea[placeholder*="Ergonomic keyboard"]) > div[data-testid="stColumn"] {
            min-height: 400px;
            box-sizing: border-box;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    header_col, status_col = st.columns([2.7, 1.3], gap="large")
    with header_col:
        st.title("Invoice GL Account Classification POC")
        line_description = st.text_area(
            "Invoice GL account line description",
            placeholder="Example: Ergonomic keyboard and monitor for workstation setup",
            height=120,
        )
        classify = st.button("Classify account type", type="primary", use_container_width=True)
    with status_col:
        st.subheader("Pipeline status")
        embedding_pill, retrieval_pill, llm_pill, mapping_pill = st.empty(), st.empty(), st.empty(), st.empty()
        status_pill("Embedding generation", "ready", embedding_pill)
        status_pill("Vector similarity search", "ready", retrieval_pill)
        status_pill("Account-type LLM", "ready", llm_pill)
        status_pill("GL-code mapping", "ready", mapping_pill)

    if not classify:
        return
    if not line_description.strip():
        st.error("Enter an invoice line description before classifying.")
        return

    try:
        settings, embedder = get_runtime()
        stage_started: dict[str, float] = {}
        placeholders = {
            "embedding": embedding_pill,
            "retrieval": retrieval_pill,
            "llm": llm_pill,
            "mapping": mapping_pill,
        }

        def update_stage(stage: str) -> None:
            state, name = stage.rsplit("_", 1)
            labels = {"embedding": "Embedding generation", "retrieval": "Vector similarity search", "llm": "Account-type LLM", "mapping": "GL-code mapping"}
            if name == "running":
                stage_started[state] = time.perf_counter()
                status_pill(labels[state], "running", placeholders[state])
            else:
                elapsed = time.perf_counter() - stage_started.get(state, time.perf_counter())
                status_pill(labels[state], "success", placeholders[state], elapsed)

        result = classify_line(
            line_description,
            embedder,
            settings,
            top_k=DEFAULT_TOP_K,
            stage_callback=update_stage,
        )
        if result.get("qualifying_case_count") == 0:
            st.warning(
                f"No historical case reached the {MIN_SIMILARITY_SCORE:.2f} threshold. "
                "The five nearest historical cases were sent as weak references; the LLM must validate semantic fit rather than copy them."
            )
        render_cases(result.pop("retrieved_cases"))
        st.header("Segment 3 classification")
        first, second = st.columns(2, gap="large")
        with first:
            st.text_input("Account type", value=str(result["account_type"]), disabled=True)
            if result["account_type"] == "Unknown":
                st.text_input("Inferred account type", value=str(result.get("inferred_account_type") or ""), disabled=True)
            st.text_input("Mapping source", value=str(result["mapping_source"]), disabled=True)
        with second:
            st.number_input("Confidence", min_value=0.0, max_value=1.0, value=float(result["confidence"]), disabled=True)
            st.text_area("Reason", value=str(result["reason"]), height=100, disabled=True)
        st.subheader("Mapped account segments")
        segment_values = {
            "Segment 1": "101",
            "Segment 2": "10",
            "Segment 3": str(result.get("segment3") or "Unmapped"),
            "Segment 4": "000",
            "Segment 5": "000",
            "Segment 6": "000",
        }
        segment_columns = st.columns(6, gap="small")
        for column, (label, value) in zip(segment_columns, segment_values.items()):
            with column:
                st.text_input(label, value=value, disabled=True, key=f"output_{label}")

        st.text_input("GL code", value=str(result["gl_code"] or "Unmapped"), disabled=True)
        st.subheader("Result JSON")
        st.json(result)
    except Exception as exc:
        LOGGER.exception("GL account classification failed")
        for label, pill in (("Embedding generation", embedding_pill), ("Vector similarity search", retrieval_pill), ("Account-type LLM", llm_pill), ("GL-code mapping", mapping_pill)):
            status_pill(label, "error", pill)
        st.error(f"Classification failed: {exc}")


if __name__ == "__main__":
    main()
