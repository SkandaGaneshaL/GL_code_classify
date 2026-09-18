# GL account-type classification POC

This is a separate POC from the department-classification flow. It uses the supplied XML history and treats `SEGMENT3_DESCRIPTION` as the historical account type.

The local `.env` supplies the mandatory Segment 3 classifier model `openai.gpt-oss-20b`, keeps the embedding model configurable, and inherits the parent POC database and wallet settings. AP_INVOICE extraction remains independently configured. OCI LLM authentication uses the explicitly resolved OCI configuration path and profile.

The runtime flow is a per-invoice, per-line cascade. The public classifier projection accepts only `LineDescription`, `LineType`, `VendorName` or `PayeeName`, `LineAmount`, `UnitPrice`, and `QuantityInvoiced` inside `LineItems`. The extraction adapter may retain a bounded internal context pack—addresses, country, currency, purchase/project references, invoice totals, tax values, and sibling lines—but those fields are never added to the public six-field payload. Empty or unsupported lines return `Unknown` with blank Segment 3.

The Streamlit workflow accepts a PDF upload and reuses the read-only extraction engine from:

`C:\Users\Skanda Ganesha L\Downloads\AP_INVOICE`

The adapter dynamically loads that repository under an isolated module namespace and supplies an in-memory extraction rule set containing only `VendorName`, `PayeeName`, and the five approved line fields. The raw extraction response is projected into a strict six-field JSON payload before classification; extraction provenance, credentials, `PONumber`, and unrelated fields never reach the classifier.

1. Normalize the six approved field groups and build the stable retrieval text.
2. Check a real-history identity cache and vendor prior.
3. Retrieve up to 15 dense OCI vector cases and 15 Oracle Text lexical cases, then fuse ranks with RRF (`k=60`). Vendor and line-type signals are soft boosts, never hard filters.
4. Skip the LLM only for strong agreement; otherwise call the native OCI Chat route for `openai.gpt-oss-20b` with one case per competing type, strict JSON-schema validation, low reasoning effort, a 256-token budget, and one 512-token retry only after length truncation.
5. Apply the fixed 16-type-to-Segment-3 mapping and the review-safe confidence gate. A heuristic score is diagnostic; automatic defaulting requires a compatible, Finance-approved calibration artifact whose held-out precision lower bound meets the configured target.

Amounts are preserved as decimals and used for consistency/unit-value reasoning, but are not embedded and are not placed into absolute currency bands because currency is outside the approved contract.

Auto-defaulting is disabled by default (`GL_AUTO_DEFAULT_ENABLED=0`) for shadow/review rollout. Enable it only after the evaluation gates are met and Finance approves the specialized-vendor scope.

## Setup

The code reuses the existing POC `.env`, OCI config, and wallet files from the parent folder. The default XML path is:

`C:\Users\tchand\Downloads\ad06361d-d790-42b9-8338-202b81aadb92.xml`

You can override it with `GL_XML_PATH` in the parent `.env` file.

Export the XML to Excel without converting codes, invoice numbers, or dates:

```powershell
python xml_to_excel.py
```

The default output is `gl_account_history.xlsx` in this folder. Use `--excel-path` to choose another output file.

Create the Oracle objects:

```powershell
cd C:\Users\tchand\Projects\AP_Invoice_Automation\Dept_Classification_POC\GL_code_classify
python setup_db.py
```

Load and embed the XML history:

```powershell
python load_history.py
```

For the Excel dataset `gl_account_history_poc_120.xlsx`, first run `upgrade_schema.sql` if the table was created from the original POC schema. Then load the 120 rows as two 60-row embedding/database batches:

```powershell
python load_history_batches.py 1 --batch-size 60 --sheet-name "POC Dataset"
python load_history_batches.py 2 --batch-size 60 --sheet-name "POC Dataset"
```

The batch loader explicitly reads the clean `POC Dataset` worksheet, excludes the malformed `Query1` worksheet, and loads `DATASET_TYPE=HISTORY` by default. Use `--dataset-type all` only for an intentional data-audit load. It embeds the exact `vendor / line_type / description` retrieval template with OCI `SEARCH_DOCUMENT`, while live queries use `SEARCH_QUERY`. Rows are upserted by a deterministic source key, so rerunning a batch does not duplicate it.

Rows are upserted by a deterministic source key, so rerunning the loader refreshes existing rows instead of duplicating them.

Run the separate UI:

```powershell
streamlit run streamlit_app.py
```

Upload a PDF, then select **Extract and classify invoice**. The UI shows the sanitized extracted JSON, one result card per line, confidence/review state, evidence, conflict flags, and correction controls. Configure `AP_INVOICE_ROOT` to point to another compatible AP_INVOICE checkout if required. Extraction settings are independent from classifier settings: `AP_EXTRACTION_MODEL_ID`, `AP_EXTRACTION_MAX_BYTES`, `AP_EXTRACTION_MAX_PAGES`, and `AP_EXTRACTION_SPLIT_PASSES` control the extraction adapter.

The command-line classifier is also available:

```powershell
python classify_line.py "Ergonomic keyboard and monitor for workstation setup"
```

Part B retrieval-only testing is available here:

```powershell
python retrieve_similar_cases.py
```

Paste a line description when prompted. It retrieves the top 3 cases by default and prints `similarity_score = 1 - cosine_distance`, along with each historical line description and account type. No LLM call is made by this script.

The LLM response is exactly `line_description`, `account_type`, `inferred_account_type`, `reason`, and `confidence`; it never emits GL codes or segment values. Segment 3 is assigned by the fixed `ACCOUNT_TYPE_TO_SEGMENT3` mapping in `classification.py`; the POC GL code is built as `101.10.<segment3>.000.000.000`.

## OCI logprob classification

Logprob collection is enabled by default in `auto` mode. `openai.gpt-oss-20b` is enforced for Segment 3 classification; a different `LLM_MODEL` fails before invoice processing and there is no silent `gpt-4o` fallback. The native OCI Chat route uses low reasoning effort, strict JSON schema output, and `log_probs`; its nested `ChatResult.chat_response` shape is normalized before parsing. The classifier parses only the generated account-type decision, never the verbalized confidence or reason text. The cached capability probe is keyed by endpoint, model, request mode, OCI SDK version, logprob mode, and cache version; a failed provisioning/capability check is surfaced as a classifier configuration failure.

Optional settings are:

```text
CLASSIFIER_PROVIDER=oci_responses
LLM_MODEL=openai.gpt-oss-20b
GL_REQUIRED_CLASSIFIER_MODEL=openai.gpt-oss-20b
GL_ENFORCE_REQUIRED_CLASSIFIER_MODEL=1
GL_ALLOW_CLASSIFIER_MODEL_FALLBACK=0
GL_CAPABILITY_PROBE_CACHE_VERSION=2
LLM_LOGPROBS_MODE=auto
LLM_LOGPROBS_TOP_K=5
LLM_LOGPROBS_REQUIRE=0
LLM_LOGPROBS_CALIBRATION_PATH=
LLM_CANDIDATE_SCORING=off
LLM_CANDIDATE_SCORING_MODE=off
LLM_GUIDED_MODE=off
GL_NATIVE_MAX_OUTPUT_TOKENS=256
GL_NATIVE_REASONING_EFFORT=LOW
LLM_COMPACT_CODE_EVIDENCE=auto
LLM_CAPABILITY_PROBE_CACHE=1
GL_AUTO_PRECISION_TARGET=0.98
GL_AUTO_MIN_ACCEPTED_VALIDATION=50
GL_AUTO_REQUIRE_CALIBRATION=1
```

Retrieved account-type candidates are encoded as one-character transport codes so native `top_logprobs` can be mapped back to account types at one output position. Candidate evidence is complete only when every requested code, including `Unknown`, is returned; missing alternatives are never assigned zero probability. Without a compatible Finance-calibrated JSON artifact, logprobs are diagnostic and self-reported model confidence is never used for automatic decisions; automatic defaulting remains disabled. Compatible calibration artifacts must identify the model, native route, code-map version, held-out sample count, validation metrics, dataset version, and (for automatic defaulting) Finance approval.

Each line card now includes **How this confidence was calculated** and **Account-type token logprobs** expanders. The first shows retrieval, identity, vendor, and LLM weighted contributions plus named penalties. The second shows the provider route/model, finish reason, evidence scope (`selected_only`, `candidate_set`, or `full_taxonomy`), candidate coverage, mapped account-type logprobs, normalized shares when coverage is complete, runner-up margin, and calibration state. Native reasoning tokens are excluded from final-answer evidence. Missing candidates remain unavailable rather than becoming zero. A line whose LLM was skipped displays `not_called` because token logprobs are not applicable. The legacy forced-label candidate scorer is not treated as a calibrated class scorer; a provider-specific neutral/teacher-forced scorer must be verified before enabling candidate scoring.

The UI also reports AP_INVOICE-style classification usage at invoice and line level: calls, input/output/total tokens, cached tokens, reasoning-token availability, missing categories, retry attempts, finish reason, route, request ID, and latency. OCI values that are not reported remain unavailable rather than zero. For gpt-oss, output tokens are explicitly marked as potentially including hidden reasoning tokens.

## Reproducibility and tests

## Review-first confidence and production controls

The classifier now distinguishes a calibrated correctness probability from
heuristic and logprob diagnostics. A heuristic value is never labelled as
accuracy, and a saturated, missing, or partial logprob response produces an
unavailable calibrated probability rather than `0.0`.

The extraction-quality gate stops compound lines that describe multiple natural
accounts before retrieval or LLM classification. Missing vendor or line type
also generates explicit review reasons. The public response includes
`prediction`, `decision`, `confidence_status`,
`calibrated_correctness_probability`, `review_reasons`, `model_version`, and
`policy_version`.

Automatic defaulting stays disabled. Future promotion requires Finance approval,
at least 200 real accepted held-out rows, and a 98% Wilson lower bound for
accepted precision. Synthetic rows never count toward that release evidence.

The separate versioned FastAPI boundary is defined in `api_service.py`; install
`requirements-production.txt` for the production observability and calibration
toolchain. The Streamlit app remains the review and reporting interface.

To create the complete URL audit from the supplied research files, run:

```powershell
python research_ingestion.py <source files...> --output research/url-inventory.json --fetch
```

The audit preserves invalid, blocked, redirected, and failed URLs rather than
silently excluding them. See `docs/accuracy-research.md` for the measurement
contract and source-evidence rules.

Dependencies are pinned in `requirements.txt`, including the AP_INVOICE runtime packages (`pydantic`, `pypdf`, `pymupdf`, OCI OpenAI compatibility, and supporting web/runtime libraries). Install them with `python -m pip install -r requirements.txt` and run `python -m pytest -q`. The tests cover allowlist enforcement, extraction projection, isolated AP adapter loading, PDF/size gates, raw-payload isolation, normalization, retrieval fusion, response validation, identity-cache skipping, software safety behavior, and Streamlit smoke rendering.

`evaluation.py` provides the leakage-safe evaluation harness. `load_evaluation_rows()` reads the frozen `TEST` split by default, while retrieval is restricted to HISTORY and the current row/source group is excluded. `run_evaluation()` / `evaluate_predictions()` report strict end-to-end accuracy, selective accuracy, coverage, accepted precision/recall, review and abstain rates, Wilson 95% intervals, macro/weighted metrics, balanced accuracy, Recall@3, calibration metrics (Brier, log loss, ECE, risk-coverage/AURC), latency, token counts, and the full 16-type confusion matrix. The Streamlit **Evaluation dashboard** runs the frozen TEST split and shows accepted precision and strict accuracy with explicit denominators.

Fit a calibration artifact only from held-out records with Finance-approved correctness labels. The command accepts a JSON array or JSONL file and keeps automatic readiness false unless the artifact has sufficient validated precision, sample support, and Finance approval:

```powershell
python calibrate_logprobs.py `
  --input heldout_predictions.jsonl `
  --output calibration.json `
  --model openai.gpt-oss-20b `
  --endpoint-route oci_native_chat `
  --request-mode structured_json `
  --code-map-version compact-code-v1 `
  --dataset-version finance-holdout-v1
```

Accepted UI corrections are written to `AG_GL_ACCOUNT_CLASSIFICATION_OVERRIDES`; mapped corrections are re-embedded in document mode and promoted as non-synthetic history rows. Use `schema.sql` for a new schema or `upgrade_schema.sql` for the original POC table.
