# GL account-type classification POC

This is a separate POC from the department-classification flow. It uses the supplied XML history and treats `SEGMENT3_DESCRIPTION` as the historical account type.

The local `.env` sets `LLM_MODEL=openai.gpt-oss-20b`, keeps the embedding model configurable, and inherits the parent POC database and wallet settings. OCI LLM authentication uses the SDK default user configuration and profile.

The runtime flow is:

1. Generate a temporary OCI embedding for the entered line description.
2. Retrieve similar GL history rows with Oracle Vector Search.
3. Ask the OCI LLM to classify only the account type.
4. Map the selected account type to `GL_CODE` and `SEGMENT1`–`SEGMENT6` using the Oracle history table.

Before the LLM call, Oracle retrieves 5 candidates. Candidates with similarity score `< 0.60` are removed, the remaining candidates are sorted by similarity descending, and only the first `GL_TOP_K` cases are included in the prompt. `GL_TOP_K=3` by default and cannot exceed 5.

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
python load_history_batches.py 1 --batch-size 60
python load_history_batches.py 2 --batch-size 60
```

The batch loader reads `LINE_DESCRIPTION` for embeddings using OCI `input_type="SEARCH_DOCUMENT"` and stores the remaining dataset fields, including invoice identifiers, vendor identifiers, account segments, account type, GL code, and synthetic/history flags. Rows are upserted by a deterministic source key, so rerunning a batch does not duplicate it.

Rows are upserted by a deterministic source key, so rerunning the loader refreshes existing rows instead of duplicating them.

Run the separate UI:

```powershell
streamlit run streamlit_app.py
```

The command-line classifier is also available:

```powershell
python classify_line.py "Ergonomic keyboard and monitor for workstation setup"
```

Part B retrieval-only testing is available here:

```powershell
python retrieve_similar_cases.py
```

Paste a line description when prompted. It retrieves the top 3 cases by default and prints `similarity_score = 1 - cosine_distance`, along with each historical line description and account type. No LLM call is made by this script.

The LLM response contains `account_type`, `inferred_account_type`, `reason`, and `confidence`. Segment 3 is then assigned by the fixed `ACCOUNT_TYPE_TO_SEGMENT3` mapping in `classification.py`; the LLM and retrieved cases do not choose the Segment 3 code. The POC GL code is built as `101.10.<segment3>.000.000.000`.
