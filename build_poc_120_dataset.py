"""Create a 120-row Phase 1 POC dataset from the 64 real curated rows.

The source rows remain unchanged. 56 generated rows are deliberately marked
as synthetic and retain a pointer to the real invoice-distribution row used as
their context. All source columns are read and written as text so GL segments
such as 000 remain unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "gl_account_history.xlsx"
OUTPUT_PATH = BASE_DIR / "gl_account_history_poc_120.xlsx"


# Four controlled variations for fourteen accounts. The first three become
# retrieval history; the fourth becomes an unseen synthetic test description.
SYNTHETIC_DESCRIPTIONS: dict[str, list[str]] = {
    "15910": [
        "Capital purchase of office workstation",
        "Desktop computer equipment acquisition",
        "Ergonomic computer monitor for employee workstation",
        "Purchase of business laptop and docking station",
    ],
    "22190": [
        "Monthly lease payment for Miami office asset",
        "Rental payment for Los Angeles equipment lease",
        "Lease settlement for Global Operations asset",
        "Scheduled payment for leased office equipment",
    ],
    "22210": [
        "Protective case for company tablet device",
        "Leather cover for corporate tablet",
        "Screen protection accessory for tablet",
        "Tablet device carrying case purchase",
    ],
    "24220": [
        "Office multifunction printer purchase",
        "Network server hardware order",
        "Colour laser printer for administration team",
        "Business printer and server equipment invoice",
    ],
    "25400": [
        "Withholding tax for consulting service invoice",
        "Consultant payment withholding tax adjustment",
        "Tax withheld from ABC Consulting payment",
        "Statutory withholding on professional services",
    ],
    "59110": [
        "Project implementation labour charge",
        "Development study contractor labour hours",
        "SCM project consulting work performed",
        "Professional labour cost for project delivery",
    ],
    "60514": [
        "Vehicle mileage for client site visit",
        "Employee travel mileage reimbursement",
        "Mileage claim for travel to client office",
        "Business driving mileage expense",
    ],
    "60521": [
        "Client meeting lunch expense",
        "Breakfast during business travel",
        "Dinner provided for customer meeting",
        "Business meal with project team",
    ],
    "60530": [
        "Hotel stay for business trip",
        "Two-night accommodation for site visit",
        "Overnight hotel booking for study meeting",
        "Lodging expense during project travel",
    ],
    "60540": [
        "General miscellaneous business service",
        "Initial legal consultation service",
        "Office printer and peripheral purchase",
        "Miscellaneous operational expense",
    ],
    "63611": [
        "Company labour cost for well construction",
        "Labour charge for compressor installation",
        "Workforce cost for pipe installation work",
        "Operational labour expense for field construction",
    ],
    "63612": [
        "Monthly equipment rental for server hardware",
        "Rental charge for field storage tanks",
        "Equipment hire expense for operations",
        "Lease cost for operational machinery",
    ],
    "63613": [
        "Electricity utility charge for March",
        "Monthly power expense for facility",
        "Utility bill for electricity consumption",
        "Electricity service charge for operations",
    ],
    "65600": [
        "External consulting services provided",
        "Professional contractor cleaning service",
        "Consulting support for operational work",
        "Third-party contractor service expense",
    ],
}


def save_as_text_excel(frame: pd.DataFrame, path: Path) -> None:
    """Save every value as text to preserve IDs and leading zeros."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="POC Dataset")
        worksheet = writer.sheets["POC Dataset"]
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.number_format = "@"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for column_cells in worksheet.columns:
            letter = column_cells[0].column_letter
            longest = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[letter].width = min(max(longest + 2, 12), 60)


def main() -> None:
    real_rows = pd.read_excel(SOURCE_PATH, dtype=str).fillna("")
    if len(real_rows) != 64:
        raise ValueError(f"Expected 64 real rows, found {len(real_rows)}")

    real_rows["IS_SYNTHETIC"] = "N"
    real_rows["SOURCE_INVOICE_DISTRIBUTION_ID"] = ""
    real_rows["SYNTHETIC_TYPE"] = ""

    generated_rows: list[dict[str, str]] = []
    for segment3, descriptions in SYNTHETIC_DESCRIPTIONS.items():
        account_rows = real_rows.loc[real_rows["SEGMENT3"] == segment3]
        if account_rows.empty:
            raise ValueError(f"Account {segment3} is absent from the real dataset")
        source_row = account_rows.iloc[0].to_dict()
        for sequence, description in enumerate(descriptions, start=1):
            row = source_row.copy()
            row["INVOICE_DISTRIBUTION_ID"] = f"SYN-{segment3}-{sequence:02d}"
            row["LINE_DESCRIPTION"] = description
            row["DATASET_TYPE"] = "HISTORY" if sequence <= 3 else "TEST"
            row["IS_SYNTHETIC"] = "Y"
            row["SOURCE_INVOICE_DISTRIBUTION_ID"] = source_row["INVOICE_DISTRIBUTION_ID"]
            row["SYNTHETIC_TYPE"] = "CONTROLLED_PARAPHRASE"
            generated_rows.append(row)

    synthetic_rows = pd.DataFrame(generated_rows, columns=real_rows.columns)
    output = pd.concat([real_rows, synthetic_rows], ignore_index=True)
    output = output.sort_values(
        ["SEGMENT3", "DATASET_TYPE", "IS_SYNTHETIC", "INVOICE_DISTRIBUTION_ID"],
        kind="stable",
    ).reset_index(drop=True)

    if len(output) != 120:
        raise ValueError(f"Expected 120 rows, created {len(output)}")
    counts = output.groupby("DATASET_TYPE").size().to_dict()
    if counts != {"HISTORY": 90, "TEST": 30}:
        raise ValueError(f"Unexpected split: {counts}")
    if output["LINE_DESCRIPTION"].str.upper().str.strip().duplicated().any():
        raise ValueError("Generated dataset contains duplicate line descriptions")

    save_as_text_excel(output, OUTPUT_PATH)
    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")
    print(output.groupby(["DATASET_TYPE", "IS_SYNTHETIC"]).size().to_string())


if __name__ == "__main__":
    main()
