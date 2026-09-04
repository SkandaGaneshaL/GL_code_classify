from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from openpyxl.styles import Alignment, Font

try:
    from .config import DEFAULT_XML_PATH, BASE_DIR
except ImportError:  # Supports running the file directly from this folder.
    from config import DEFAULT_XML_PATH, BASE_DIR


DEFAULT_EXCEL_PATH = BASE_DIR / "gl_account_history.xlsx"


def convert_xml_to_excel(xml_path: Path, excel_path: Path) -> int:
    """Export every XML G_1 field as text without numeric/date coercion."""
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML file was not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    records = root.findall(".//G_1")
    if not records:
        raise ValueError(f"No G_1 records found in {xml_path}")

    # Preserve the XML field order and keep every value as its original text.
    columns: list[str] = []
    for record in records:
        for child in record:
            if child.tag not in columns:
                columns.append(child.tag)

    rows = []
    for record in records:
        rows.append({column: record.findtext(column) or "" for column in columns})

    frame = pd.DataFrame(rows, columns=columns, dtype=object)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="G1 Records")
        worksheet = writer.sheets["G1 Records"]

        for cell in worksheet[1]:
            cell.font = Font(bold=True)

        # Explicitly mark every data cell as text. This preserves values such as
        # 0001, 000, account codes, invoice numbers, and ISO date strings.
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.number_format = "@"
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        for column_cells in worksheet.columns:
            letter = column_cells[0].column_letter
            longest = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[letter].width = min(max(longest + 2, 12), 60)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all XML G_1 fields to Excel as text without value coercion"
    )
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--excel-path", type=Path, default=DEFAULT_EXCEL_PATH)
    args = parser.parse_args()

    count = convert_xml_to_excel(args.xml_path, args.excel_path)
    print(f"Rows exported: {count}")
    print(f"Excel file created: {args.excel_path}")


if __name__ == "__main__":
    main()
