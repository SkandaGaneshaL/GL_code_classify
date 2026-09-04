from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GLHistoryRecord:
    source_key: str
    invoice_num: str
    invoice_date: str | None
    invoice_line_number: int | None
    distribution_line_number: int | None
    line_description: str
    line_type: str | None
    account_type: str
    gl_code: str
    segment1: str | None
    segment2: str | None
    segment3: str | None
    segment4: str | None
    segment5: str | None
    segment6: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(element: ET.Element, *names: str) -> str | None:
    for name in names:
        value = element.findtext(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _integer(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Expected an integer, received {value!r}") from exc


def _source_key(values: list[str | None]) -> str:
    payload = "\x1f".join(value or "" for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_gl_history(xml_path: Path, skip_empty_descriptions: bool = False) -> list[GLHistoryRecord]:
    """Read XML G_1 rows while preserving codes and leading zeroes as text."""
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML dataset was not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    elements = root.findall(".//G_1")
    if not elements:
        raise ValueError(f"No G_1 records found in {xml_path}")

    records: list[GLHistoryRecord] = []
    skipped = 0
    for row_number, element in enumerate(elements, start=1):
        invoice_num = _text(element, "INVOICE_NUM")
        description = _text(element, "LINE_DESCRIPTION", "LINE_ITEM_DESCRIPTION")
        account_type = _text(element, "SEGMENT3_DESCRIPTION")
        gl_code = _text(element, "GL_CODE")
        if not invoice_num or not gl_code or not account_type:
            raise ValueError(
                f"XML row {row_number} must contain INVOICE_NUM, GL_CODE, and SEGMENT3_DESCRIPTION"
            )
        if not description:
            if skip_empty_descriptions:
                skipped += 1
                continue
            raise ValueError(f"XML row {row_number} has an empty line description")

        invoice_date = _text(element, "INVOICE_DATE")
        line_number = _text(element, "INVOICE_LINE_NUMBER")
        distribution_number = _text(element, "DISTRIBUTION_LINE_NUMBER")
        line_type = _text(element, "LINE_TYPE")
        segments = [_text(element, f"SEGMENT{i}") for i in range(1, 7)]
        key = _source_key(
            [
                invoice_num,
                invoice_date,
                line_number,
                distribution_number,
                description,
                account_type,
                gl_code,
                *segments,
            ]
        )
        records.append(
            GLHistoryRecord(
                source_key=key,
                invoice_num=invoice_num,
                invoice_date=invoice_date,
                invoice_line_number=_integer(line_number),
                distribution_line_number=_integer(distribution_number),
                line_description=description,
                line_type=line_type,
                account_type=account_type,
                gl_code=gl_code,
                segment1=segments[0],
                segment2=segments[1],
                segment3=segments[2],
                segment4=segments[3],
                segment5=segments[4],
                segment6=segments[5],
            )
        )

    if skipped:
        LOGGER.warning("Skipped %d XML rows with empty descriptions", skipped)
    if not records:
        raise ValueError("No usable GL history rows were found")
    LOGGER.info("Loaded %d GL history rows from %s", len(records), xml_path)
    return records

