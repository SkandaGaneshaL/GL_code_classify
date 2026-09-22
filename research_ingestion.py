"""Audit every supplied research URL without treating source text as instructions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx


URL_PATTERN = re.compile(
    r"https?://[^\s\)\]>\"']+(?:\r?\n\s*(?!https?://)[A-Za-z0-9][^\s\)\]>\"']*)*"
)


def _clean_url_token(value: str) -> str:
    """Join Markdown line-wrapped URLs and remove surrounding punctuation."""
    cleaned = re.sub(r"\s+", "", str(value or ""))
    cleaned = cleaned.strip("<>[](){}\u0060\"'")
    return cleaned.rstrip(".,;:!\u3002\u0060")


def _canonical_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or "<" in parsed.netloc:
        return None
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def build_url_inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        for raw_url in URL_PATTERN.findall(content):
            raw_url = _clean_url_token(raw_url)
            canonical = _canonical_url(raw_url)
            key = canonical or raw_url
            entry = entries.setdefault(
                key,
                {
                    "url": raw_url,
                    "canonical_url": canonical,
                    "source_files": [],
                    "source_count": 0,
                    "status": "queued" if canonical else "invalid_url",
                    "reviewed": False,
                    "failure_reason": None if canonical else "invalid_url",
                },
            )
            entry["source_files"].append(str(path))
            entry["source_count"] += 1
    for entry in entries.values():
        entry["source_files"] = sorted(set(entry["source_files"]))
    return sorted(entries.values(), key=lambda item: item["url"])


def _evidence_grade(final_url: str, status_code: int) -> str:
    host = urlsplit(final_url).netloc.lower()
    if status_code != 200:
        return "unavailable"
    if host.endswith(("oracle.com", "openai.com", "nanonets.com", "snowfox.ai", "highradius.com")):
        return "primary_vendor"
    if host.endswith((".gov", ".edu", "arxiv.org", "aclanthology.org")):
        return "primary_research"
    return "secondary_or_index"


def fetch_inventory(
    inventory: list[dict[str, Any]], *, timeout_seconds: float = 20.0, retries: int = 2
) -> list[dict[str, Any]]:
    """Fetch all valid sources with explicit failures; never bypass access controls."""
    audit_run_id = str(uuid.uuid4())
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds, headers={"User-Agent": "GL-Classification-Research-Audit/1.0"}) as client:
        for entry in inventory:
            if entry["status"] != "queued":
                continue
            response: httpx.Response | None = None
            error: str | None = None
            for attempt in range(1, retries + 2):
                try:
                    response = client.get(entry["canonical_url"])
                    if response.status_code < 500:
                        break
                except httpx.HTTPError as exc:
                    error = type(exc).__name__
                if attempt <= retries:
                    time.sleep(min(2**(attempt - 1), 4))
            entry["audit_run_id"] = audit_run_id
            entry["attempts"] = attempt
            entry["retry_count"] = max(0, attempt - 1)
            entry["fetched_at"] = datetime.now(timezone.utc).isoformat()
            if response is None:
                entry.update(
                    {
                        "status": "fetch_error",
                        "error": error or "unknown_error",
                        "failure_reason": error or "unknown_error",
                        "evidence_grade": "unavailable",
                        "reviewed": False,
                    }
                )
                continue
            content = response.text
            failure_reason = None if response.status_code == 200 else f"http_{response.status_code}"
            entry.update(
                {
                    "status": "fetched" if response.status_code == 200 else "http_error",
                    "http_status": response.status_code,
                    "final_url": str(response.url),
                    "content_type": response.headers.get("content-type"),
                    "content_sha256": hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest(),
                    "content_length": len(content),
                    "evidence_grade": _evidence_grade(str(response.url), response.status_code),
                    "failure_reason": failure_reason,
                    "reviewed": response.status_code == 200,
                }
            )
    return inventory


def summarize_audit(inventory: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report source-audit outcomes without promoting failed URLs to evidence."""
    entries = list(inventory)
    status_counts: dict[str, int] = defaultdict(int)
    reviewed = 0
    for entry in entries:
        status = str(entry.get("status") or "unknown")
        status_counts[status] += 1
        if status == "fetched" and int(entry.get("http_status") or 0) == 200:
            reviewed += 1
    return {
        "total_urls": len(entries),
        "reviewed_urls": reviewed,
        "unreviewed_urls": len(entries) - reviewed,
        "status_counts": dict(sorted(status_counts.items())),
        "evidence_reviewed_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Markdown or text files containing URLs")
    parser.add_argument("--output", required=True, type=Path, help="Audit JSON output")
    parser.add_argument("--fetch", action="store_true", help="Fetch queued URLs after building the inventory")
    args = parser.parse_args()
    inventory = build_url_inventory(args.sources)
    if args.fetch:
        fetch_inventory(inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**summarize_audit(inventory), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
