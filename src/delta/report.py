"""Render the delta as a human-readable Markdown report and a machine-readable JSON report."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.canonical.model import CanonicalDocument, DeltaEntry
from src.delta.compatibility import CompatibilityResult
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class DeltaReportGenerator:
    """Builds one report dict, then writes it as both Markdown and JSON."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, old_doc: CanonicalDocument, new_doc: CanonicalDocument,
                 deltas: list[DeltaEntry], compatibility: CompatibilityResult | None = None) -> dict:
        """Build the report and write it to disk. Returns the dict for indexing/eval."""
        with stage(logger, "delta_report_generation"):
            report = self._build_report(old_doc, new_doc, deltas, compatibility)
            self._write_markdown(report)
            self._write_json(report)
        return report

    def _build_report(self, old_doc: CanonicalDocument, new_doc: CanonicalDocument,
                      deltas: list[DeltaEntry], compatibility: CompatibilityResult | None) -> dict:
        counts = Counter(delta.change_type.value for delta in deltas)
        report = {
            "documents": {
                "old": old_doc.metadata.file_name,
                "new": new_doc.metadata.file_name,
                "old_revision": old_doc.metadata.revision,
                "new_revision": new_doc.metadata.revision,
                "pid": old_doc.metadata.pid,
            },
            "summary": {
                "total_entries": len(deltas),
                "actual_changes": counts.get("modified", 0) + counts.get("moved", 0) + counts.get("added", 0) + counts.get("removed", 0),
                "unchanged": counts.get("unchanged", 0),
                "modified": counts.get("modified", 0),
                "added": counts.get("added", 0),
                "removed": counts.get("removed", 0),
                "moved": counts.get("moved", 0),
            },
            "entries": [],
        }
        if compatibility:
            report["revision_compatibility"] = {
                "score": compatibility.score,
                "compatible": compatibility.compatible,
                "warning": compatibility.message,
            }

        for delta in deltas:
            entry = {
                "change_type": delta.change_type.value,
                "element_type": delta.element_type.value,
                "page_number": delta.page_number,
                "confidence": round(delta.confidence, 2),
                "description": delta.description,
                "element_id": delta.element_id,
                "previous_element_id": delta.previous_element_id,
                "location_changed": delta.location_changed,
            }
            if delta.region:
                entry["bounding_box"] = delta.region.model_dump()
            report["entries"].append(entry)

        return report

    def _write_markdown(self, report: dict) -> None:
        summary = report["summary"]
        lines = [
            "# Delta Report", "",
            "## Document Information", "",
            f"Old Document : {report['documents']['old']}", "",
            f"New Document : {report['documents']['new']}", "",
            "## Summary", "",
            f"- Total Entries  : {summary['total_entries']}",
            f"- Actual Changes : {summary['actual_changes']}", "",
            f"- Unchanged : {summary['unchanged']}",
            f"- Modified  : {summary['modified']}",
            f"- Added     : {summary['added']}",
            f"- Removed   : {summary['removed']}", "",
            f"- Moved     : {summary['moved']}", "",
            "---", "",
            "## Delta Entries", "",
        ]
        compatibility = report.get("revision_compatibility")
        if compatibility and compatibility["warning"]:
            lines[4:4] = ["## Compatibility warning", "", compatibility["warning"], ""]

        for index, entry in enumerate(report["entries"], start=1):
            lines += [
                f"### Entry {index}", "",
                f"- Type : {entry['change_type']}",
                f"- Element : {entry['element_type']}",
                f"- Page : {entry['page_number']}",
                f"- Confidence : {entry['confidence']:.2f}",
                f"- Description : {entry['description']}",
            ]
            if "bounding_box" in entry:
                bbox = entry["bounding_box"]
                lines.append(f"- Bounding Box : ({bbox['x0']:.2f}, {bbox['y0']:.2f}) → ({bbox['x1']:.2f}, {bbox['y1']:.2f})")
            lines.append("")

        self._write_text("delta_report.md", "\n".join(lines))

    def _write_json(self, report: dict) -> None:
        self._write_text("delta_report.json", json.dumps(report, indent=2, sort_keys=True))

    def _write_text(self, file_name: str, content: str) -> None:
        """Keep generated reports LF-normalized on every operating system."""
        with (self.output_dir / file_name).open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(content)
