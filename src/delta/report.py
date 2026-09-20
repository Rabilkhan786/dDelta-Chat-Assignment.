"""Render the delta as human-readable Markdown and machine-readable JSON."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.canonical.model import CanonicalDocument, DeltaEntry, DeltaType
from src.delta.compatibility import CompatibilityResult
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class DeltaReportGenerator:
    """Build one report and write it to the configured JSON and Markdown paths."""

    def __init__(self, json_path: Path, markdown_path: Path) -> None:
        self.json_path = json_path
        self.markdown_path = markdown_path
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        old_doc: CanonicalDocument,
        new_doc: CanonicalDocument,
        deltas: list[DeltaEntry],
        compatibility: CompatibilityResult | None = None,
    ) -> dict:
        """Build the report and write both configured artifacts."""
        with stage(logger, "delta_report_generation"):
            report = self._build_report(old_doc, new_doc, deltas, compatibility)
            self._write_markdown(report)
            self._write_json(report)
        return report

    def _build_report(
        self,
        old_doc: CanonicalDocument,
        new_doc: CanonicalDocument,
        deltas: list[DeltaEntry],
        compatibility: CompatibilityResult | None,
    ) -> dict:
        counts = Counter(delta.change_type.value for delta in deltas)
        changes = [delta for delta in deltas if delta.change_type != DeltaType.UNCHANGED]

        report = {
            "documents": {
                "old": old_doc.metadata.file_name,
                "new": new_doc.metadata.file_name,
                "pid_a": old_doc.metadata.pid,
                "pid_b": new_doc.metadata.pid,
                "old_revision": old_doc.metadata.revision,
                "new_revision": new_doc.metadata.revision,
            },
            "summary": {
                "total_entries": len(deltas),
                "actual_changes": len(changes),
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

        for index, delta in enumerate(changes, start=1):
            entry = {
                "delta_id": f"delta-{index}",
                "change_type": delta.change_type.value,
                "element_type": delta.element_type.value,
                "page_number": delta.page_number,
                "location_revision": (
                    "A" if delta.change_type == DeltaType.REMOVED else "B"
                ),
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
        documents = report["documents"]

        lines = [
            "# Delta Report",
            "",
            "## Document Information",
            "",
            f"- PID A: {documents['pid_a']} ({documents['old']})",
            f"- PID B: {documents['pid_b']} ({documents['new']})",
            "",
        ]

        compatibility = report.get("revision_compatibility")
        if compatibility and compatibility["warning"]:
            lines.extend(
                [
                    "## Compatibility Warning",
                    "",
                    compatibility["warning"],
                    "",
                ]
            )

        lines.extend(
            [
                "## Summary",
                "",
                f"- Compared entries: {summary['total_entries']}",
                f"- Actual changes: {summary['actual_changes']}",
                f"- Unchanged: {summary['unchanged']}",
                f"- Modified: {summary['modified']}",
                f"- Added: {summary['added']}",
                f"- Removed: {summary['removed']}",
                f"- Moved: {summary['moved']}",
                "",
                "---",
                "",
                "## Changes",
                "",
            ]
        )

        if not report["entries"]:
            lines.extend(["No meaningful changes were detected.", ""])

        for entry in report["entries"]:
            lines.extend(
                [
                    f"### {entry['delta_id']}",
                    "",
                    f"- Type: {entry['change_type']}",
                    f"- Element: {entry['element_type']}",
                    f"- Revision location: {entry['location_revision']}",
                    f"- Page: {entry['page_number']}",
                    f"- Confidence: {entry['confidence']:.2f}",
                    f"- Description: {entry['description']}",
                ]
            )
            if "bounding_box" in entry:
                bbox = entry["bounding_box"]
                lines.append(
                    "- Bounding Box: "
                    f"({bbox['x0']:.2f}, {bbox['y0']:.2f}) -> "
                    f"({bbox['x1']:.2f}, {bbox['y1']:.2f})"
                )
            lines.append("")

        self._write_text(self.markdown_path, "\n".join(lines))

    def _write_json(self, report: dict) -> None:
        self._write_text(
            self.json_path,
            json.dumps(report, indent=2, sort_keys=True),
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        """Write LF-normalized report output on every operating system."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(content)
