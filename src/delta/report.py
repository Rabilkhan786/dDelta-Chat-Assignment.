from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.canonical.model import CanonicalDocument, DeltaEntry
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class DeltaReportGenerator:
    """
    Generates human-readable (Markdown)
    and machine-readable (JSON) delta reports.
    """

    def __init__(self, output_dir: Path):

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    
    # Public API
    

    def generate(
        self,
        old_doc: CanonicalDocument,
        new_doc: CanonicalDocument,
        deltas: list[DeltaEntry],
    ) -> dict:
        """
        Generate both Markdown and JSON reports.

        Returns
        -------
        dict
            Dictionary representation of the delta report.
            This can be directly used for indexing.
        """

        with stage(logger, "delta_report_generation"):
            report = self._build_report(old_doc, new_doc, deltas)
            self._generate_markdown(report)
            self._generate_json(report)

        return report

    
    # Report Builder


    def _build_report(
        self,
        old_doc: CanonicalDocument,
        new_doc: CanonicalDocument,
        deltas: list[DeltaEntry],
    ) -> dict:

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
                "actual_changes": (
                    counts.get("modified", 0)
                    + counts.get("added", 0)
                    + counts.get("removed", 0)
                ),
                "unchanged": counts.get("unchanged", 0),
                "modified": counts.get("modified", 0),
                "added": counts.get("added", 0),
                "removed": counts.get("removed", 0),
            },
            "entries": [],
        }

        for delta in deltas:

            item = {
                "change_type": delta.change_type.value,
                "element_type": delta.element_type.value,
                "page_number": delta.page_number,
                "confidence": round(delta.confidence, 2),
                "description": delta.description,
            }

            if delta.region:
                item["bounding_box"] = {
                    "x0": delta.region.x0,
                    "y0": delta.region.y0,
                    "x1": delta.region.x1,
                    "y1": delta.region.y1,
                }

            report["entries"].append(item)

        return report

   
    # Markdown Report
    

    def _generate_markdown(
        self,
        report: dict,
    ) -> None:

        report_path = self.output_dir / "delta_report.md"

        summary = report["summary"]

        with open(report_path, "w", encoding="utf-8") as f:

            f.write("# Delta Report\n\n")

            f.write("## Document Information\n\n")
            f.write(f"Old Document : {report['documents']['old']}\n\n")
            f.write(f"New Document : {report['documents']['new']}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total Entries  : {summary['total_entries']}\n")
            f.write(f"- Actual Changes : {summary['actual_changes']}\n\n")

            f.write(f"- Unchanged : {summary['unchanged']}\n")
            f.write(f"- Modified  : {summary['modified']}\n")
            f.write(f"- Added     : {summary['added']}\n")
            f.write(f"- Removed   : {summary['removed']}\n\n")

            f.write("---\n\n")
            f.write("## Delta Entries\n\n")

            for index, entry in enumerate(report["entries"], start=1):

                f.write(f"### Entry {index}\n\n")
                f.write(f"- Type : {entry['change_type']}\n")
                f.write(f"- Element : {entry['element_type']}\n")
                f.write(f"- Page : {entry['page_number']}\n")
                f.write(f"- Confidence : {entry['confidence']:.2f}\n")
                f.write(f"- Description : {entry['description']}\n")

                if "bounding_box" in entry:

                    bbox = entry["bounding_box"]

                    f.write(
                        "- Bounding Box : "
                        f"({bbox['x0']:.2f}, {bbox['y0']:.2f}) "
                        f"→ "
                        f"({bbox['x1']:.2f}, {bbox['y1']:.2f})\n"
                    )

                f.write("\n")

   
    # JSON Report
    

    def _generate_json(
        self,
        report: dict,
    ) -> None:

        report_path = self.output_dir / "delta_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
