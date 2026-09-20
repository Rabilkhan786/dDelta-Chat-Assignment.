
"""Command line entry point for the Delta Chat assignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chat.answer import GroundedChatService
from src.config.settings import project_path, settings
from src.observability.logging import get_logger
from src.pipeline import DeltaPipeline, adapter_for

logger = get_logger(__name__)


def run_command(arguments: argparse.Namespace) -> None:
    """Run ingestion through report/index creation for one document pair."""
    result = DeltaPipeline(adapter_for(arguments.adapter)).run(Path(arguments.revision_a), Path(arguments.revision_b))
    logger.info("run_complete", extra={"report": str(project_path(settings.paths.delta_markdown)),
        "changes": result.report["summary"]["actual_changes"], "indexed_excerpts": result.indexed_documents,
        "request_id": result.request_id})


def chat_command(arguments: argparse.Namespace) -> None:
    """Answer one grounded question after `run` has built the retrieval sources."""
    answer = GroundedChatService().answer(arguments.question)
    logger.info("chat_answer", extra={"answer": answer.text, "citations": answer.citations,
        "request_id": answer.request_id})


def parser() -> argparse.ArgumentParser:
    """Configure the documented reproducible command-line interface."""
    command_parser = argparse.ArgumentParser(description="Document Delta and Grounded Chat")
    commands = command_parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="ingest, compare, report, and index a document pair")
    run.add_argument("--revision-a", default=str(project_path(settings.paths.revision_a)))
    run.add_argument("--revision-b", default=str(project_path(settings.paths.revision_b)))
    run.add_argument("--adapter", choices=("auto", "native", "scanned"), default="auto",
        help="automatic PDF detection is the default; explicit modes are retained for diagnostics")
    run.set_defaults(func=run_command)
    chat = commands.add_parser("chat", help="ask a cited question over PID A, PID B, and the delta report")
    chat.add_argument("question")
    chat.set_defaults(func=chat_command)
    return command_parser


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.func(parsed)
