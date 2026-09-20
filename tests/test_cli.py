"""Check the single-command comparison and chat flow without model downloads."""

from types import SimpleNamespace

import main
from src.chat.answer import GroundedAnswer


def test_run_can_chat_with_the_pipeline_request_id(monkeypatch):
    result = SimpleNamespace(
        report={"summary": {"actual_changes": 3}}, indexed_documents=8, request_id="demo-request"
    )
    monkeypatch.setattr(main.DeltaPipeline, "run", lambda *args: result)
    calls = []

    def answer(self, question, request_id=None):
        calls.append((question, request_id))
        return GroundedAnswer("No supporting evidence", [], request_id, "unsupported")

    monkeypatch.setattr(main.GroundedChatService, "answer", answer)
    arguments = main.parser().parse_args(["run", "--question", "What changed?"])
    arguments.func(arguments)
    assert calls == [("What changed?", "demo-request")]


def test_plain_run_does_not_require_a_question():
    assert main.parser().parse_args(["run"]).question is None
