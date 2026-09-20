"""Check the single-command comparison and chat flow without model downloads."""

from types import SimpleNamespace

import main
from src.chat.answer import GroundedAnswer


class _ConsoleStream:
    def __init__(self):
        self.settings = None

    def reconfigure(self, **settings):
        self.settings = settings


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


def test_chat_command_prints_the_answer(monkeypatch, capsys):
    answer = GroundedAnswer("Grounded answer", ["[source]"], "request")
    monkeypatch.setattr(main.GroundedChatService, "answer", lambda self, question: answer)

    arguments = main.parser().parse_args(["chat", "question"])
    arguments.func(arguments)

    output = capsys.readouterr().out
    assert "Grounded answer" in output
    assert "[source]" in output


def test_console_uses_utf8_for_model_punctuation(monkeypatch):
    output = _ConsoleStream()
    errors = _ConsoleStream()
    monkeypatch.setattr(main.sys, "stdout", output)
    monkeypatch.setattr(main.sys, "stderr", errors)
    main.configure_console_output()
    assert output.settings == {"encoding": "utf-8", "errors": "replace"}
    assert errors.settings == {"encoding": "utf-8", "errors": "replace"}
