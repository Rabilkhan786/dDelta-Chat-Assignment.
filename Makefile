run:
	uv run python main.py run

chat:
	uv run python main.py chat "What changed?"

eval:
	uv run python -m eval.run_eval

serve:
	uv run uvicorn src.api:app --reload

test:
	uv run pytest -q
