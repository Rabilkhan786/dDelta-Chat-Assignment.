run:
	uv run python main.py run --adapter native

chat:
	uv run python main.py chat "What changed?"

eval:
	uv run python -m eval.run_eval

test:
	uv run pytest -q
