run:
	uv run python main.py run

demo:
	uv run python main.py run --question "What changed on PSV-9066?"

chat:
	uv run python main.py chat "What changed?"

eval:
	uv run python -m eval.run_eval

serve:
	uv run uvicorn src.api:app --reload

test:
	uv run pytest -q

check:
	uv run ruff check src tests eval main.py data/samples/make_samples.py
	uv run ruff format --check src tests eval main.py data/samples/make_samples.py

format:
	uv run ruff check --fix src tests eval main.py data/samples/make_samples.py
	uv run ruff format src tests eval main.py data/samples/make_samples.py
