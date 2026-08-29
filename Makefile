run:
	uv run python main.py run

chat:
	uv run python main.py chat "What changed?"

eval:
	uv run python -m eval.run_eval

test:
	uv run pytest -q

api:
	uv run uvicorn src.api.main:app --reload --port 8000

ui:
	uv run streamlit run streamlit_app.py
