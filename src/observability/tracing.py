from contextlib import contextmanager
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

# Initialize Langfuse
langfuse = get_client()


@contextmanager
def trace(name: str):
    """
    Root trace for one request.
    Example:
        with trace("document-chat"):
            ...
    """
    with langfuse.start_as_current_observation(
        as_type="span",
        name=name
    ) as trace:

        try:
            yield trace

        except Exception as e:
            trace.update(
                status_message=str(e)
            )
            raise

        finally:
            langfuse.flush()


@contextmanager
def span(name: str):
    """
    Track one stage of the pipeline.
    Example:
        with span("ingest"):
            ...
    """
    with langfuse.start_as_current_observation(
        as_type="span",
        name=name
    ) as span:

        try:
            yield span

        except Exception as e:
            span.update(
                status_message=str(e)
            )
            raise


@contextmanager
def generation(name: str, model: str, prompt: str):
    """
    Track an LLM call.
    """
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input=prompt
    ) as generation:

        try:
            yield generation

        except Exception as e:
            generation.update(
                status_message=str(e)
            )
            raise