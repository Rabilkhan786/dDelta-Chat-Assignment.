"""Streamlit UI for Delta Chat.

Upload two PDF revisions (native text or scanned/OCR), send them to the FastAPI
backend (`src/api/main.py`) to extract + diff + index, review the delta report,
then ask grounded, cited questions over both revisions and the delta.

Run the backend first:  uv run uvicorn src.api.main:app --reload
Then the UI:            uv run streamlit run streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("DELTA_CHAT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Delta Chat", page_icon="\U0001f4d0", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ready" not in st.session_state:
    st.session_state.ready = False
if "summary" not in st.session_state:
    st.session_state.summary = None
if "entries" not in st.session_state:
    st.session_state.entries = []


def _error_detail(error: requests.RequestException) -> str:
    response = getattr(error, "response", None)
    if response is None:
        return str(error)
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


st.title("\U0001f4d0 Delta Chat")
st.caption(
    "Upload two revisions of an engineering document (native or scanned/OCR PDF), "
    "get a structured delta, and ask grounded, cited questions."
)

with st.sidebar:
    st.header("1. Upload revisions")
    file_a = st.file_uploader("Revision A (older)", type="pdf", key="file_a")
    file_b = st.file_uploader("Revision B (newer)", type="pdf", key="file_b")
    process = st.button("Process documents", type="primary", disabled=not (file_a and file_b))

    st.divider()
    st.caption(f"Backend: {API_URL}")
    try:
        health = requests.get(f"{API_URL}/api/health", timeout=5).json()
        st.success(f"API reachable · ready={health.get('ready')}")
    except requests.RequestException:
        st.error("Cannot reach the API. Start it with:\n\n`uv run uvicorn src.api.main:app --reload`")

if process and file_a and file_b:
    with st.spinner("Extracting (native or OCR), aligning, computing delta, and building the retrieval index..."):
        try:
            response = requests.post(
                f"{API_URL}/api/upload",
                files={
                    "revision_a": (file_a.name, file_a.getvalue(), "application/pdf"),
                    "revision_b": (file_b.name, file_b.getvalue(), "application/pdf"),
                },
                timeout=600,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            st.error(f"Processing failed: {_error_detail(error)}")
        else:
            payload = response.json()
            st.session_state.ready = True
            st.session_state.summary = payload["summary"]
            st.session_state.entries = payload["entries"]
            st.session_state.messages = []
            st.success(
                f"Indexed {payload['indexed_documents']} excerpts. "
                f"Found {payload['summary']['actual_changes']} change(s)."
            )

st.header("2. Delta report")
if st.session_state.summary:
    summary = st.session_state.summary
    columns = st.columns(5)
    columns[0].metric("Total entries", summary["total_entries"])
    columns[1].metric("Changes", summary["actual_changes"])
    columns[2].metric("Modified", summary["modified"])
    columns[3].metric("Added", summary["added"])
    columns[4].metric("Removed", summary["removed"])

    changed_entries = [entry for entry in st.session_state.entries if entry["change_type"] != "unchanged"]
    if changed_entries:
        st.dataframe(changed_entries, use_container_width=True)
    else:
        st.info("No changes detected between the two revisions.")
else:
    st.info("Upload and process two PDF revisions to see the delta report.")

st.header("3. Ask a grounded question")
if not st.session_state.ready:
    st.info("Process a document pair first (or a prior `main.py run` result was found by the backend).")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.caption("Citations: " + ", ".join(message["citations"]))

    question = st.chat_input("What changed near the compressor?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating a cited answer..."):
                try:
                    response = requests.post(f"{API_URL}/api/chat", json={"question": question}, timeout=120)
                    response.raise_for_status()
                    payload = response.json()
                    answer_text = payload["answer"]
                    citations = payload["citations"]
                except requests.RequestException as error:
                    answer_text = f"Error: {_error_detail(error)}"
                    citations = []
            st.markdown(answer_text)
            if citations:
                st.caption("Citations: " + ", ".join(citations))
        st.session_state.messages.append({"role": "assistant", "content": answer_text, "citations": citations})
