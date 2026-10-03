"""Chat page for dietary guidance.

Answers are one block per document. Not-in-corpus names what was searched.
Out of scope declines, with no document list. A generation failure is a
third state. A blank API URL answers in this process. A URL calls
``POST /chat``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.constants import CORPUS
from src.rag.pipeline import answer
from src.ui.client import ChatApiError, post_chat
from src.ui.present import ReplyShapeError, ReplyView, present

DISCLAIMER = (
    "Answers come from the seven official guidance documents. "
    "This page is a reading of those documents."
)
ALL_DOCUMENTS = "All documents"
EXAMPLE_PROMPTS: tuple[str, ...] = (
    "How long can I keep a whole chicken in the fridge?",
    "What do the documents say about cooking oil?",
    "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?",
    "What should I weigh?",
)
LOCAL_API_BASE_URL = "http://127.0.0.1:8000"
_SECRET_KEYS = ("GROQ_API_KEY", "GROQ_MODEL")


def picker_labels() -> tuple[str, ...]:
    """“All documents”, then the seven registry names in registry order."""
    return (ALL_DOCUMENTS, *(document.document_name for document in CORPUS))


def document_id_for_label(label: str) -> str | None:
    """None searches every document. A registry name sends that ``document_id``."""
    if label == ALL_DOCUMENTS:
        return None
    for document in CORPUS:
        if document.document_name == label:
            return document.document_id
    raise ValueError(f"unknown document label: {label}")


def default_api_base_url() -> str:
    """``CHAT_API_BASE_URL`` when it is set. Blank answers inside this process."""
    return os.environ.get("CHAT_API_BASE_URL", "").strip()


def apply_streamlit_secrets() -> None:
    """Copy Cloud secrets into the environment. A value already set stays put."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in _SECRET_KEYS:
        try:
            value = secrets[key]
        except Exception:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if os.environ.get(key, "").strip():
            continue
        os.environ[key] = value.strip()


def ask(message: str, document_id: str | None, base_url: str) -> dict[str, object]:
    """Answer in-process, or post to ``base_url`` when that URL is set."""
    apply_streamlit_secrets()
    url = base_url.strip()
    if url:
        return post_chat(url, message, document_id)
    return answer(message, document_id)


def main() -> None:
    st.set_page_config(page_title="Dietary guidance", layout="centered")
    apply_streamlit_secrets()
    if "history" not in st.session_state:
        st.session_state.history = []

    document_id, base_url, example = _sidebar()
    st.title("Dietary guidance")
    st.caption(DISCLAIMER)

    typed = st.chat_input("Ask about the guidance")
    outgoing = example or typed
    if outgoing:
        _submit(base_url, outgoing, document_id)
    _render_history()


def _sidebar() -> tuple[str | None, str, str | None]:
    st.sidebar.header("Scope")
    label = st.sidebar.selectbox("Document", picker_labels(), key="document")
    st.sidebar.header("Examples")
    example: str | None = None
    for prompt in EXAMPLE_PROMPTS:
        if st.sidebar.button(prompt, key=f"example-{prompt}"):
            example = prompt
    st.sidebar.header("API")
    base_url = st.sidebar.text_input(
        "API base URL",
        value=default_api_base_url(),
        key="api_base_url",
        help="Leave this blank to answer inside the app. A URL calls POST /chat on that host.",
    ).strip()
    return document_id_for_label(label), base_url or default_api_base_url(), example


def _submit(base_url: str, message: str, document_id: str | None) -> None:
    st.session_state.history.append({"role": "user", "text": message})
    try:
        response = ask(message, document_id, base_url)
        present(response)
    except ChatApiError as exc:
        st.session_state.history.append({"role": "assistant", "error": str(exc)})
        return
    except ReplyShapeError:
        st.session_state.history.append(
            {
                "role": "assistant",
                "error": "The chat API returned an unexpected response.",
            }
        )
        return
    st.session_state.history.append({"role": "assistant", "response": response})


def _render_history() -> None:
    for turn in st.session_state.history:
        role = "user" if turn["role"] == "user" else "assistant"
        with st.chat_message(role):
            if turn["role"] == "user":
                st.markdown(turn["text"])
            elif "error" in turn:
                st.markdown(f"**Chat API**\n\n{turn['error']}")
            else:
                _show(present(turn["response"]))


def _show(view: ReplyView) -> None:
    if view.kind == "answer":
        for section in view.sections:
            with st.container(border=True):
                st.markdown(f"**{section.document_name}**")
                for claim in section.claims:
                    st.markdown(claim.text)
                    st.markdown(
                        f"{claim.document_name} · {claim.publisher} · {claim.year}\n\n"
                        f"{claim.section_heading}\n\n"
                        f"[Source]({claim.source_url})"
                    )
        return
    if view.kind == "not_in_corpus":
        st.markdown("**Not covered by the guidance**")
        st.warning(view.message)
        st.markdown("**Guidance searched**")
        for document in view.searched:
            st.markdown(
                f"- {document.document_name} — {document.publisher}, {document.year}"
            )
        return
    if view.kind == "out_of_scope":
        st.markdown("**Outside this assistant**")
        st.error(view.message)
        return
    if view.kind == "generation_unavailable":
        st.markdown("**Answer unavailable**")
        st.info(view.message)
        return
    st.markdown("**Chat API**\n\nThe chat API returned an unexpected response.")


if __name__ == "__main__":
    main()
