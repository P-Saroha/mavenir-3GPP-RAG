"""
app.py
------
Streamlit UI for the 3GPP Release 17 RAG chatbot.
Calls the FastAPI backend — contains no RAG logic.

Usage (run the API first):
    uvicorn src.api:app --port 8000
    streamlit run app.py
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3GPP Release 17 Assistant",
    page_icon="📡",
    layout="centered",
)

# ── header ─────────────────────────────────────────────────────────────────────
st.title("📡 3GPP Release 17 5G Core Assistant")
st.caption(
    "Answers grounded in TS 23.501 · TS 23.502 · TS 23.503 — Release 17. "
    "Every factual claim is cited to the exact specification and section."
)
st.divider()

# ── question input ─────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question",
    placeholder="e.g. What is the role of the AMF in 5G core network?",
    label_visibility="collapsed",
)
ask = st.button("Ask", type="primary", disabled=not question.strip())

# ── API call + rendering ───────────────────────────────────────────────────────
if ask and question.strip():
    with st.spinner("Retrieving evidence and generating answer..."):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": question.strip()},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Make sure the backend is running:\n"
                     "`uvicorn src.api:app --port 8000`")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── evidence indicator ─────────────────────────────────────────────────────
    if data["supported"]:
        st.success("Evidence found — answer is grounded in the specifications.")
    else:
        st.warning(
            "Insufficient evidence — the question may be outside the scope of "
            "the 3GPP Release 17 5G Core specifications."
        )

    # ── answer ─────────────────────────────────────────────────────────────────
    st.markdown("### Answer")
    st.markdown(data["answer"])

    # ── sources ────────────────────────────────────────────────────────────────
    if data["sources"]:
        st.markdown("### Sources")
        for src in data["sources"]:
            spec    = src.get("spec", "—")
            section = src.get("section", "—")
            page    = src.get("page", "—")
            title   = src.get("title", "")
            sid     = src.get("id", "")
            label   = f"{sid}  **TS {spec}** · §{section} · p. {page}"
            if title:
                label += f" — *{title}*"
            st.markdown(f"- {label}")
