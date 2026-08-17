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

# ── PDF upload section ─────────────────────────────────────────────────────────
st.markdown("### 📄 Add New PDF to Corpus")
uploaded_file = st.file_uploader(
    "Upload a 3GPP specification PDF",
    type=["pdf"],
    help="Upload a new PDF to expand the knowledge corpus. "
         "The system will automatically parse, chunk, embed, and index it.",
)

if uploaded_file is not None:
    import subprocess
    from pathlib import Path
    import tempfile

    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / uploaded_file.name

    # Save uploaded file
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✓ Uploaded: {uploaded_file.name}")
    
    # Check for duplicates
    from src.ingestion.upload_handler import UploadHandler
    handler = UploadHandler()
    file_hash = handler.compute_hash(pdf_path)
    is_duplicate, prev_status = handler.check_duplicate(uploaded_file.name)
    
    if is_duplicate:
        st.warning(f"⚠ This file was already processed (status: {prev_status})")
    else:
        # Auto-run ingestion pipeline
        st.info(
            "🔄 Processing PDF...\n\n"
            "Steps: Parsing → Chunking → Embedding → Indexing\n\n"
            "⏱ Note: Embedding is slow on CPU (~2h for 2k chunks). "
            "For faster embedding on GPU, use Google Colab or a machine with CUDA."
        )
        
        try:
            # Run ingestion (timeout increased to 2 hours for embedding)
            result = subprocess.run(
                ["python", "-m", "src.ingestion.ingest", str(pdf_path)],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hours max
            )
            
            if result.returncode == 0:
                st.success(
                    "✓ Done! PDF has been parsed, chunked, embedded, and indexed.\n\n"
                    "You can now ask questions about this document."
                )
                st.balloons()
            else:
                st.error(f"Ingestion failed:\n{result.stderr}")
                if result.stdout:
                    st.write("Output:\n" + result.stdout[-1000:])  # Last 1000 chars
        
        except subprocess.TimeoutExpired:
            st.error("Ingestion took too long (>2 hours). PDF may be very large. Check backend logs.")
        except Exception as e:
            st.error(f"Error during ingestion: {e}")

st.divider()

# ── session state for chat history ────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# ── question input ─────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question",
    placeholder="e.g. What is the role of the AMF in 5G core network?",
    label_visibility="collapsed",
    key="question_input",
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
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question.strip(),
                "answer": data["answer"],
                "sources": data.get("sources", []),
                "supported": data["supported"]
            })
            
            # Clear the input after successful response
            st.session_state.question_input = ""
            
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Make sure the backend is running:\n"
                     "`uvicorn src.api:app --port 8000`")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")

# ── display chat history ──────────────────────────────────────────────────────
st.divider()
if st.session_state.chat_history:
    st.markdown("### Chat History")
    
    for idx, msg in enumerate(st.session_state.chat_history):
        # Question (in a blue container, right-aligned style)
        st.markdown(f"**You:** {msg['question']}")
        
        # Evidence indicator
        if msg["supported"]:
            st.success("✓ Evidence found — answer is grounded in the specifications.")
        else:
            st.warning("⚠ Insufficient evidence — may be outside the scope of 3GPP R17.")
        
        # Answer
        st.markdown(f"**Assistant:**\n\n{msg['answer']}")
        
        # Sources
        if msg["sources"]:
            st.markdown("**Sources:**")
            for src in msg["sources"]:
                # Determine source type and format accordingly
                source_type = src.get("source_type", "3gpp_official")
                
                if source_type == "uploaded":
                    # Uploaded PDF: show filename instead of spec
                    filename = src.get("document", "uploaded")
                    section = src.get("section", "—")
                    page = src.get("page", "—")
                    sid = src.get("id", "")
                    label = f"{sid}  **{filename}** · §{section} · p. {page}"
                else:
                    # Official 3GPP: show full spec
                    spec = src.get("spec", "—")
                    section = src.get("section", "—")
                    page = src.get("page", "—")
                    title = src.get("title", "")
                    sid = src.get("id", "")
                    label = f"{sid}  **TS {spec}** · Release {src.get('release', '17')} · §{section} · p. {page}"
                    if title:
                        label += f" — *{title}*"
                
                st.markdown(f"- {label}")
        
        st.divider()
