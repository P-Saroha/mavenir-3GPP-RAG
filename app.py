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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── header ─────────────────────────────────────────────────────────────────────
st.title("📡 3GPP Release 17 5G Core Assistant")
st.caption(
    "Ask questions about 3GPP specifications. Every answer is grounded with citations to TS 23.501 · TS 23.502 · TS 23.503 — Release 17."
)

# ── PDF upload section ─────────────────────────────────────────────────────────
with st.expander("📄 Upload PDF to expand knowledge base", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload a 3GPP specification PDF",
        type=["pdf"],
        help="Upload a new PDF to expand the knowledge corpus. The system will automatically parse, chunk, embed, and index it.",
    )

    if uploaded_file is not None:
        import subprocess
        from pathlib import Path

        pdf_dir = Path("data/pdfs")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / uploaded_file.name

        # Save uploaded file
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✓ Uploaded: {uploaded_file.name}")
        
        # Check for duplicates
        try:
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
        except Exception as e:
            st.error(f"Error checking duplicates: {e}")

st.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── chat display area (scrollable) ─────────────────────────────────────────────
chat_placeholder = st.container()

with chat_placeholder:
    for msg in st.session_state.chat_history:
        # User message
        st.markdown(f"**🧑 You:** {msg['question']}")

        # AI response
        st.markdown(f"**🤖 Assistant:**")
        st.markdown(msg['answer'])

        # Evidence indicator
        if msg["supported"]:
            st.success("✅ Evidence found")
        else:
            st.warning("⚠️ Low confidence")

        # Sources
        if msg["sources"]:
            st.markdown(f"**📚 Sources ({len(msg['sources'])}):**")
            for src in msg["sources"]:
                source_type = src.get("source_type", "3gpp_official")
                
                if source_type == "uploaded":
                    filename = src.get("document", "uploaded")
                    section = src.get("section", "—")
                    page_start = src.get("page", "—")
                    page_end = src.get("page_end", "")
                    page_range = f"{page_start}-{page_end}" if page_end else page_start
                    sid = src.get("id", "")
                    label = f"{sid} 📄 {filename} · §{section} · pp. {page_range}"
                else:
                    spec = src.get("spec", "—")
                    section = src.get("section", "—")
                    page_start = src.get("page", "—")
                    page_end = src.get("page_end", "")
                    page_range = f"{page_start}-{page_end}" if page_end else page_start
                    title = src.get("title", "")
                    sid = src.get("id", "")
                    label = f"{sid} TS {spec} Release {src.get('release', '17')} · §{section} · pp. {page_range}"
                    if title:
                        label += f" — {title}"
                
                with st.expander(label):
                    text_content = src.get("text", "").strip()
                    
                    if text_content:
                        st.markdown("**Evidence text:**")
                        display_text = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
                        st.markdown(f"> {display_text}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.caption(f"**Spec:** {spec if source_type != 'uploaded' else filename}")
                    with col2:
                        st.caption(f"**Section:** {section}")
                    with col3:
                        st.caption(f"**Pages:** {page_range}")
                    with col4:
                        st.caption(f"**ID:** {src.get('chunk_id', 'N/A')[:12]}")
        
        st.divider()

# ── input area (sticky at bottom) ──────────────────────────────────────────────
st.markdown("---")

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is the role of the AMF in 5G core network?",
        label_visibility="collapsed",
        key="question_input",
    )
with col2:
    send_btn = st.button("Send ➤", type="primary", use_container_width=True, disabled=not question.strip())

# ── handle submit ──────────────────────────────────────────────────────────────
if send_btn and question.strip():
    with st.spinner("🔍 Retrieving evidence..."):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": question.strip()},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Add to history
            st.session_state.chat_history.append({
                "question": question.strip(),
                "answer": data["answer"],
                "sources": data.get("sources", []),
                "supported": data["supported"]
            })
            
            st.rerun()
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach the API. Make sure the backend is running:\n`uvicorn src.api:app --port 8000`")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The backend may be slow or overloaded.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API error: {e.response.text}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ── sidebar info ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 About")
    st.markdown("""
This chatbot answers questions about **3GPP Release 17** 5G core network specifications.

**Features:**
- 📚 Answers grounded in official specs
- 🔍 Full citation with page numbers
- 📄 Expandable source text
- ✅ No hallucination — only real evidence

**Specs included:**
- TS 23.501 — System Architecture
- TS 23.502 — Procedures
- TS 23.503 — Service‑Based Architecture
""")
