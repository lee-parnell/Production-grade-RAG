from pathlib import Path

import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"


def save_uploaded_pdf(uploaded_file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output = uploads_dir / uploaded_file.name
    output.write_bytes(uploaded_file.getbuffer())
    return output


st.set_page_config(page_title="RAG Engine", page_icon="📄", layout="centered")
st.title("RAG Engine")
st.caption("Upload documents and ask grounded questions.")

st.subheader("1) Ingest PDF")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)
if uploaded is not None:
    file_path = save_uploaded_pdf(uploaded)
    if st.button("Ingest"):
        response = requests.post(
            f"{API_BASE}/documents/ingest",
            json={"pdf_path": str(file_path.resolve()), "source_id": file_path.name},
            timeout=120,
        )
        if response.ok:
            st.success(f"Ingested {response.json().get('ingested', 0)} chunks")
        else:
            st.error(response.text)

st.subheader("2) Query")
question = st.text_input("Question")
top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
if st.button("Ask") and question.strip():
    response = requests.post(
        f"{API_BASE}/query",
        json={"question": question.strip(), "top_k": int(top_k)},
        timeout=120,
    )
    if response.ok:
        data = response.json()
        st.write(data.get("answer", ""))
        if data.get("sources"):
            st.caption("Sources")
            for source in data["sources"]:
                st.write(f"- {source}")
    else:
        st.error(response.text)
