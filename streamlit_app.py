import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🔎")
st.title("🔎 Agentic RAG Assistant")
st.caption("Ask about AI-research papers — answers are grounded in a retrieved corpus.")

question = st.text_input(
    "Your question",
    placeholder="How do you evaluate large language models?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking…"):
        try:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
            resp.raise_for_status()
            st.markdown(resp.json()["answer"])
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")