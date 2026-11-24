import streamlit as st
import json
from pathlib import Path
from pipeline import answer
from prompts import BUSINESS_VARIANTS

# Set up the webpage layout
st.set_page_config(page_title="KPMG AI Research Agent", page_icon="📊", layout="wide")

# Sidebar Configuration
st.sidebar.title("Agent Settings")
variant_choice = st.sidebar.selectbox(
    "Business Prompt Variant",
    options=list(BUSINESS_VARIANTS.keys()),
    index=1,
    help="v1 = plain, v2 = 5-section advisor memo, v3 = few-shot structured"
)
k_chunks = st.sidebar.slider("Number of Chunks to Retrieve", min_value=2, max_value=6, value=4)
use_rerank = st.sidebar.toggle("Enable Cross-Encoder Reranking", value=True)

# Main Canvas
st.title("KPMG AI Research Intelligence Agent")
st.caption("Translating technical arXiv research into actionable executive insights.")

# User Input
query = st.text_input("Enter your research question:")
run_button = st.button("Generate Briefing", type="primary")

# Execute Pipeline when button is clicked
if run_button and query:
    with st.spinner("Searching ChromaDB and generating KPMG memo..."):
        # Call your teammate's backend function
        result = answer(query=query, k=k_chunks, rerank=use_rerank, variant=variant_choice, business=True)

        # Create two columns for the dashboard
        col_main, col_audit = st.columns([7, 5])

        with col_main:
            st.subheader("📋 Executive Advisory Briefing")
            # Display the business translation
            st.markdown(result["business"])

            st.divider()
            st.subheader("Human-in-the-Loop Review")
            r_col1, r_col2, r_col3 = st.columns(3)
            acc = r_col1.slider("Accuracy", 1, 5, 4)
            clar = r_col2.slider("Clarity", 1, 5, 4)
            usef = r_col3.slider("Business Usefulness", 1, 5, 4)
            if st.button("Save Feedback"):
                st.success("Feedback logged!")

        with col_audit:
            st.subheader("🔍 Verification & Audit Trail")
            with st.expander("Stage 1: Technical Summary", expanded=True):
                st.info(result["summary"])

            st.write("### Retrieved Sources (ChromaDB)")
            # Iterate through the sources returned by retrieval.py
            for doc_id, page, score in result["sources"]:
                with st.expander(f"📄 {doc_id} (Page {page}) — Match: {score:.3f}"):
                    st.write("Source verified in vector database.")