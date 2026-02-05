"""
Ticket Triage RAG Bot -- Streamlit Dashboard

Interactive demo for classifying and triaging oil & gas support tickets
using Gemini embeddings, cosine similarity retrieval, and LLM generation.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import GEMINI_API_KEY, TICKETS_PATH, EMBEDDINGS_PATH, TICKET_IDS_PATH


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ticket Triage RAG Bot",
    page_icon="",
    layout="wide",
)

st.title("Ticket Triage RAG Bot")
st.caption("Embedding-based retrieval + Gemini classification for oil & gas support tickets")


# ---------------------------------------------------------------------------
# Status checks
# ---------------------------------------------------------------------------

has_api_key = bool(GEMINI_API_KEY)
has_tickets = TICKETS_PATH.exists()
has_embeddings = EMBEDDINGS_PATH.exists() and TICKET_IDS_PATH.exists()

with st.sidebar:
    st.header("System Status")

    st.metric("Gemini API", "Connected" if has_api_key else "Not configured")
    st.metric("Ticket Data", "Loaded" if has_tickets else "Missing")
    st.metric("Embeddings", "Cached" if has_embeddings else "Not generated")

    if not has_api_key:
        st.warning("Set GEMINI_API_KEY in .env to enable classification and RAG.")

    if has_tickets:
        with open(TICKETS_PATH) as f:
            all_tickets = json.load(f)
        st.metric("Total Tickets", len(all_tickets))

        categories = {}
        for t in all_tickets:
            cat = t.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        st.markdown("**Ticket Categories:**")
        for cat, count in sorted(categories.items()):
            st.text(f"  {cat}: {count}")

    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.text("Gemini 1.5 Flash")
    st.text("text-embedding-004")
    st.text("NumPy cosine similarity")
    st.text("No external vector DB")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_triage, tab_browse, tab_search = st.tabs(["Triage New Ticket", "Browse Tickets", "Similarity Search"])


# ---------------------------------------------------------------------------
# Tab 1: Triage a new ticket
# ---------------------------------------------------------------------------

with tab_triage:
    st.subheader("Submit a New Ticket for Triage")

    col1, col2 = st.columns([2, 1])

    with col1:
        ticket_title = st.text_input(
            "Ticket Title",
            placeholder="e.g., Wellhead pressure dropping on F-11",
        )
        ticket_description = st.text_area(
            "Description",
            placeholder="Describe the issue in detail...",
            height=150,
        )

    with col2:
        st.markdown("**Example issues:**")
        examples = [
            "ESP motor tripped overnight on well F-1 C",
            "Production dropped 30% on F-11 with rising water cut",
            "H2S alarm triggered at wellpad 14",
            "Missing production data for March in the database",
            "Schedule annual BOP inspection before month end",
        ]
        for ex in examples:
            st.text(f"- {ex}")

    if st.button("Triage Ticket", disabled=not has_api_key or not has_embeddings):
        if not ticket_title or not ticket_description:
            st.error("Please provide both a title and description.")
        else:
            with st.spinner("Running RAG pipeline..."):
                from src.rag import triage_ticket
                from src.vector_store import VectorStore

                store = VectorStore.load()
                result = triage_ticket(
                    title=ticket_title,
                    description=ticket_description,
                    store=store,
                    top_k=5,
                )

            # Classification
            st.markdown("---")
            st.subheader("Classification")
            cls = result["classification"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Category", cls.get("category", "N/A").replace("_", " ").title())
            c2.metric("Priority", cls.get("priority", "N/A").title())
            c3.metric("Confidence", cls.get("confidence", "N/A").title())
            st.info(f"**Reasoning:** {cls.get('reasoning', 'N/A')}")

            # Resolution suggestion
            st.subheader("Suggested Resolution")
            st.markdown(result["resolution_suggestion"])

            # Similar tickets table
            st.subheader("Similar Historical Tickets")
            similar_df = pd.DataFrame(result["similar_tickets"])
            if not similar_df.empty:
                similar_df["score"] = similar_df["score"].round(3)
                st.dataframe(
                    similar_df[["id", "title", "category", "priority", "score", "resolution"]],
                    use_container_width=True,
                    hide_index=True,
                )

    if not has_api_key:
        st.info("Configure GEMINI_API_KEY to enable triage. Without it, you can still browse tickets below.")
    elif not has_embeddings:
        st.info("Run `python -m src.embeddings` to generate embeddings before triaging.")


# ---------------------------------------------------------------------------
# Tab 2: Browse historical tickets
# ---------------------------------------------------------------------------

with tab_browse:
    st.subheader("Historical Ticket Database")

    if has_tickets:
        with open(TICKETS_PATH) as f:
            tickets = json.load(f)

        df = pd.DataFrame(tickets)

        # Filters
        col_cat, col_pri, col_well = st.columns(3)
        with col_cat:
            cat_filter = st.selectbox(
                "Category", ["All"] + sorted(df["category"].unique().tolist())
            )
        with col_pri:
            pri_filter = st.selectbox(
                "Priority", ["All"] + sorted(df["priority"].unique().tolist())
            )
        with col_well:
            well_filter = st.selectbox(
                "Wellbore", ["All"] + sorted(df["wellbore"].unique().tolist())
            )

        filtered = df.copy()
        if cat_filter != "All":
            filtered = filtered[filtered["category"] == cat_filter]
        if pri_filter != "All":
            filtered = filtered[filtered["priority"] == pri_filter]
        if well_filter != "All":
            filtered = filtered[filtered["wellbore"] == well_filter]

        st.metric("Showing", f"{len(filtered)} of {len(df)} tickets")
        st.dataframe(
            filtered[["id", "title", "category", "priority", "wellbore", "resolution"]],
            use_container_width=True,
            hide_index=True,
        )

        # Download
        csv = filtered.to_csv(index=False)
        st.download_button("Download CSV", csv, "tickets.csv", "text/csv")
    else:
        st.warning("No ticket data found. Run `python -m src.generate_tickets` first.")


# ---------------------------------------------------------------------------
# Tab 3: Similarity search
# ---------------------------------------------------------------------------

with tab_search:
    st.subheader("Semantic Similarity Search")

    if not has_embeddings:
        st.warning("Embeddings not generated. Run `python -m src.embeddings` first.")
    elif not has_api_key:
        st.warning("GEMINI_API_KEY required for embedding queries.")
    else:
        search_query = st.text_input(
            "Search query",
            placeholder="e.g., pump failure causing production loss",
        )
        top_k = st.slider("Results to return", 1, 10, 5)

        if st.button("Search"):
            if not search_query:
                st.error("Please enter a search query.")
            else:
                with st.spinner("Embedding query and searching..."):
                    from src.embeddings import embed_text
                    from src.vector_store import VectorStore

                    query_emb = np.array(embed_text(search_query))
                    store = VectorStore.load()
                    results = store.search(query_emb, top_k=top_k)

                if results:
                    for r in results:
                        with st.expander(
                            f"[{r.get('score', 0):.3f}] {r.get('id', '?')} -- {r.get('title', 'N/A')}"
                        ):
                            st.markdown(f"**Category:** {r.get('category', 'N/A')} | **Priority:** {r.get('priority', 'N/A')}")
                            st.markdown(f"**Description:** {r.get('description', 'N/A')}")
                            st.markdown(f"**Resolution:** {r.get('resolution', 'N/A')}")
                else:
                    st.info("No results found.")
