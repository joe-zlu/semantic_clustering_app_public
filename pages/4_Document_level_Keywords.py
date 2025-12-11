# pages/4_Document_level_Keywords.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

from modules.embeddings import load_embeddings_from_csv, save_embeddings_to_json, load_embeddings_from_json
from modules.clustering import BERTopicKeywordExtractor, DocumentTopicRelevanceAnalyzer
from modules.nav import Navbar

def convert_to_json_serializable(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# Initialize session state
if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = None
if 'bertopic_results' not in st.session_state:
    st.session_state.bertopic_results = None
if 'relevance_results' not in st.session_state:
    st.session_state.relevance_results = None

# Reuse base name logic
uploaded_file_name = st.session_state.get('uploaded_file_name', "texts")
base_name = uploaded_file_name.replace('.csv', '') if uploaded_file_name else "texts"

# Sidebar (for consistency)
Navbar()

st.set_page_config(
    page_title="Document-level Keywords",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧩 Document-level Keywords")
st.header("Document-Topic Relevance Analysis")
st.markdown("This page analyzes which keywords/topics from each cluster are most relevant to individual documents within that cluster.")

# Validation
if not st.session_state.embeddings_data:
    st.info("Please upload or load embeddings in **Data Management** before analyzing document-topic relevance.")
else:
    has_clusters = any(item.get("cluster", -1) != -1 for item in st.session_state.embeddings_data)
    if not has_clusters:
        st.warning("⚠️ No clusters found. Please perform clustering in **Semantic Clustering** before analyzing relevance.")
    elif not st.session_state.bertopic_results:
        st.warning("⚠️ No BERTopic keyword results found. Please run **Cluster Keyword Extraction** first.")
    else:
        st.success("Found embeddings with clusters and BERTopic keywords ✅")

        # Parameters
        st.subheader("Relevance Analysis Configuration")
        col1, col2 = st.columns(2)
        with col1:
            relevance_method = st.selectbox(
                "Relevance Calculation Method",
                ["tfidf", "frequency"],
                index=0,
                help="TF-IDF: accounts for exclusivity within cluster. Frequency: simple word count."
            )
        with col2:
            min_relevance_threshold = st.slider(
                "Minimum Relevance Threshold",
                0.0, 1.0, 0.0, step=0.05,
                help="Only show keywords with relevance scores above this threshold"
            )

        with st.expander("ℹ️ Understanding Relevance Methods"):
            st.markdown("""
            - **TF-IDF**: Measures how important a keyword is to a document *within its cluster*.  
              Best for identifying *distinctive* terms.
            - **Frequency**: Counts raw occurrences. Simpler but may highlight common rather than meaningful terms.
            """)

        # Extract topic keywords from BERTopic results
        bertopic_results = st.session_state.bertopic_results
        topic_keywords = {}
        for cluster_id, cluster_data in bertopic_results.items():
            keywords = [kw["word"] for kw in cluster_data["keywords"]]
            topic_keywords[cluster_id] = keywords

        if st.button("Analyze Document-Topic Relevance", type="primary"):
            texts = [item["text"] for item in st.session_state.embeddings_data]
            cluster_labels = np.array([item.get("cluster", -1) for item in st.session_state.embeddings_data])

            analyzer = DocumentTopicRelevanceAnalyzer()
            with st.spinner("Analyzing relevance..."):
                relevance_results = analyzer.calculate_document_topic_relevance(
                    texts=texts,
                    cluster_labels=cluster_labels,
                    topic_keywords=topic_keywords,
                    method=relevance_method
                )
                summary_stats = analyzer.get_cluster_topic_summary(relevance_results)

            st.session_state.relevance_results = relevance_results
            st.session_state.relevance_summary = summary_stats
            st.session_state.current_relevance_method = relevance_method

            if relevance_results:
                st.success(f"Analyzed relevance for {len(relevance_results)} clusters!")
            else:
                st.error("No results generated. Check clustering and keyword extraction.")

        # Display results
        if st.session_state.relevance_results:
            relevance_results = st.session_state.relevance_results
            summary_stats = st.session_state.relevance_summary

            st.markdown("### Cluster Summary")
            summary_data = []
            for cluster_id, stats in summary_stats.items():
                top_keywords = ", ".join(stats["top_keywords_by_avg_relevance"][:3])
                summary_data.append({
                    "Cluster ID": cluster_id,
                    "Total Documents": stats["total_documents"],
                    "Method": stats["method"],
                    "Top 3 Keywords": top_keywords
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch')

            cluster_options = ["All"] + [f"Cluster {cid}" for cid in sorted(relevance_results.keys())]
            selected_cluster = st.selectbox("Select Cluster for Detailed View", cluster_options)

            if selected_cluster != "All":
                cluster_id = int(selected_cluster.replace("Cluster ", ""))
                cluster_data = relevance_results[cluster_id]
                cluster_summary = summary_stats[cluster_id]

                st.markdown(f"### Cluster {cluster_id} Details")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", cluster_summary["total_documents"])
                with col2:
                    st.metric("Keywords", len(cluster_summary["keyword_statistics"]))
                with col3:
                    st.metric("Method", cluster_summary["method"])

                # Keyword stats
                st.markdown("#### Keyword Relevance Statistics")
                keyword_stats = []
                for kw, stats in cluster_summary["keyword_statistics"].items():
                    keyword_stats.append({
                        "Keyword": kw,
                        "Avg Relevance": f"{stats['avg_relevance']:.4f}",
                        "Max": f"{stats['max_relevance']:.4f}",
                        "Std Dev": f"{stats['std_relevance']:.4f}",
                        "Docs with Keyword": f"{stats['documents_with_keyword']}/{stats['total_documents']}"
                    })
                kw_df = pd.DataFrame(keyword_stats).sort_values("Avg Relevance", ascending=False)
                st.dataframe(kw_df, width='stretch', height=300)

                # Document-level view
                st.markdown("#### Document-Level Relevance")
                document_data = []
                for doc in cluster_data["documents"]:
                    # Get the original text_id from embeddings_data using document_index
                    doc_index = doc["document_index"]
                    original_text_id = st.session_state.embeddings_data[doc_index].get("text_id", f"Index_{doc_index}")

                    relevant = [kw for kw in doc["topic_relevance_scores"] if kw["relevance_score"] >= min_relevance_threshold]
                    top_kws = ", ".join([f"{k['keyword']} ({k['relevance_score']:.3f})" for k in relevant[:5]])
                    document_data.append({
                        "Doc ID": doc["document_index"],
                        "Text ID": original_text_id,
                        "Text Preview": doc["text"][:100] + ("..." if len(doc["text"]) > 100 else ""),
                        "Relevant Keywords (Top 5)": top_kws if relevant else "None above threshold",
                        "Count": len(relevant)
                    })

                if document_data:
                    doc_df = pd.DataFrame(document_data).sort_values("Count", ascending=False)
                    st.dataframe(doc_df, width='stretch', height=400)

                    # Select individual doc
                    doc_ids = [doc["document_index"] for doc in cluster_data["documents"]]
                    selected_doc = st.selectbox("View Full Document Relevance", doc_ids)

                    if selected_doc is not None:
                        full_doc = next(d for d in cluster_data["documents"] if d["document_index"] == selected_doc)
                        st.write("**Text:**", full_doc["text"])
                        full_scores = [
                            {"Keyword": s["keyword"], "Score": f"{s['relevance_score']:.4f}"}
                            for s in full_doc["topic_relevance_scores"]
                            if s["relevance_score"] >= min_relevance_threshold
                        ]
                        if full_scores:
                            st.dataframe(pd.DataFrame(full_scores).sort_values("Score", ascending=False), width='stretch')
                        else:
                            st.info("No keywords meet the threshold for this document.")

            # Download
            st.subheader("Download Results")
            custom_name = st.text_input(
                "Filename",
                value=f"document_topic_relevance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                key="relevance_fn"
            )

            export_data = []
            for cid, cdata in relevance_results.items():
                for doc in cdata["documents"]:
                    # Get the original text_id from embeddings_data using document_index
                    doc_index = doc["document_index"]
                    original_text_id = st.session_state.embeddings_data[doc_index].get("text_id", f"Index_{doc_index}")

                    relevant = [kw for kw in doc["topic_relevance_scores"] if kw["relevance_score"] >= min_relevance_threshold]
                    export_data.append({
                        "Cluster ID": cid,
                        "Document ID": doc["document_index"],
                        "Text ID": original_text_id,
                        "Text": doc["text"],
                        "Relevant Keywords": ", ".join([f"{k['keyword']} ({k['relevance_score']:.3f})" for k in relevant]),
                        "Count": len(relevant),
                        "Top Keyword": relevant[0]["keyword"] if relevant else "",
                        "Top Score": relevant[0]["relevance_score"] if relevant else 0.0
                    })

            if export_data:
                json_data = json.dumps(convert_to_json_serializable(export_data), indent=2, ensure_ascii=False)
                csv_data = pd.DataFrame(export_data).to_csv(index=False)

                st.download_button("Download as JSON", json_data, f"{custom_name}.json", "application/json")
                st.download_button("Download as CSV", csv_data, f"{custom_name}.csv", "text/csv")