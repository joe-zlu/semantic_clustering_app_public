# pages/3_Cluster_Keyword_Extraction.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

from modules.embeddings import load_embeddings_from_csv, save_embeddings_to_json, load_embeddings_from_json
from modules.clustering import BERTopicKeywordExtractor
from modules.nav import Navbar

# Initialize session state safely
if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = None
if 'bertopic_results' not in st.session_state:
    st.session_state.bertopic_results = None

# Reuse base name logic
uploaded_file_name = st.session_state.get('uploaded_file_name', "texts")
base_name = uploaded_file_name.replace('.csv', '') if uploaded_file_name else "texts"

# Sidebar (for consistency)
Navbar()

st.set_page_config(
    page_title="Cluster Keyword Extraction",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧩 Cluster Keyword Extraction")
st.header("BERTopic Keyword Extraction")
st.markdown("This page uses BERTopic's c-TF-IDF to extract keywords from your clustered texts.")

# Validation
if not st.session_state.embeddings_data:
    st.info("Please upload or load embeddings in **Data Management** before extracting keywords.")
else:
    has_clusters = any(item.get("cluster", -1) != -1 for item in st.session_state.embeddings_data)
    if not has_clusters:
        st.warning("⚠️ No clusters found. Please perform clustering in **Semantic Clustering** before extracting keywords.")
    else:
        st.success(f"Found embeddings with cluster assignments")

        # c-TF-IDF Controls
        st.subheader("c-TF-IDF Parameters")
        st.markdown("Adjust these parameters to refine the keyword extraction results:")
        col1, col2, col3 = st.columns(3)
        with col1:
            top_n_words = st.slider("Top N Words per Cluster", 5, 30, 20,
                                  help="Number of keywords to extract for each cluster.")
        with col2:
            ngram_min = st.slider("Min N-gram Size", 1, 5, 1)
            ngram_max = st.slider("Max N-gram Size", 1, 5, 2)
        with col3:
            min_df = st.slider("Min Document Frequency", 1, 5, 1)
            max_df = st.slider("Max Document Frequency (0.1–1.0)", 0.1, 1.0, 1.0, step=0.1)

        with st.expander("ℹ️ Understanding c-TF-IDF Parameters"):
            st.markdown("""
            **Class-based Term Frequency-Inverse Document Frequency (c-TF-IDF)** identifies representative keywords per cluster.
            - **Top N Words**: Controls keyword count per cluster.
            - **N-gram Range**: 1=unigrams, 2=bigrams, etc.
            - **Min DF**: Filters rare terms.
            - **Max DF**: Filters overly common terms.
            """)

        if ngram_min > ngram_max:
            st.error("Min N-gram size cannot be greater than Max N-gram size")
            ngram_range = (1, 2)
        else:
            ngram_range = (ngram_min, ngram_max)

        if st.button("Extract Keywords using BERTopic", type="primary"):
            texts = [item["text"] for item in st.session_state.embeddings_data]
            cluster_labels = np.array([item.get("cluster", -1) for item in st.session_state.embeddings_data])
            embeddings = np.array([item["embedding"] for item in st.session_state.embeddings_data])
            text_ids = [item.get("text_id", f"Index_{i}") for i, item in enumerate(st.session_state.embeddings_data)]

            extractor = BERTopicKeywordExtractor(
                top_n_words=top_n_words,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df
            )

            with st.spinner("Extracting keywords using BERTopic's c-TF-IDF..."):
                results = extractor.extract_keywords(
                    texts=texts,
                    cluster_labels=cluster_labels,
                    embeddings=embeddings,
                    text_ids=text_ids
                )

            st.session_state.bertopic_results = results
            st.session_state.current_extractor = extractor

            if results:
                st.success(f"Successfully extracted keywords for {len(results)} clusters!")
            else:
                st.error("No keywords extracted. Please check your clustering results.")

        # Display results
        if st.session_state.bertopic_results:
            results = st.session_state.bertopic_results
            total_clusters = len(results)
            total_texts = sum(r["n_texts"] for r in results.values())
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Clusters", total_clusters)
            with col2:
                st.metric("Total Texts Processed", total_texts)

            st.markdown("### Keywords by Cluster")
            cluster_summary_data = []
            for cluster_id, cluster_data in results.items():
                keywords_str = ", ".join([
                    f"{kw['word']} ({kw['score']:.3f})"
                    for kw in cluster_data["keywords"][:5]
                ])
                cluster_summary_data.append({
                    "Cluster ID": cluster_id,
                    "Number of Texts": cluster_data["n_texts"],
                    "Top 5 Keywords": keywords_str
                })
            summary_df = pd.DataFrame(cluster_summary_data)
            st.dataframe(summary_df, width='stretch')

            st.markdown("### Detailed View with Texts")
            cluster_options = ["All"] + [f"Cluster {cid}" for cid in sorted(results.keys())]
            selected_cluster = st.selectbox("Select Cluster to View", cluster_options)

            table_data = []
            for cluster_id, cluster_data in results.items():
                keywords_str = ", ".join([f"{kw['word']}" for kw in cluster_data["keywords"]])
                for text_info in cluster_data["texts"]:
                    table_data.append({
                        "Cluster ID": cluster_id,
                        "Document ID": text_info["text_id"],
                        "Text ID": text_info.get("text_id", text_info["text_id"]),
                        "Text": text_info["text"],
                        "Extracted Keywords": keywords_str
                    })
            full_df = pd.DataFrame(table_data)

            if selected_cluster != "All":
                cluster_id = int(selected_cluster.replace("Cluster ", ""))
                full_df = full_df[full_df["Cluster ID"] == cluster_id]

            st.dataframe(full_df, width='stretch', height=800)

            # Detailed scores
            with st.expander("View Detailed c-TF-IDF Scores"):
                for cluster_id, cluster_data in results.items():
                    st.markdown(f"**Cluster {cluster_id} Keywords with c-TF-IDF Scores:**")
                    kw_df = pd.DataFrame([
                        {"Keyword": kw["word"], "c-TF-IDF Score": f"{kw['score']:.4f}"}
                        for kw in cluster_data["keywords"]
                    ])
                    st.dataframe(kw_df, width='stretch')
                    st.markdown("---")

            # Download
            st.subheader("Download Results")
            if 'keywords_default_filename' not in st.session_state:
                st.session_state.keywords_default_filename = f"{base_name}_clusters_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            custom_name = st.text_input("Custom filename", value=st.session_state.keywords_default_filename, key="kw_fn")
            st.session_state.keywords_default_filename = custom_name

            # JSON
            table_data_export = []
            for _, row in full_df.iterrows():
                table_data_export.append({
                    "Cluster ID": int(row["Cluster ID"]),
                    "Document ID": int(row["Document ID"]),
                    "Text ID": row["Text ID"],
                    "Text": row["Text"],
                    "Extracted Keywords": row["Extracted Keywords"]
                })
            json_data = json.dumps(table_data_export, indent=2, ensure_ascii=False)
            st.download_button("Download Table as JSON", json_data, f"{custom_name}.json", "application/json")

            # CSV
            csv_data = full_df.to_csv(index=False)
            st.download_button("Download Table as CSV", csv_data, f"{custom_name}.csv", "text/csv")