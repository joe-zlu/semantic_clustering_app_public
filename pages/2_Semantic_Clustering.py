# pages/2_Semantic_Clustering.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import plotly.figure_factory as ff
from urllib.parse import urlparse

from modules.embeddings import load_embeddings_from_csv, save_embeddings_to_json, load_embeddings_from_json
from modules.clustering import BERTopicKeywordExtractor, DocumentTopicRelevanceAnalyzer
from modules.nav import Navbar

# Initialize session state (idempotent)
if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = None
if 'json_files' not in st.session_state:
    st.session_state.json_files = []
if 'selected_json_file' not in st.session_state:
    st.session_state.selected_json_file = None

# Reuse llama URL from session if available
llama_server_url = st.session_state.get('llama_server_url', "http://127.0.0.1:8080/")

# --- Shared Helper Functions (copied from main) ---
def wrap_text_for_hover(text, max_length=100):
    return text if len(text) <= max_length else text[:max_length] + "..."

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
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj

def calculate_tsne_perplexity(n_samples):
    default_perplexity = 30
    if n_samples <= default_perplexity:
        return max(1, n_samples // 2)
    else:
        return default_perplexity

# Derive base_name for filenames
uploaded_file_name = st.session_state.get('uploaded_file_name', "texts")
base_name = uploaded_file_name.replace('.csv', '') if uploaded_file_name else "texts"

# Sidebar (for consistency)
Navbar()

st.set_page_config(
    page_title="Semantic Clustering",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧩 Semantic Clustering")

tab1, tab2 = st.tabs(["Step 2a - Clustering by Number of Clusters", "Step 2b - Clustering by Tree Cutting"])

# === TAB 3: Clustering by Number of Clusters ===
with tab1:
    st.header("Clustering by Number of Clusters")

    if not st.session_state.embeddings_data:
        st.info("Please upload or load embeddings in **Data Management** before clustering.")
    else:
        st.write(f"Available embeddings: {len(st.session_state.embeddings_data)}")
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Set Target Number of Clusters", 1, min(100, len(st.session_state.embeddings_data)), 5)
        with col2:
            linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], index=0)

        if st.button("Perform Agglomerative Clustering by Number of Clusters"):
            if len(st.session_state.embeddings_data) < 2:
                st.error("Need at least 2 embeddings to perform clustering")
            else:
                embeddings_list = [item["embedding"] for item in st.session_state.embeddings_data]
                embeddings_array = np.array(embeddings_list)
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                cluster_labels = clusterer.fit_predict(embeddings_array)
                for i, item in enumerate(st.session_state.embeddings_data):
                    item["cluster"] = int(cluster_labels[i])
                st.session_state.cluster_labels = cluster_labels
                st.session_state.n_clusters = n_clusters
                st.success(f"Agglomerative clustering completed! Found {n_clusters} clusters")

        # Visualization
        if hasattr(st.session_state, 'cluster_labels'):
            st.subheader("Visualise Hierarchical Clusters")
            viz_method = st.selectbox("Select Visualisation Method", ["t-SNE", "PCA"], key="hierarchical_viz")
            embeddings_list = [item["embedding"] for item in st.session_state.embeddings_data]
            texts = [item["text"] for item in st.session_state.embeddings_data]
            text_ids = [item.get("text_id", i) for i, item in enumerate(st.session_state.embeddings_data)]
            cluster_labels = st.session_state.cluster_labels

            if viz_method == "PCA":
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(np.array(embeddings_list))
            else:
                perplexity = calculate_tsne_perplexity(len(embeddings_list))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                embeddings_2d = reducer.fit_transform(np.array(embeddings_list))

            wrapped_texts = [wrap_text_for_hover(t, 150) for t in texts]
            viz_df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'full_text': texts,
                'wrapped_text': wrapped_texts,
                'cluster': [f'Cluster {label}' for label in cluster_labels],
                'text_id': text_ids
            })

            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                color='cluster',
                hover_data={'wrapped_text': True, 'cluster': True, 'text_id': True},
                title=f"{viz_method} Visualisation of Agglomerative Clustering by Number of Clusters",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(
                hovertemplate='<b>Cluster:</b> %{customdata[0]}<br>' +
                            '<b>Text ID:</b> %{customdata[1]}<br>' +
                            '<b>Text:</b> %{customdata[2]}<extra></extra>',
                customdata=viz_df[['cluster', 'text_id', 'full_text']].values
            )
            fig.update_layout(width=600, height=600, xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
            st.plotly_chart(fig, width='stretch')

            # Cluster details
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            cluster_df = pd.DataFrame({'Cluster ID': unique_clusters, 'Count': counts})
            st.write(cluster_df)

            # Cluster-text mapping
            st.subheader("Cluster-Text Mapping")
            unique_clusters_sorted = sorted(unique_clusters)
            selected_cluster = st.selectbox("Filter by Cluster ID", ["All"] + [f"Cluster {c}" for c in unique_clusters_sorted], key="cluster_filter_number")
            cluster_text_df = pd.DataFrame({'Text ID': text_ids, 'Cluster ID': cluster_labels, 'Text': texts}).sort_values('Cluster ID')
            if selected_cluster != "All":
                cid = int(selected_cluster.replace("Cluster ", ""))
                cluster_text_df = cluster_text_df[cluster_text_df['Cluster ID'] == cid]
            st.dataframe(cluster_text_df, width='stretch', height=800)

            # Export
            st.subheader("Download Clustering Results")
            if 'cluster_number_default_filename' not in st.session_state:
                st.session_state.cluster_number_default_filename = f"{base_name}_clustering_by_number_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            custom_name = st.text_input("Custom filename", value=st.session_state.cluster_number_default_filename, key="cluster_num_filename")
            st.session_state.cluster_number_default_filename = custom_name
            export_data = [{
                "text_id": item.get("text_id", i),
                "text": item["text"],
                "cluster": int(item.get("cluster", -1)),
                "embedding": item["embedding"]
            } for i, item in enumerate(st.session_state.embeddings_data)]
            json_data = json.dumps(convert_to_json_serializable(export_data), indent=2)
            st.download_button("Download Agglomerative Clustering Results", data=json_data, file_name=f"{custom_name}.json", mime="application/json")

with tab2: 
    # === TAB 4: Clustering by Tree Cutting ===
    st.header("Clustering by Tree Cutting")

    if not st.session_state.embeddings_data:
        st.info("Please upload or load embeddings in **Data Management** before clustering.")
    elif len(st.session_state.embeddings_data) < 2:
        st.warning("Need at least 2 embeddings for hierarchical clustering.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            tree_height = st.slider("Tree Height (Distance Threshold)", 0.0, 10.0, 5.0, step=0.1)
        with col2:
            linkage_method_hc = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], index=0, key="linkage_height")

        if st.button("Perform Agglomerative Clustering by Tree Cutting"):
            embeddings_list = [item["embedding"] for item in st.session_state.embeddings_data]
            embeddings_array = np.array(embeddings_list)
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=tree_height, linkage=linkage_method_hc)
            cluster_labels = clusterer.fit_predict(embeddings_array)
            for i, item in enumerate(st.session_state.embeddings_data):
                item["cluster"] = int(cluster_labels[i])
            st.session_state.cluster_labels = cluster_labels
            st.session_state.n_clusters = len(set(cluster_labels))
            st.success(f"Agglomerative clustering by tree cutting completed! Found {st.session_state.n_clusters} clusters")

        # Dendrogram
        st.subheader("Dendrogram Visualisation")
        embeddings_list = [item["embedding"] for item in st.session_state.embeddings_data]
        text_ids = [item.get("text_id", i) for i, item in enumerate(st.session_state.embeddings_data)]
        texts = [item["text"] for item in st.session_state.embeddings_data]
        embeddings_array = np.array(embeddings_list)
        linkage_matrix = linkage(embeddings_array, method=linkage_method_hc)
        labels = [str(tid) for tid in text_ids]

        fig = ff.create_dendrogram(
            embeddings_array,
            orientation='bottom',
            labels=labels,
            linkagefun=lambda x: linkage(x, method=linkage_method_hc),
            color_threshold=tree_height
        )
        fig.add_hline(y=tree_height, line_dash="dash", line_color="red", annotation_text=f"Tree Cutting Height: {tree_height}")
        fig.update_layout(
            title="Interactive Agglomerative Clustering Dendrogram with Text IDs",
            xaxis_title='Text ID',
            yaxis_title='Distance',
            height=600,
            margin=dict(l=50, r=50, t=80, b=200),
            dragmode='zoom',
            hovermode=False,
            xaxis=dict(tickangle=45, tickfont=dict(size=8))
        )
        fig.add_annotation(
            text="💡 Tip: Use your mouse to zoom (click and drag) or scroll (mouse wheel). Hold Shift to pan.",
            xref="paper", yref="paper", x=0.5, y=-0.3, showarrow=False, font=dict(size=10, color="#666666"), align="center"
        )
        st.plotly_chart(fig, width='stretch')

        # Extract leaf order
        R = dendrogram(linkage_matrix, labels=labels, no_plot=True)
        leaf_order = R['leaves']
        ordered_text_ids = [text_ids[i] for i in leaf_order]
        st.session_state.dendrogram_order = ordered_text_ids

        # Visualization & export (same as Tab 3 but with tree-specific filename)
        if hasattr(st.session_state, 'cluster_labels'):
            st.subheader("Visualize Hierarchical Clusters by Tree Cutting")
            viz_method = st.selectbox("Select Visualisation Method", ["t-SNE", "PCA"], key="tree_cutting_viz")
            embeddings_list = [item["embedding"] for item in st.session_state.embeddings_data]
            texts = [item["text"] for item in st.session_state.embeddings_data]
            text_ids = [item.get("text_id", i) for i, item in enumerate(st.session_state.embeddings_data)]
            cluster_labels = st.session_state.cluster_labels

            if viz_method == "PCA":
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(np.array(embeddings_list))
            else:
                perplexity = calculate_tsne_perplexity(len(embeddings_list))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                embeddings_2d = reducer.fit_transform(np.array(embeddings_list))

            wrapped_texts = [wrap_text_for_hover(t, 150) for t in texts]
            viz_df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'full_text': texts,
                'wrapped_text': wrapped_texts,
                'cluster': [f'Cluster {label}' for label in cluster_labels],
                'text_id': text_ids
            })

            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                color='cluster',
                hover_data={'wrapped_text': True, 'cluster': True, 'text_id': True},
                title=f"{viz_method} Visualisation of Agglomerative Clustering by Tree Cutting",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(
                hovertemplate='<b>Cluster:</b> %{customdata[0]}<br>' +
                            '<b>Text ID:</b> %{customdata[1]}<br>' +
                            '<b>Text:</b> %{customdata[2]}<extra></extra>',
                customdata=viz_df[['cluster', 'text_id', 'full_text']].values
            )
            fig.update_layout(width=700, height=700, xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
            st.plotly_chart(fig, width='stretch')

            # Cluster details & mapping (same logic as above)
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            cluster_df = pd.DataFrame({'Cluster ID': unique_clusters, 'Count': counts})
            st.write(cluster_df)

            st.subheader("Cluster-Text Mapping")
            unique_clusters_sorted = sorted(unique_clusters)
            selected_cluster = st.selectbox("Filter by Cluster ID", ["All"] + [f"Cluster {c}" for c in unique_clusters_sorted], key="cluster_filter_tree")
            cluster_text_df = pd.DataFrame({'Text ID': text_ids, 'Cluster ID': cluster_labels, 'Text': texts})
            if hasattr(st.session_state, 'dendrogram_order'):
                order_map = {tid: idx for idx, tid in enumerate(st.session_state.dendrogram_order)}
                cluster_text_df['DendrogramOrder'] = cluster_text_df['Text ID'].map(order_map)
                cluster_text_df = cluster_text_df.sort_values(['Cluster ID', 'DendrogramOrder']).drop('DendrogramOrder', axis=1)
            else:
                cluster_text_df = cluster_text_df.sort_values('Cluster ID')
            if selected_cluster != "All":
                cid = int(selected_cluster.replace("Cluster ", ""))
                cluster_text_df = cluster_text_df[cluster_text_df['Cluster ID'] == cid]
            st.dataframe(cluster_text_df, width='stretch', height=800)

            # Export
            st.subheader("Download Tree Cutting Clustering Results")
            if 'cluster_tree_default_filename' not in st.session_state:
                st.session_state.cluster_tree_default_filename = f"{base_name}_clustering_by_tree_cutting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            custom_name = st.text_input("Custom filename", value=st.session_state.cluster_tree_default_filename, key="cluster_tree_filename")
            st.session_state.cluster_tree_default_filename = custom_name
            export_data = [{
                "text_id": item.get("text_id", i),
                "text": item["text"],
                "cluster": int(item.get("cluster", -1)),
                "embedding": item["embedding"]
            } for i, item in enumerate(st.session_state.embeddings_data)]
            json_data = json.dumps(convert_to_json_serializable(export_data), indent=2)
            st.download_button("Download Agglomerative Clustering Results", data=json_data, file_name=f"{custom_name}.json", mime="application/json")