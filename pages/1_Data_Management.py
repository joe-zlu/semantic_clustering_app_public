# pages/1_Data_Management.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
from urllib.parse import urlparse

from modules.embeddings import load_embeddings_from_csv, save_embeddings_to_json, load_embeddings_from_json

# Initialize shared session state (safe to re-run)
if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = None
if 'json_files' not in st.session_state:
    st.session_state.json_files = []
if 'selected_json_file' not in st.session_state:
    st.session_state.selected_json_file = None

# Helper: wrap hover text
def wrap_text_for_hover(text, max_length=100):
    return text if len(text) <= max_length else text[:max_length] + "..."

# Helper: make data JSON serializable
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

# Helper: count words
def count_words(text):
    return len(text.split())

# Sidebar config (repeated for usability per page)
with st.sidebar:
    st.header("Navigation")
    st.page_link('app.py', label='Home')
    st.page_link('pages/1_Data_Management.py', label='Step 1 - Generate & Load Embeddings')
    st.page_link('pages/2_Semantic_Clustering.py', label='Step 2 - Semantic Clustering')
    st.page_link('pages/3_Cluster_Keyword_Extraction.py', label='Step 3a1 - Cluster Keyword Extraction')
    st.page_link('pages/4_Document_level_Keywords.py', label='Step 3a2 - Document-level Keywords')
    st.page_link('pages/5_LLM_Prompt_Generation.py', label='Step 3b - LLM Prompt Generation')

    st.markdown("---")

    st.header("Configuration")
    llama_server_url = st.text_input(
        "Llama.cpp Word Embedding Model URL",
        value="http://evlchdprs02.edw.health:8889/"
    )
    st.info("""
            Med quality/fast embeddings (Gemma 300M Model) - http://evlchdprs02.edw.health:8889/  
            Med quality/fast embeddings (Qwen3 600M Model) - http://evlchdprs02.edw.health:8890/  
            High quality/slow embeddings (Qwen3 4B Model) - http://evlchdprs02.edw.health:8891/  
            For benchmarks - https://huggingface.co/spaces/mteb/leaderboard
            """)
    
st.set_page_config(
    page_title="Generate & Load Embeddings",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧩 Generate & Load Embeddings")

# Store URL in session state for use across pages
st.session_state.llama_server_url = llama_server_url

st.header("Option 1 - Upload a CSV File Containing Texts & Generate Embeddings")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing text chunks", 
    type=["csv"],
    help="CSV file should contain a column with text chunks for embedding generation"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    text_column = st.selectbox(
        "Select the column containing text for embeddings",
        options=[col for col in df.columns if col != 'text_id']
    )

    text_id_column = 'text_id' if 'text_id' in df.columns else None
    if not text_id_column:
        st.warning("No 'text_id' column found. Using row indices as IDs.")

    if text_column:
        df_clean = df[[text_column]].dropna() if not text_id_column else df[[text_id_column, text_column]].dropna(subset=[text_column])
        text_chunks = df_clean[text_column].tolist()
        text_ids = df_clean[text_id_column].tolist() if text_id_column else df_clean.index.tolist()

        if st.button("Generate Embeddings"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            embeddings = []
            failed_chunks = []

            for i, text in enumerate(text_chunks):
                try:
                    status_text.text(f"Processing text {i+1}/{len(text_chunks)}: {text[:50]}...")
                    response = requests.post(
                        f"{llama_server_url}/embedding",
                        json={"content": text}
                    )
                    embedding = None
                    if response.status_code == 200:
                        response_data = response.json()

                        # Extract embedding from various possible response formats
                        if isinstance(response_data, dict):
                            if "embedding" in response_data:
                                embedding = response_data["embedding"]
                            elif "data" in response_data and isinstance(response_data["data"], list):
                                embedding = response_data["data"][0]
                            else:
                                for key, val in response_data.items():
                                    if isinstance(val, list) and len(val) > 0:
                                        if isinstance(val[0], list):
                                            embedding = val[0]
                                        elif all(isinstance(x, (int, float)) for x in val):
                                            embedding = val
                                        break
                        elif isinstance(response_data, list):
                            if isinstance(response_data[0], dict) and "embedding" in response_data[0]:
                                embedding = response_data[0]["embedding"]
                            elif all(isinstance(x, (int, float)) for x in response_data):
                                embedding = response_data
                            elif all(isinstance(x, list) for x in response_data):
                                embedding = response_data[0]

                        if embedding is not None:
                            # Handle double-nested embeddings
                            if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
                                embedding = embedding[0]
                            embeddings.append({
                                "text_id": text_ids[i],
                                "text": text,
                                "embedding": embedding
                            })
                        else:
                            failed_chunks.append((i, text, f"API Error: Could not find embedding in response"))
                    else:
                        failed_chunks.append((i, text, f"API Error: {response.status_code} - {response.text}"))
                except Exception as e:
                    failed_chunks.append((i, text, f"Error: {str(e)}"))
                progress_bar.progress((i + 1) / len(text_chunks))

            status_text.text("Processing complete!")
            st.session_state.embeddings_data = embeddings

            if failed_chunks:
                st.warning(f"Failed to process {len(failed_chunks)} out of {len(text_chunks)} chunks.")
                for idx, text, error in failed_chunks[:5]:
                    st.error(f"Chunk {idx}: {text[:50]}... - {error}")
            st.success(f"Successfully generated embeddings for {len(embeddings)} text chunks!")
            if embeddings:
                st.write(f"Embedding dimension: {len(embeddings[0]['embedding'])}")

# Compare two embeddings
if st.session_state.embeddings_data and len(st.session_state.embeddings_data) >= 2:
    st.markdown("---")
    st.header("Compare Two Embeddings (Test)")
    col1, col2 = st.columns(2)
    with col1:
        emb1_idx = st.selectbox("Select first embedding", 
                               range(len(st.session_state.embeddings_data)), 
                               format_func=lambda x: f"{st.session_state.embeddings_data[x]['text'][:30]}...")
    with col2:
        emb2_idx = st.selectbox("Select second embedding", 
                               range(len(st.session_state.embeddings_data)), 
                               format_func=lambda x: f"{st.session_state.embeddings_data[x]['text'][:30]}...")

    if st.button("Calculate Cosine Similarity"):
        emb1 = np.array(st.session_state.embeddings_data[emb1_idx]["embedding"])
        emb2 = np.array(st.session_state.embeddings_data[emb2_idx]["embedding"])
        if emb1.shape != emb2.shape:
            st.error(f"Embeddings have different dimensions: {emb1.shape} vs {emb2.shape}")
        elif np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            st.error("One embedding is a zero vector")
        else:
            cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            st.metric("Cosine Similarity", f"{cosine_sim:.4f}")
            st.subheader("Text 1:")
            st.write(st.session_state.embeddings_data[emb1_idx]["text"])
            st.subheader("Text 2:")
            st.write(st.session_state.embeddings_data[emb2_idx]["text"])

# Download & Save Embeddings (from original Tab 1)
if st.session_state.embeddings_data:
    st.markdown("---")
    st.subheader("Store Embeddings")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = uploaded_file.name.replace('.csv', '') if uploaded_file else "texts"
    if 'embeddings_default_filename' not in st.session_state:
        st.session_state.embeddings_default_filename = f"{base_name}_embeddings_{timestamp}"
    
    custom_filename = st.text_input(
        "Custom filename (without extension)",
        value=st.session_state.embeddings_default_filename,
        help="Enter your desired filename. Extension will be added automatically."
    )
    st.session_state.embeddings_default_filename = custom_filename
    download_filename = f"{custom_filename}.json"

    export_embeddings = []
    for item in st.session_state.embeddings_data:
        item_copy = item.copy()
        item_copy.setdefault("cluster", -1)
        export_embeddings.append(item_copy)

    json_data = json.dumps(convert_to_json_serializable(export_embeddings), indent=2)

    col1, col2, _ = st.columns([0.3, 0.3, 0.4])
    with col1:
        st.download_button(
            "Download Embeddings as JSON",
            data=json_data,
            file_name=download_filename,
            mime="application/json"
        )
    with col2:
        if st.button("Save Embeddings to Server"):
            save_dir = "./temp_embeddings/"
            os.makedirs(save_dir, exist_ok=True)
            port = urlparse(llama_server_url).port
            model_name = {
                8889: "Gemma300M",
                8890: "Qwen3_600M",
                8891: "Qwen3_4B"
            }.get(port, "unknown_model")
            filename = f"{custom_filename}_{model_name}.json"
            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(export_embeddings, f, indent=2)
            st.success(f"Saved to {file_path}")
            st.session_state.json_files = [f for f in os.listdir(save_dir) if f.endswith('.json')]

# === Tab 2 functionality: Load Embeddings ===
st.header("Option 2 - Upload a JSON File Containing Embeddings")

uploaded_json = st.file_uploader(
    "Upload a JSON file containing embeddings",
    type=["json"]
)
if uploaded_json:
    try:
        loaded = json.load(uploaded_json)
        if isinstance(loaded, list) and loaded:
            valid = all(isinstance(i, dict) and "text" in i and "embedding" in i for i in loaded)
            if valid:
                for item in loaded:
                    item.setdefault("cluster", -1)
                st.session_state.embeddings_data = loaded
                st.session_state.selected_json_file = uploaded_json.name
                st.success(f"Loaded {len(loaded)} embeddings from {uploaded_json.name}")
                with st.expander("Preview"):
                    sample = loaded[0]
                    st.json({
                        "text_id": sample.get("text_id", "N/A"),
                        "text": sample["text"][:100] + ("..." if len(sample["text"]) > 100 else ""),
                        "embedding_dim": len(sample["embedding"]),
                        "cluster": sample.get("cluster", -1)
                    })
            else:
                st.error("Invalid JSON: expected list of {text, embedding}")
        else:
            st.error("JSON must be a non-empty list")
    except Exception as e:
        st.error(f"Error: {e}")

# Manage server files
st.markdown("---")

st.header("Option 3 - Load a Server File Containing Embeddings")

save_dir = "./temp_embeddings/"
os.makedirs(save_dir, exist_ok=True)
json_files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
st.session_state.json_files = json_files

if json_files:
    st.write(f"Found {len(json_files)} JSON file(s) in /temp_embeddings/:")
    selected_file = st.selectbox("Select a server file to load", json_files)
    if st.button("Load from Server"):
        try:
            with open(os.path.join(save_dir, selected_file), 'r') as f:
                loaded = json.load(f)
            fixed = []
            for i, item in enumerate(loaded):
                if not isinstance(item, dict):
                    continue
                if "text" not in item or "embedding" not in item:
                    continue
                if not isinstance(item["embedding"], list) or not item["embedding"]:
                    continue
                item.setdefault("cluster", -1)
                item.setdefault("text_id", i)
                fixed.append(item)
            if fixed:
                st.session_state.embeddings_data = fixed
                st.session_state.selected_json_file = selected_file
                st.success(f"Loaded {len(fixed)} embeddings from {selected_file}")
            else:
                st.error("No valid embeddings found")
        except Exception as e:
            st.error(f"Load error: {e}")

    st.subheader("Delete Server Files")
    files_to_delete = st.multiselect("Select files to delete", json_files)
    if files_to_delete and st.button("Delete Selected Files"):
        for f in files_to_delete:
            try:
                os.remove(os.path.join(save_dir, f))
            except Exception as e:
                st.warning(f"Failed to delete {f}: {e}")
        st.success("Deletion attempted")
        st.rerun()
else:
    st.info("No JSON files in /temp_embeddings/")