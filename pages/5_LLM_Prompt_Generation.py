# pages/5_LLM_Prompt_Generation.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import zipfile
import io

from modules.embeddings import load_embeddings_from_csv, save_embeddings_to_json, load_embeddings_from_json
from modules.clustering import BERTopicKeywordExtractor, DocumentTopicRelevanceAnalyzer
from modules.nav import Navbar

# Initialize session state
if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = None
if 'generated_prompts' not in st.session_state:
    st.session_state.generated_prompts = None

# Reuse helpers from main app
def count_words(text):
    return len(text.split())

def split_text_by_word_limit(texts, max_words=3000):
    chunks = []
    current_chunk = []
    current_word_count = 0
    for text in texts:
        text_word_count = count_words(text)
        if text_word_count > max_words:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_word_count = 0
            words = text.split()
            for i in range(0, len(words), max_words):
                chunk_text = " ".join(words[i:i + max_words])
                chunks.append([chunk_text])
        elif current_word_count + text_word_count > max_words:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [text]
            current_word_count = text_word_count
        else:
            current_chunk.append(text)
            current_word_count += text_word_count
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_texts_by_cluster(embeddings_data):
    cluster_texts = {}
    for item in embeddings_data:
        cluster_id = item.get("cluster", -1)
        if cluster_id not in cluster_texts:
            cluster_texts[cluster_id] = []
        cluster_texts[cluster_id].append({
            "text_id": item.get("text_id", "Unknown"),
            "text": item["text"]
        })
    return cluster_texts

# Reuse base name logic
uploaded_file_name = st.session_state.get('uploaded_file_name', "texts")
base_name = uploaded_file_name.replace('.csv', '') if uploaded_file_name else "texts"

# Sidebar (for consistency)
Navbar()

st.set_page_config(
    page_title="LLM Prompt Generation",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧩 LLM Prompt Generation")
st.markdown("Construct prompts programmatically for LLM analysis of clustered texts. Generate prompts to analyse texts and assign relevant key issue tags.")

# Validation
if not st.session_state.embeddings_data:
    st.info("Please upload or load embeddings in **Data Management** before generating prompts.")
else:
    has_clusters = any(item.get("cluster", -1) != -1 for item in st.session_state.embeddings_data)
    if not has_clusters:
        st.warning("⚠️ No clusters found. Please perform clustering in **Semantic Clustering** before generating prompts.")
    else:
        st.success("Found embeddings with cluster assignments ✅")

        cluster_texts = get_texts_by_cluster(st.session_state.embeddings_data)
        total_clusters = len(cluster_texts)

        st.subheader("Prompt Configuration")
        st.markdown("Configure the three parts of your prompt:")

        col1_prompt, col2_info = st.columns([0.7, 0.3])
        with col1_prompt:
            user_prompt = st.text_area(
                "Part 1 - Analysis Instruction",
                value="""Summarise the texts below and identify key themes. Return the themes as a list of bullet points.""",
                height=200
            )
        with col2_info:
            st.info("💡 Customise the instruction based on what you want the LLM to do.")

        col1_tags, col2_info = st.columns([0.7, 0.3])
        with col1_tags:
            key_issue_tags = st.text_area(
                "Part 2 - Key Issue Tags (Optional)",
                value="",
                placeholder="Enter comma-separated tags. Leave empty if not assigning issue tags.",
                height=200
            )
        with col2_info:
            st.info("💡 Provide tags the LLM can choose from.")

        col1_limit, col2_info = st.columns([0.7, 0.3])
        with col1_limit:
            word_limit = st.number_input(
                "Maximum Words per Prompt (Part 3)",
                min_value=500,
                max_value=10000,
                value=1000,
                step=500
            )
        with col2_info:
            st.info("💡 Large texts will be split automatically.")

        st.markdown("---")

        if st.button("Generate LLM Prompts", type="primary"):
            tags_list = [tag.strip() for tag in key_issue_tags.split(",") if tag.strip()] if key_issue_tags else []
            all_prompts = {}

            with st.spinner("Generating prompts for all clusters..."):
                for cluster_id, texts_data in cluster_texts.items():
                    texts_only = [item["text"] for item in texts_data]
                    text_ids = [item["text_id"] for item in texts_data]
                    text_chunks = split_text_by_word_limit(texts_only, word_limit)
                    cluster_prompts = []
                    for i, chunk in enumerate(text_chunks):
                        texts_for_prompt = [
                            f"Text ID {text_ids[idx + sum(len(ch) for ch in text_chunks[:i])]}: {chunk[idx]}"
                            for idx in range(len(chunk))
                        ]
                        part3_text = "\n".join(texts_for_prompt)
                        full_prompt = user_prompt.strip()
                        if tags_list:
                            full_prompt += f"\n\nAvailable key issue tags: {', '.join(tags_list)}"
                        full_prompt += f"\n\n **Texts to analyse:**\n\n{part3_text}"

                        cluster_prompts.append({
                            "prompt_number": i + 1,
                            "prompt": full_prompt,
                            "word_count": count_words(full_prompt),
                            "texts_included": len(chunk),
                            "total_texts_in_cluster": len(texts_only)
                        })
                    all_prompts[cluster_id] = cluster_prompts

            st.session_state.generated_prompts = all_prompts
            st.session_state.prompts_generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Generated prompts for {len(all_prompts)} clusters!")

        # Display results
        if st.session_state.generated_prompts:
            generated_prompts = st.session_state.generated_prompts
            total_prompts = sum(len(prompts) for prompts in generated_prompts.values())
            st.info(f"Total prompts: {total_prompts} across {len(generated_prompts)} clusters")

            if hasattr(st.session_state, 'prompts_generated_time'):
                st.caption(f"Last generated: {st.session_state.prompts_generated_time}")

            cluster_options = ["All"] + [f"Cluster {cid}" for cid in sorted(generated_prompts.keys())]
            selected_cluster_display = st.selectbox("Select Cluster to View Prompts", cluster_options)

            if selected_cluster_display == "All":
                for cluster_id in sorted(generated_prompts.keys()):
                    cluster_prompts = generated_prompts[cluster_id]
                    st.markdown(f"### Cluster {cluster_id} ({len(cluster_prompts)} prompt{'s' if len(cluster_prompts) > 1 else ''})")
                    for i, prompt_data in enumerate(cluster_prompts):
                        with st.expander(f"Prompt {prompt_data['prompt_number']} - {prompt_data['word_count']} words - {prompt_data['texts_included']}/{prompt_data['total_texts_in_cluster']} texts"):
                            st.code(prompt_data['prompt'])
            else:
                cluster_id = int(selected_cluster_display.replace("Cluster ", ""))
                cluster_prompts = generated_prompts[cluster_id]
                st.markdown(f"### Cluster {cluster_id} - {len(cluster_prompts)} prompt{'s' if len(cluster_prompts) > 1 else ''}")
                for i, prompt_data in enumerate(cluster_prompts):
                    st.markdown(f"#### Prompt {prompt_data['prompt_number']} - {prompt_data['word_count']} words - {prompt_data['texts_included']}/{prompt_data['total_texts_in_cluster']} texts")
                    st.text_area(
                        f"Full Prompt - Cluster {cluster_id} - Part {prompt_data['prompt_number']}",
                        value=prompt_data['prompt'],
                        height=400,
                        key=f"prompt_cluster_{cluster_id}_{i}"
                    )
                    st.markdown("**Copy to Clipboard:**")
                    st.code(prompt_data['prompt'])
                    st.markdown("---")

            # Download
            st.subheader("Download Prompts")
            custom_prompts_filename = st.text_input(
                "Custom filename",
                value=f"llm_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                key="prompts_fn"
            )

            export_data = []
            for cluster_id, cluster_prompts in generated_prompts.items():
                for prompt_data in cluster_prompts:
                    export_data.append({
                        "Cluster ID": cluster_id,
                        "Prompt Number": prompt_data["prompt_number"],
                        "Word Count": prompt_data["word_count"],
                        "Texts Included": prompt_data["texts_included"],
                        "Total Texts in Cluster": prompt_data["total_texts_in_cluster"],
                        "Prompt": prompt_data["prompt"],
                        "User Instruction": user_prompt,
                        "Key Issue Tags": key_issue_tags or "None provided",
                        "Word Limit": word_limit
                    })

            if export_data:
                json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                st.download_button(
                    "Download Prompts as JSON",
                    json_data,
                    f"{custom_prompts_filename}.json",
                    "application/json"
                )

                # ZIP of individual .txt files
                prompt_files = {}
                for item in export_data:
                    fname = f"cluster_{item['Cluster ID']}_prompt_{item['Prompt Number']}.txt"
                    prompt_files[fname] = item["Prompt"]

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, content in prompt_files.items():
                        zf.writestr(fname, content)
                zip_buffer.seek(0)
                st.download_button(
                    "Download All Prompts as ZIP",
                    zip_buffer,
                    f"{custom_prompts_filename}.zip",
                    "application/zip"
                )