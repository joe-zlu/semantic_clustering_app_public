# app.py
import streamlit as st
from modules.nav import Navbar

def main():

    Navbar()

    st.set_page_config(
        page_title="Semantic Clustering App with Llama.cpp",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧩 Semantic Clustering and Keyword Extraction Application")

    st.markdown("""
This Streamlit application helps to turn unstructured text into structured insights using LLM-generated embeddings (i.e. numerical representations of meaning), semantic clustering, keyword extraction, and LLM prompt generation (for use with Microsoft Copilot). Follow the steps below to get the most out of the tool.

---

## 📌 **Overview of Workflow**

The app is organised into **5 sequential pages** (accessible via the sidebar):

Step 1. **Data Management** → Upload & generate or load word embeddings  
Step 2. **Semantic Clustering** → Group similar texts using agglomerative clustering  
Step 3a Part 1. **Cluster Keyword Extraction** → Extract representative keywords per cluster using BERTopic language processing algorithm  
Step 3a Part 2. **Document-level Keywords** → Analyse which keywords are most relevant to each document  
Step 3b. **LLM Prompt Generation** → Auto-generate prompts for LLMs to analyse your clustered data  

> ✅ **Tip**: Always proceed in order—each step builds on the previous one.

---

## 🔹 Step 1: **Data Management**

### What it does:
- Upload a CSV file containing text chunks
- Generate embeddings using a local Llama.cpp server
- Save embeddings to server/load embeddings from server

### How to use:

1. **Configure the Llama.cpp server URL** in the sidebar. Common options:
   - `http://evlchdprs02.edw.health:8889/` → Gemma 300M (fast)
   - `http://evlchdprs02.edw.health:8890/` → Qwen3 600M (balanced)
   - `http://evlchdprs02.edw.health:8891/` → Qwen3 4B (high quality, slower)

2. **Upload a CSV file** with a "text" column (e.g., survey responses, policy comments, clinical notes) and a "text_id" column (e.g., row number, respondent ID) 

3. **Select the text column** to embed.

4. **Click “Generate Embeddings”** → The app will call your Llama.cpp server to create vector embeddings for each text.

5. **(Optional)** Compare two embeddings using cosine similarity to test the embeddings are correct.

6. **Save your embeddings**:
   - Click **“Download Embeddings as JSON”** to save locally
   - Or **“Save to Server”** for later use

---

## 🔹 Step 2: **Semantic Clustering**

### What it does:
- Group similar texts using hierarchical (agglomerative) clustering
- Visualise clusters using Principal Component Analysis (PCA) or t-distributed stochastic neighbor embedding (t-SNE)
- Explore results via dendrograms or scatter plots

### How to use:

> ⚠️ **You must have embeddings from Step 1 loaded.**

Choose **one** of two clustering methods:

### A. **By Number of Clusters**
- Set your desired number of clusters (e.g., 5–10), each cluster represents a semantic key theme
- Choose a linkage method (`ward` is usually best for compact clusters)
- Click **“Perform Clustering”**
- View interactive 2D plot and cluster-text mapping

### B. **By Tree Cutting (Distance Threshold)**
- Adjust the **Tree Height** slider to control granularity of clusters
  - Lower = more clusters, higher = fewer clusters
- View the **interactive dendrogram** → red line shows cut point
- Explore clusters in the same way as above

> 📤 **Export**: Download results as JSON to preserve cluster assignments.

---

## 🔹 Step 3 Option A - Part 1: **Cluster Keyword Extraction**

### What it does:
- Uses **BERTopic’s c-TF-IDF** to extract meaningful keywords for each cluster
- Shows which terms best represent each cluster of texts

### How to use:

> ⚠️ **You must have performed clustering in Step 2.**

1. Adjust keyword parameters:
   - **Top N Words**: How many keywords per cluster (5–20 recommended)
   - **N-gram Range**: 1–2 for phrases like “machine learning”
   - **Min/Max DF**: Filter rare or overly common words

2. Click **“Extract Keywords using BERTopic”**

3. View:
   - Summary table of top keywords per cluster
   - Full list of texts with their cluster’s keywords
   - Detailed c-TF-IDF scores (expand section)

4. **Download** results as JSON or CSV

> 💡 Use this to **label your clusters** (e.g., “Cluster 3 = Regulatory Concerns”).

---

## 🔹 Step 3 Option A Part 2: **Document-level Keywords**

### What it does:
- For each document **within a cluster**, shows **which keywords are most relevant**
- Uses **TF-IDF** or **frequency** to score relevance

### How to use:

> ⚠️ **Requires BERTopic results from Step 3 Option A Part 1.**

1. Choose a **relevance method**:
   - `tfidf` → better for distinctive terms (recommended)
   - `frequency` → simpler word counts

2. Set a **relevance threshold** (e.g., 0.1) to filter out weak matches

3. Click **“Analyse Document-Topic Relevance”**

4. Explore:
   - Cluster summary (avg. relevance per keyword)
   - Per-document keyword scores
   - Full text with relevance-highlighted terms

5. **Download** detailed results

> 💡 Helps identify **outliers** or **multi-theme documents**.

---

## 🔹 Step 3 Option B: **LLM Prompt Generation**

### What it does:
- Auto-generates ready-to-use prompts for use with Microsoft Copilot to gather additional insights (e.g. cluster summarisation, key issue tagging)
- Each prompt contains:
  - Your instruction
  - Optional issue tags
  - Clustered texts (split to respect word limits)

### How to use:

1. **Write your analysis instruction**, e.g.:
   > “Summarise key concerns in these texts and suggest policy recommendations.”

2. **(Optional)** Provide a **comma-separated list of issue tags** for categorisation.

3. Set a **word limit** (e.g., 3000 words) — large clusters will be split into multiple prompts.

4. Click **“Generate LLM Prompts”**

5. View or **copy prompts**:
   - Use the `</>` **code block** to copy to clipboard
   - Or download as **JSON** or **ZIP of .txt files**

> 💡 Paste these prompts directly into your LLM interface for batch analysis!

---

## 🛠️ **Tips & Best Practices**

- **Start small**: Test with 20–50 texts before scaling up.
- **Use consistent Llama.cpp models**: Don’t mix embeddings from different models.
- **Name your files clearly**: Include model name, date, and parameters.
- **Clusters are not final**: Adjust tree height or cluster count until groups make sense.
- **Keyword extraction improves with cluster quality**: Refine clustering if keywords seem off-topic.
- **Always validate LLM outputs**: The app prepares prompts—**you** interpret results.

---

## ❓ **Troubleshooting**

| Issue | Solution |
|------|--------|
| “No embeddings found” | Go back to **Data Management** and upload/generate data |
| Clustering fails | Ensure ≥2 embeddings; check for `NaN` in text |
| Keywords seem irrelevant | Try lowering `max_df` or increasing `min_df` |
| App is slow | Use smaller model (e.g., Gemma 300M) or reduce text length |
| JSON download fails | Ensure all integers are Python-native (app handles this automatically now) |

    """)


if __name__ == '__main__':
    main()

