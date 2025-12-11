# Semantic Clustering & Keyword Extraction App

This Streamlit application extracts insights from unstructured text. It leverages word embeddings, semantic clustering (using reproducible Agglomerative methods), and BERTopic-style keyword extraction to help analyse and categorise large text datasets.

The application is designed to work with a local **Llama.cpp** server for privacy-preserving, offline-capable embedding generation and prompt execution.

## Features

*   **Data Management**: Upload CSVs, generate embeddings locally using Llama.cpp, and manage datasets.
*   **Semantic Clustering**: Group similar text chunks using hierarchical agglomerative clustering. Visualise results with interactive PCA/t-SNE scatter plots and dendrograms.
*   **Keyword Extraction**: Extract representative keywords for each cluster using BERTopic-style c-TF-IDF.
*   **Document Analysis**: Score document relevance to cluster keywords.
*   **LLM Prompt Generation**: Generate prompts for user to paste into LLMs (like Microsoft Copilot) for cluster summarisation and insight generation.
*   **LLM Prompt Execution**: Execute the generated prompts locally using any open-weight LLM hosted via Llama.cpp’s llama-server application.

---

## Setup Instructions

### 1. Prerequisites

Ensure you have the following installed:

*   **uv** (Python Package Manager):
    *   MacOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
*   **Llama.cpp**: You need the `llama-server` application to serve the open-weight embedding and language models. More informationcan be found here: https://github.com/ggml-org/llama.cpp
    *   [Download pre-built binaries](https://github.com/ggml-org/llama.cpp/releases) or build from source.

### 2. Installation

Clone the repository and set up the Python environment using `uv`:

```bash
# Clone the repository
git clone https://github.com/joe-zlu/semantic_clustering_app_public.git
cd semantic_clustering_app_public

# Sync dependencies (creates virtual environment)
uv sync
```

### 3. Model Deployment (Llama.cpp)

The application requires a running Llama.cpp server to generate embeddings.

1.  **Download a Model**:
    You need a GGUF model capable of embeddings (or a general LLM). Recommended models:
    *   *Nomic Embed Text* (Specifically for embeddings)
    *   *Qwen* or *Gemma* (General purpose)

2.  **Run the Server**:
    Start the server with embedding support enabled. The app defaults to looking for servers on ports `8889`, `8890`, or `8891`.

    ```bash
    # Example: Running on port 8889 with embedding enabled
    ./llama-server -m path/to/your/model.gguf -p 8889 --embedding --ctx-size 8192
    ```

    *   `-m`: Path to your GGUF model file.
    *   `-p`: Port number (match this with the URL you configure in the app).
    *   `--embedding`: **Required** to enable embedding endpoint.
    *   `--ctx-size`: Context size (ensure it covers your text chunk length).

### 4. Running the Application

Once the environment is synced and the model server is running (or ready to run), start the Streamlit app:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open your browser to the URL displayed (usually `http://localhost:8501`).

---

## How to Use

The application follows a linear 5-step workflow accessible via the sidebar.

### Step 1: Data Management
*   **Configure Server**: Enter the URL of your running generic Llama.cpp server (e.g., `http://localhost:8889`).
*   **Upload Data**: Upload a CSV file containing your text data. It must have:
    *   `text` column: The content to analyze.
    *   `text_id` column: A unique identifier for each row.
*   **Generate Embeddings**: Select the text column and click "Generate Embeddings". The app will send requests to your local Llama.cpp server.
*   **Save/Load**: You can download the generated embeddings as JSON or save them to the server for the current session.

### Step 2: Semantic Clustering
*   **Choose Method**:
    *   **By Number of Clusters**: Specify an exact number (e.g., 5 clusters). Good when you have a target taxonomy size.
    *   **By Distance Threshold**: Cut the dendrogram at a specific height. Good for natural grouping without forcing a fixed count.
*   **Visualize**: Explore the 2D scatter plot to see how your texts group together.

### Step 3 Option 1: Keyword Extraction
*   **Extract**: Use the BERTopic-inspired algorithm (c-TF-IDF) to find words that define each cluster.
*   **Refine**: Adjust "Top N Words", "N-gram range" (for phrases), and Frequency filters to clean up the keywords.
*   **Review**: See which keywords characterize each group (e.g., "service, wait time, staff" vs "quality, price, value").

### Step 3 Option 1: Document-level Keywords (Optional)
*   **Analyze**: Score individual documents against their cluster's keywords.
*   **Filter**: Identify which documents strongly align with the cluster theme and which might be outliers.

### Step 3 Option 2: LLM Prompt Generation
*   **Prepare**: Use this step to operationalize your clusters.
*   **Generate**: Write an instruction (e.g., "Summarize the key themes in these comments"). The app will bundle the clustered texts into prompt chunks acceptable for tools like Microsoft Copilot or ChatGPT.
*   **Export**: Copy the prompts or download them as text files to paste into your LLM of choice.
