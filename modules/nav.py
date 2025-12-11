import streamlit as st

def Navbar():
    # Sidebar config (repeated for usability per page)
    with st.sidebar:
        st.header("Navigation")
        st.page_link('app.py', label='Home')
        st.page_link('pages/1_Data_Management.py', label='Step 1 - Generate & Load Embeddings')
        st.page_link('pages/2_Semantic_Clustering.py', label='Step 2 - Semantic Clustering')
        st.page_link('pages/3_Cluster_Keyword_Extraction.py', label='Step 3a1 - Cluster Keyword Extraction')
        st.page_link('pages/4_Document_level_Keywords.py', label='Step 3a2 - Document-level Keywords')
        st.page_link('pages/5_LLM_Prompt_Generation.py', label='Step 3b - LLM Prompt Generation')