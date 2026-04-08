import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


# FIX 1: @st.cache_resource was only on get_embeddings — get_llm() reloaded
#         the model on every single skill call. Now both are cached.

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_llm():
    # FIX 2: flan-t5-small (80M params) cannot reliably follow structured
    #         output instructions like "Score: <number>".
    #         flan-t5-base (250M params) handles this correctly.
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=True,          # ✅ IMPORTANT
        temperature=0.7,         # works now
        top_p=0.9,
        device=-1
    )