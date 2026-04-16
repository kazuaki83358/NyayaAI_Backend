import streamlit as st
from agent import run_agent, initialize_vector_db
import os

# UI Page Config
st.set_page_config(page_title="Indian Law AI Agent ⚖️", layout="wide")

st.title("Indian Law AI Assistant ⚖️🇮🇳")
st.markdown("""
Welcome to your AI-powered legal assistant. This agent uses **Retrieval-Augmented Generation (RAG)** to answer queries based on official Indian legal documents.
""")

# Sidebar for Setup Instructions
with st.sidebar:
    st.header("Project Setup Guide 📚")
    st.write("1. Add your legal PDF files into the **`data/`** folder.")
    st.write("2. Ensure your **`.env`** file has a valid `GOOGLE_API_KEY`.")
    
    if st.button("🚀 Process / Re-index Documents"):
        with st.spinner("Processing PDFs and building vector database..."):
            success, message = initialize_vector_db(force_reindex=True)
            if success:
                st.success(message)
            else:
                st.error(message)

    st.divider()
    st.info("Currently, this agent can help with: \n- Constitutional Rights \n- Criminal Laws (IPC/BNS) \n- Civil Procedures \n- Family Law \n- Legal Definitions")

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# Chat Input
user_input = st.chat_input("Ask a legal question (e.g., 'What are fundamental rights?')")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(("user", user_input))

    # Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            response = run_agent(user_input)
            st.markdown(response)
            st.session_state.messages.append(("assistant", response))
