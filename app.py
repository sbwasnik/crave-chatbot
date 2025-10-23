import streamlit as st
import os
from typing import Optional

# Import logic from other project files
# Note: chat_history_manager is removed
from llm_manager import (
    process_document,
    generate_llm_response,
    delete_context,
    generate_clarifying_questions,
)
from llm_config import check_llm_ready
from rag_engine import get_conversational_context

# --- 1. Page Configuration and Styling ---
st.set_page_config(page_title="Document Chatbot", layout="wide")

st.markdown("""
<style>
    /* Remove sidebar padding */
    .css-1y4p8pa {
        padding-top: 2rem;
    }
    footer { visibility: hidden; }
    .st-emotion-cache-1629p8f e1nzilvr5 > p {
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Static Chat ID ---
# Since we removed chat history, we use a single, static ID for this session's
# RAG context in HANA.
CHAT_ID = "single_chat_session"

# --- 3. Callback Functions for UI Actions ---

def document_upload_callback():
    """Processes an uploaded document and saves its vector store to HANA."""
    uploaded_file = st.session_state.get("file_uploader_widget")

    # Clear any existing context first
    delete_context(CHAT_ID)
    st.session_state.vector_store = None
    
    if not uploaded_file:
        st.toast("Document context cleared.")
        return

    file_ext = uploaded_file.name.split('.')[-1].lower()
    supported = ['pdf', 'txt', 'docx', 'odt']
    if file_ext not in supported:
        st.error(f"Unsupported file type: .{file_ext}")
        return

    with st.spinner(f"Processing '{uploaded_file.name}'..."):
        vector_store = process_document(uploaded_file, file_ext, CHAT_ID)
        st.session_state.vector_store = vector_store
        if vector_store:
            st.success("Document loaded. You can now ask questions about it.")
            # Add a message to the chat
            doc_message = f"Document loaded: `{uploaded_file.name}`. I'm ready for your questions about it."
            st.session_state.messages.append({"role": "assistant", "content": doc_message})
        else:
            st.error("Failed to process the document.")

# --- 4. Session State Initialization ---

if 'llm_ready' not in st.session_state:
    st.session_state.llm_ready = check_llm_ready()

# vector_store holds the RAG context
st.session_state.setdefault('vector_store', None)

# 'messages' now only holds the current session's chat
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Clean up any potential old context from a previous run
    delete_context(CHAT_ID)


# --- 5. Sidebar UI (New) ---

with st.sidebar:
    st.title("Document Chatbot")
    st.markdown("---")
    
    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload PDF, TXT, DOCX, or ODT files.",
        type=['pdf', 'txt', 'docx', 'odt'],
        key="file_uploader_widget",
        on_change=document_upload_callback,
        help="Upload a document to chat with it. Re-uploading or clearing the file will clear the document context."
    )
    if st.session_state.vector_store:
        st.success("Context: A document is active.")
    else:
        st.info("No document context is active.")
    
    st.markdown("---")


# --- 6. Main Chat Interface ---

st.header("Chatbot")
st.markdown("Ask a question, or upload a document in the sidebar to ask questions about it.")

# Display chat messages from the current session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.llm_ready:
        st.error("LLM is not configured. Please check your secrets.")
    else:
        # Append user message to history immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- UPDATED: Clarification Router Logic ---
        with st.spinner("Analyzing your question..."):
            # Pass the *current* chat history
            chat_history_for_analysis = st.session_state.messages[:-1]
            document_is_loaded = st.session_state.vector_store is not None
            
            suggestions = generate_clarifying_questions(
                prompt=prompt, 
                history=chat_history_for_analysis,
                document_is_loaded=document_is_loaded
            )

        # ROUTE A: The question is vague and needs clarification.
        if suggestions:
            with st.chat_message("assistant"):
                clarification_response = "That's a great question! To give you the best answer, could you provide a bit more detail? For example:"
                st.markdown(clarification_response)
                
                suggestion_markdown = "\n".join([f"- *{q}*" for q in suggestions])
                st.markdown(suggestion_markdown)
            
            full_response = f"{clarification_response}\n{suggestion_markdown}"
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # ROUTE B: The question is specific enough.
        else:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                generator = None
                context = None

                # Check if we should use RAG (document context)
                if st.session_state.vector_store:
                    with st.spinner("Searching document for context..."):
                        context = get_conversational_context(
                            chat_history=st.session_state.messages[:-1],
                            user_prompt=prompt,
                            vector_store=st.session_state.vector_store,
                            chat_id=CHAT_ID
                        )
                
                # Generate the response, with or without context
                generator = generate_llm_response(
                    history=st.session_state.messages[:-1], 
                    prompt=prompt, 
                    context=context
                )

                if generator:
                    for chunk in generator:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
