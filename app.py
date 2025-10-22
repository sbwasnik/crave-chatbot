import streamlit as st
import os
from typing import Optional

# Import logic from other project files
import chat_history_manager
from llm_manager import (
    process_document,
    generate_llm_response,
    delete_context,
    generate_clarifying_questions, # <-- Import is the same
)
from llm_config import check_llm_ready
from rag_engine import get_conversational_context

# --- 1. Page Configuration and Styling ---
st.set_page_config(page_title="Document Chatbot", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; text-align: left; }
    .active-chat {
        background-color: #e0f7fa !important;
        border-radius: 0.5rem;
    }
    .active-chat > div > div > button {
        background-color: #e0f7fa !important;
        color: #00796b !important;
        border: 1px solid #00796b;
    }
    footer { visibility: hidden; }
    .st-emotion-cache-1629p8f e1nzilvr5 > p {
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Callback Functions for UI Actions ---

def switch_chat_callback(chat_id: str):
    """Switches the active chat session."""
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = chat_history_manager.load_chat_history(chat_id)
    st.session_state.vector_store = None

def new_chat_callback():
    """Creates a new chat session."""
    new_id = chat_history_manager.create_new_chat()
    switch_chat_callback(new_id)

def document_upload_callback():
    """Processes an uploaded document and saves its vector store to HANA."""
    chat_id = st.session_state.current_chat_id
    uploaded_file = st.session_state.get("file_uploader_widget")

    if not uploaded_file:
        delete_context(chat_id)
        st.session_state.vector_store = None
        return

    file_ext = uploaded_file.name.split('.')[-1].lower()
    supported = ['pdf', 'txt', 'docx', 'odt']
    if file_ext not in supported:
        st.error(f"Unsupported file type: .{file_ext}")
        return

    with st.spinner(f"Processing '{uploaded_file.name}'..."):
        delete_context(chat_id)
        vector_store = process_document(uploaded_file, file_ext, chat_id)
        st.session_state.vector_store = vector_store
        if vector_store:
            st.success("Document loaded and ready. You can now ask questions about it.")
        else:
            st.error("Failed to process the document.")

# --- 3. Session State Initialization ---

if 'llm_ready' not in st.session_state:
    st.session_state.llm_ready = check_llm_ready()

st.session_state.setdefault('vector_store', None)

if 'current_chat_id' not in st.session_state:
    all_chats = chat_history_manager.get_all_chats()
    st.session_state.current_chat_id = next(iter(all_chats), None) or chat_history_manager.create_new_chat()

if 'messages' not in st.session_state:
    switch_chat_callback(st.session_state.current_chat_id)


# --- 4. Sidebar UI ---

with st.sidebar:
    st.title("Document Chatbot")
    st.button("➕ New Chat", on_click=new_chat_callback, use_container_width=True)

    st.subheader("Chat History")
    all_chats = chat_history_manager.get_all_chats()
    for chat_id, title in all_chats.items():
        is_active = chat_id == st.session_state.current_chat_id
        # Using a custom class for styling the active chat button
        container_class = "active-chat" if is_active else ""
        with st.container():
            st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
            st.button(title, key=f"chat_{chat_id}", on_click=switch_chat_callback, args=(chat_id,))
            st.markdown('</div>', unsafe_allow_html=True)

# --- 5. Main Chat Interface ---

st.header(all_chats.get(st.session_state.current_chat_id, "New Chat"))

with st.expander("Upload a Document", expanded=not st.session_state.vector_store):
    st.file_uploader(
        "Upload PDF, TXT, DOCX, or ODT files.",
        type=['pdf', 'txt', 'docx', 'odt'],
        key="file_uploader_widget",
        on_change=document_upload_callback
    )
    if st.session_state.vector_store:
        st.success("Context: A document is active for this chat.")
    else:
        st.info("No document context is active. Upload a file to begin.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.llm_ready:
        st.error("LLM is not configured. Please check your secrets.")
    else:
        # Append user message to history immediately for display
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_history_manager.save_message(st.session_state.current_chat_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        # --- UPDATED: Clarification Router Logic ---
        with st.spinner("Analyzing your question..."):
            # Pass the existing chat history to the analysis function
            chat_history_for_analysis = st.session_state.messages[:-1]
            suggestions = generate_clarifying_questions(prompt, chat_history_for_analysis)

        # ROUTE A: The question is still vague.
        if suggestions:
            with st.chat_message("assistant"):
                clarification_response = "That's a great question! To give you the best answer, could you provide a bit more detail? For example:"
                st.markdown(clarification_response)
                
                suggestion_markdown = "\n".join([f"- *{q}*" for q in suggestions])
                st.markdown(suggestion_markdown)
            
            full_response = f"{clarification_response}\n{suggestion_markdown}"
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            chat_history_manager.save_message(st.session_state.current_chat_id, "assistant", full_response)

        # ROUTE B: The question is specific enough.
        else:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                generator = None

                if st.session_state.vector_store:
                    with st.spinner("Searching document for context..."):
                        context = get_conversational_context(
                            chat_history=st.session_state.messages[:-1],
                            user_prompt=prompt,
                            vector_store=st.session_state.vector_store,
                            chat_id=st.session_state.current_chat_id
                        )
                    generator = generate_llm_response(st.session_state.messages[:-1], prompt, context)
                else:
                    generator = generate_llm_response(st.session_state.messages[:-1], prompt)

                if generator:
                    for chunk in generator:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            chat_history_manager.save_message(st.session_state.current_chat_id, "assistant", full_response)

