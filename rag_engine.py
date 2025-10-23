import os
import streamlit as st
from typing import Optional, List, Dict

# LangChain Imports
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredODTLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Local Project Imports
from llm_config import (
    get_embedding_model, get_hana_connection, get_vector_store, get_chat_model
)
from tempfile import NamedTemporaryFile


# --- Document Processing ---

def process_and_embed_document(uploaded_file, file_type: str, chat_id: str) -> Optional[HanaDB]:
    """Loads, chunks, and embeds a document into the SAP HANA vector store."""
    docs: List[Document] = []
    temp_file_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        loader_map = {
            'pdf': PyPDFLoader,
            'txt': lambda path: TextLoader(path, encoding='utf-8'),
            'docx': Docx2txtLoader,
            'odt': UnstructuredODTLoader
        }
        if file_type in loader_map:
            loader = loader_map[file_type](temp_file_path)
            docs = loader.load()
        else:
            st.error(f"Unsupported file type for processing: {file_type}")
            return None

    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    if not docs:
        st.error("Could not extract any content from the document.")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata = {"chat_id": chat_id, "document_name": uploaded_file.name}

        embeddings_model = get_embedding_model()
        connection = get_hana_connection()
        vector_store = get_vector_store(embeddings_model, connection)

        vector_store.add_documents(chunks)
        st.toast("Document chunks and embeddings saved to HANA.", icon="💾")
        return vector_store

    except Exception as e:
        st.error(f"Error processing and embedding document: {e}")
        return None

# --- RAG Chain Creation ---

def create_history_aware_retriever_chain(retriever: BaseRetriever):
    """Creates the chain that rephrases a question based on chat history."""
    llm = get_chat_model()
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(llm, retriever, prompt)


# --- Data Retrieval and Deletion ---

def get_conversational_context(chat_history: List[Dict[str, str]], user_prompt: str, vector_store: HanaDB, chat_id: str, top_k: int = 3) -> str:
    """
    Retrieves document chunks from HANA, using the conversation history to
    rephrase the user's prompt into a more effective search query.
    """
    if not vector_store:
        return ""
    try:
        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k, "filter": {"chat_id": chat_id}}
        )
        history_aware_retriever_chain = create_history_aware_retriever_chain(base_retriever)
        langchain_chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in chat_history
        ]
        relevant_docs = history_aware_retriever_chain.invoke({
            "chat_history": langchain_chat_history,
            "input": user_prompt
        })
        return "\n---\n".join([doc.page_content for doc in relevant_docs])
    except Exception as e:
        st.error(f"Error during conversational context retrieval: {e}")
        return ""


def delete_hana_context(chat_id: str):
    """Deletes all vector embeddings for a specific chat_id from HanaDB."""
    try:
        connection = get_hana_connection()
        vector_store = get_vector_store(get_embedding_model(), connection)
        vector_store.delete(filter={"chat_id": chat_id})
        st.toast(f"Deleted context for chat from HANA.", icon="🗑️")
    except Exception as e:
        # MODIFIED: Don't crash the app, just log the error to the Streamlit UI
        st.warning(f"Warning: Could not delete old context from HANA: {e}", icon="⚠️")
