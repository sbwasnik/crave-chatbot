import os
import shutil
import streamlit as st
import json
from typing import List, Dict, Optional, Generator

# Import components from other modules
from rag_engine import process_and_embed_document, delete_hana_context
from llm_api_utils import _call_llm_streaming, _call_llm_non_streaming_json

# Import types for hinting
from langchain_community.vectorstores.hanavector import HanaDB

# --- 1. Persistence Management ---

def delete_context(chat_id: str):
    """Deletes the persisted context from HANA for a specific chat."""
    delete_hana_context(chat_id)

# --- 2. Document Processing ---

def process_document(uploaded_file, file_type: str, chat_id: str) -> Optional[HanaDB]:
    """Routes document processing to the RAG engine for embedding in HANA."""
    supported_types = ['pdf', 'txt', 'docx', 'odt']
    if file_type in supported_types:
        return process_and_embed_document(uploaded_file, file_type, chat_id)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None

# --- 3. Core LLM Functionality ---

def generate_llm_response(history: List[Dict[str, str]], prompt: str, context: Optional[str] = None) -> Generator[str, None, None]:
    """Generates a standard streaming LLM response, optionally with RAG context."""
    system_content = "You are a helpful and friendly AI assistant."
    if context:
        system_content += (
            f"\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
            "Answer the user's question based ONLY on the provided context. "
            "If the answer isn't in the context, say so clearly."
        )

    messages = [{"role": "system", "content": system_content}] + history
    messages.append({"role": "user", "content": prompt})

    yield from _call_llm_streaming(messages)


# --- 4. Question Analysis for Clarification (History-Aware) ---

def generate_clarifying_questions(prompt: str, history: List[Dict[str, str]]) -> Optional[List[str]]:
    """
    Analyzes the user's prompt in the context of the chat history. If the prompt
    is still too vague, returns clarifying questions. Otherwise, returns None.
    """
    system_prompt = """
    You are an expert prompt analyst. Your task is to analyze the user's latest question in the context of the entire chat history to determine if it can be answered effectively.

    1.  **Review the chat history** to understand the conversation and see what information the user has already provided.
    2.  **Analyze the latest user question.**
    3.  **Decide:** Given the history, is the latest question specific enough to be answered well?
        -   If the question is now specific (because the user provided details in previous messages), or was specific to begin with, you MUST determine that no clarification is needed.
        -   Only if the latest question remains vague and the necessary details are MISSING from the chat history, should you generate 3-4 concise clarifying questions.

    Respond ONLY with a JSON object in the following format:
    {"clarification_needed": boolean, "questions": ["question 1", "question 2", ...]}

    **Example 1: Clarification is NOT needed.**
    -   History: [{"role": "user", "content": "Which phone should I buy?"}, {"role": "assistant", "content": "What's your budget?"}, {"role": "user", "content": "Under $500."}]
    -   Latest Question: "What are my best options?"
    -   Your JSON Output: {"clarification_needed": false, "questions": []}

    **Example 2: Clarification IS needed.**
    -   History: []
    -   Latest Question: "What's the best car?"
    -   Your JSON Output: {"clarification_needed": true, "questions": ["What is your budget?", "What is the primary use (e.g., family, commute, sports)?", "Do you have a preference for electric or gasoline?"]}
    """
    
    # Combine history with the new prompt for the LLM call
    messages = history + [{"role": "user", "content": f"Here is the full chat history. Now, analyze the following latest user question and provide your JSON response based on the instructions:\n\nLatest Question: \"{prompt}\""}]
    
    # Prepend the main system prompt
    final_messages = [{"role": "system", "content": system_prompt}] + messages

    response_str = _call_llm_non_streaming_json(final_messages)

    if response_str:
        try:
            response_json = json.loads(response_str)
            if response_json.get("clarification_needed") and response_json.get("questions"):
                return response_json["questions"]
        except (json.JSONDecodeError, KeyError):
            return None
    return None

