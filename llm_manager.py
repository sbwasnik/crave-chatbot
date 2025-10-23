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
            "You MUST answer the user's question based ONLY on the provided context. "
            "If the answer is not in the context, state that clearly."
            "Cite the source of your answer from the context if possible."
        )
    else:
        system_content += "\nAnswer the user's question directly and helpfully."

    messages = [{"role": "system", "content": system_content}] + history
    messages.append({"role": "user", "content": prompt})

    yield from _call_llm_streaming(messages)


# --- 4. Question Analysis for Clarification (History & Document Aware) ---

def generate_clarifying_questions(prompt: str, history: List[Dict[str, str]], document_is_loaded: bool = False) -> Optional[List[str]]:
    """
    Analyzes the user's prompt in the context of chat history AND whether
    a document is loaded. If vague, returns clarifying questions.
    """
    # --- MODIFIED: More robust system prompt ---
    system_prompt = f"""
    You are an expert prompt analyst. Your task is to analyze the user's latest question.
    You must consider three things:
    1.  The User's Latest Question.
    2.  The full Chat History.
    3.  Whether a document is currently loaded (`document_is_loaded`: {document_is_loaded}).

    **Your Goal:** Decide if the question is too vague and needs clarification before it can be answered well.

    **--- PRIMARY RULE ---**
    Your **FIRST** task is to check if the chat history *already contains* the information to answer the latest question.
    - If the answer is present in the history, clarification is **NEVER** needed.

    **--- SCENARIO 1: A document IS loaded (`document_is_loaded`: true) ---**
    (If primary rule does not apply)
    -   **Step 1: Classify the question.** Is it about the *specific content* of the document (e.g., "Summarize this," "What is the `process_document` function?") OR is it a *general knowledge* question related to the document's topic (e.g., "How do I learn Python?", "What is Streamlit?")?
    
    -   **Step 2: Apply rules based on classification.**
        -   **If about document content:** These questions are usually valid. Broad questions like "Summarize this," or "What are the main points?" are VALID and do NOT need clarification.
        
        -   **If general knowledge:** Treat this question *exactly* like SCENARIO 2. Analyze it for vagueness, *ignoring the document*. A broad, open-ended question that needs personalization IS vague.

    **--- SCENARIO 2: NO document is loaded (`document_is_loaded`: false) ---**
    (If primary rule does not apply)
    -   Analyze the question for general vagueness.
    -   If the question is broad, open-ended, or requires personalization (like advice, recommendations, or learning plans) and the history doesn't provide context, then clarification IS needed.

    **Your Response:**
    Respond ONLY with a JSON object in the specified format:
    {{"clarification_needed": boolean, "questions": ["question 1", "question 2", ...]}}
    """
    # --- END OF MODIFICATION ---
    
    # Combine history with the new prompt for the LLM call
    messages = history + [
        {"role": "user", "content": f"""
        Here is the full chat history. 
        Note: `document_is_loaded` is {document_is_loaded}.
        
        Analyze the following latest user question and provide your JSON response based on the instructions:
        
        Latest Question: "{prompt}"
        """
        }
    ]
    
    # Prepend the main system prompt
    final_messages = [{"role": "system", "content": system_prompt}] + messages

    response_str = _call_llm_non_streaming_json(final_messages)

    if response_str:
        try:
            response_json = json.loads(response_str)
            if response_json.get("clarification_needed") and response_json.get("questions"):
                return response_json["questions"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: If JSON is malformed or keys are missing, proceed as if no clarification is needed.
            print(f"Error parsing clarification JSON: {response_str}")
            return None
    
    # Default: Proceed without clarification
    return None

