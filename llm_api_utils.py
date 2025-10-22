import json
import requests
import time
import streamlit as st #type: ignore
from typing import List, Dict, Generator, Optional

# Import from project files
from llm_config import get_chat_url, get_llm_headers

def _call_llm_streaming(messages: List[Dict[str, str]], temperature: float = 0.1) -> Generator[str, None, None]:
    """
    Handles the low-level streaming API call to Azure OpenAI with retry logic.
    This function is separated to prevent circular imports.
    """
    headers = get_llm_headers()
    url = get_chat_url()
    payload = {"messages": messages, "temperature": temperature, "stream": True}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=90) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith(b'data: '):
                        continue
                    if line == b'data: [DONE]':
                        return
                    try:
                        chunk_str = line[6:].decode('utf-8')
                        chunk = json.loads(chunk_str)
                        if content := chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                            yield content
                    except (json.JSONDecodeError, IndexError):
                        continue
                return # End of stream
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                st.error(f"LLM API Error: {e}. Max retries reached.")
                yield "\n\nError: Could not connect to the AI model. Please check your configuration."
                return

def _call_llm_non_streaming_json(messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]:
    """
    Handles a non-streaming API call to Azure OpenAI, expecting a JSON response string.
    Note: Some models require specific prompts for JSON output, which is handled in the calling function.
    """
    headers = get_llm_headers()
    # For some Azure models, you may need to add "response_format": {"type": "json_object"} to the payload
    # if the model supports it. We will handle JSON parsing in the manager.
    payload = {
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "stream": False
    }
    url = get_chat_url()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with requests.post(url, headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                return response.json().get('choices', [{}])[0].get('message', {}).get('content')
        except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                st.error(f"LLM JSON API Error: {e}")
                return None
