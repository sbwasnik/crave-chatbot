import json
import os
import uuid
from typing import List, Dict, Any, Optional

# Define the path for local history storage
HISTORY_FILE = "chat_history.json"

def _load_all_history_data() -> Dict[str, Any]:
    """Loads all chat data from the local JSON file, handling potential errors."""
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read or parse {HISTORY_FILE}. Resetting history. Error: {e}")
        return {}

def _save_all_history_data(data: Dict[str, Any]):
    """Saves all chat data to the local JSON file."""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving chat history: {e}")

def get_all_chats() -> Dict[str, str]:
    """Returns a dictionary mapping each chat_id to its title."""
    data = _load_all_history_data()
    return {
        chat_id: data[chat_id].get('title', f"Chat {chat_id[:4]}...")
        for chat_id in data
    }

def load_chat_history(chat_id: str) -> List[Dict[str, str]]:
    """Loads the message history for a specific chat ID."""
    data = _load_all_history_data()
    return data.get(chat_id, {}).get('messages', [])

def save_message(chat_id: str, role: str, content: str):
    """Appends a new message to a chat's history and updates the title if needed."""
    data = _load_all_history_data()

    if chat_id not in data:
        data[chat_id] = {
            'title': f"New Chat {chat_id[:4]}",
            'messages': []
        }

    # If this is the first user message, set it as the title
    if role == 'user' and len(data[chat_id]['messages']) == 0 and content:
        data[chat_id]['title'] = content[:50] + "..." if len(content) > 50 else content

    data[chat_id]['messages'].append({'role': role, 'content': content})
    _save_all_history_data(data)

def create_new_chat() -> str:
    """Creates a new chat entry and returns its unique ID."""
    chat_id = str(uuid.uuid4())
    data = _load_all_history_data()
    data[chat_id] = {'title': f"New Chat {chat_id[:4]}", 'messages': []}
    _save_all_history_data(data)
    return chat_id

def delete_chat(chat_id: str):
    """Deletes a specific chat from the history file."""
    data = _load_all_history_data()
    if chat_id in data:
        del data[chat_id]
        _save_all_history_data(data)

def clear_all_history():
    """Deletes the entire chat history file."""
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except OSError as e:
            print(f"Error removing history file: {e}")
