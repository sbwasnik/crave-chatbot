import os
import streamlit as st
from typing import Dict, Any

# LangChain & Database Imports
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.hanavector import HanaDB
from hdbcli import dbapi

# --- 1. Centralized Secret and Configuration Loading ---

# Define all required secret keys
REQUIRED_SECRET_KEYS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
    "HANA_ADDRESS",
    "HANA_PORT",
    "HANA_USER",
    "HANA_PASSWORD",
]

def load_configuration() -> Dict[str, Any]:
    """
    Loads secrets from Streamlit's secrets management, provides clear errors,
    and stops the app if configuration is incomplete.
    """
    missing_keys = [key for key in REQUIRED_SECRET_KEYS if not st.secrets.get(key)]

    if missing_keys:
        st.error(
            "Configuration Error: The following required secrets are missing:\n\n"
            f"**{', '.join(missing_keys)}**\n\n"
            "Please create a file named `.streamlit/secrets.toml` and ensure all "
            "required keys are present at the top level (not under section headers like [azure]). "
            "Check the `secrets.toml.example` file for the correct structure."
        )
        st.stop()  # Halt the app execution

    # If all keys are present, return a dictionary of them.
    return {key: st.secrets[key] for key in REQUIRED_SECRET_KEYS}

# Load all configurations at the start. App will stop if anything is missing.
config = load_configuration()

# --- 2. Assign Loaded Config to Variables ---

# LLM and Embedding Configuration
AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = config["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = config["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# SAP HANA Cloud Configuration
HANA_ADDRESS = config["HANA_ADDRESS"]
HANA_PORT = config["HANA_PORT"]
HANA_USER = config["HANA_USER"]
HANA_PASSWORD = config["HANA_PASSWORD"]
HANA_TABLE_NAME = "CHATBOT_DOCUMENT_EMBEDDINGS"

# --- 3. Utility Functions using the Configuration ---

def get_llm_headers() -> Dict[str, str]:
    """Returns standard headers for Azure OpenAI API calls."""
    return {"Content-Type": "application/json", "api-key": str(AZURE_OPENAI_API_KEY)}

def get_chat_url() -> str:
    """Constructs the full URL for the Azure OpenAI Chat Completions API."""
    return (
        f"{str(AZURE_OPENAI_ENDPOINT).rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )

def get_embedding_model() -> AzureOpenAIEmbeddings:
    """Initializes and returns the AzureOpenAIEmbeddings model for use with LangChain."""
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        chunk_size=16,
    )

def get_chat_model() -> AzureChatOpenAI:
    """Initializes and returns the AzureChatOpenAI model for use with LangChain."""
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.1,
    )

def get_hana_connection() -> dbapi.Connection:
    """Establishes and returns a connection to the SAP HANA database."""
    try:
        return dbapi.connect(
            address=str(HANA_ADDRESS),
            port=int(HANA_PORT),
            user=str(HANA_USER),
            password=str(HANA_PASSWORD)
            )
    except Exception as e:
        st.error(f"HANA Connection Error: {e}")
        raise

def get_vector_store(embedding_model: AzureOpenAIEmbeddings, connection: dbapi.Connection) -> HanaDB:
    """Initializes and returns the HanaDB vector store instance."""
    return HanaDB(
        embedding=embedding_model, connection=connection, table_name=HANA_TABLE_NAME
    )

def check_llm_ready() -> bool:
    """
    Checks if the system can connect to HANA. The secret loading check
    is now handled globally at the start of the script.
    """
    conn = None
    try:
        # Establish connection without a 'with' statement
        conn = get_hana_connection()
        is_ready = conn.isconnected()
        return is_ready
    except Exception as e:
        # The error from get_hana_connection will already be displayed
        st.error(f"System Initialization Error: Could not connect to HANA. Details: {e}")
        return False
    finally:
        # Ensure the connection is closed if it was successfully created
        if conn and conn.isconnected():
            conn.close()
