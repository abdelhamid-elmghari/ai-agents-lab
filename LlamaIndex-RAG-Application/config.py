"""Configuration settings for the RAG Application with OpenRouter and Llama."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# OpenRouter Settings (Resolved from environment variables or .env)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_APIKEY = OPENROUTER_API_KEY  # Alias for backward compatibility
OPENAI_API_KEY = OPENROUTER_API_KEY      # Alias for OpenAI-compatible clients

# Proxycurl Settings (Optional for live LinkedIn extraction)
PROXYCURL_API_KEY = os.getenv("PROXYCURL_API_KEY", "")

# Model settings - OpenRouter Meta Llama models
# Popular choices:
# - "meta-llama/llama-3.1-8b-instruct" (Fast, balanced, cost-effective)
# - "meta-llama/llama-3.3-70b-instruct" (Powerful reasoning)
# - "meta-llama/llama-3.2-3b-instruct" (Lightweight & ultra-fast)
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
LLM_MODEL_ID = LLM_MODEL  # Alias for backward compatibility

# Embedding settings (Local HuggingFace embeddings - free, fast, no API key needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_MODEL_ID = EMBEDDING_MODEL

# Mock data URL (for LinkedIn profile extractor / Icebreaker tasks without paid API)
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Retrieval & Generation settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 20
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1.0

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the provided documents."
"""


class AppConfig:
    """Config wrapper object allowing `from config import config` usage."""
    def __init__(self):
        self.BASE_DIR = BASE_DIR
        self.DATA_DIR = DATA_DIR
        self.STORAGE_DIR = STORAGE_DIR
        self.OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
        self.OPENROUTER_API_KEY = OPENROUTER_API_KEY
        self.OPENROUTER_APIKEY = OPENROUTER_APIKEY
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.PROXYCURL_API_KEY = PROXYCURL_API_KEY
        self.LLM_MODEL = LLM_MODEL
        self.LLM_MODEL_ID = LLM_MODEL_ID
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.EMBEDDING_MODEL_ID = EMBEDDING_MODEL_ID
        self.MOCK_DATA_URL = MOCK_DATA_URL
        self.CHUNK_SIZE = CHUNK_SIZE
        self.CHUNK_OVERLAP = CHUNK_OVERLAP
        self.SIMILARITY_TOP_K = SIMILARITY_TOP_K
        self.TEMPERATURE = TEMPERATURE
        self.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        self.MIN_NEW_TOKENS = MIN_NEW_TOKENS
        self.TOP_K = TOP_K
        self.TOP_P = TOP_P
        self.INITIAL_FACTS_TEMPLATE = INITIAL_FACTS_TEMPLATE
        self.USER_QUESTION_TEMPLATE = USER_QUESTION_TEMPLATE


# Singleton instance so `from config import config` works everywhere
config = AppConfig()
