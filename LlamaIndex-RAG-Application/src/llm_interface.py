"""Module for interfacing with OpenRouter LLMs (Llama models) and Embeddings."""

import logging
from typing import Dict, Any, Optional

from llama_index.core import Settings
import config

logger = logging.getLogger(__name__)


def create_openrouter_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    **kwargs
) -> Any:
    """Creates a Llama LLM via OpenRouter for generating responses.
    
    Args:
        model: OpenRouter model identifier (defaults to config.LLM_MODEL).
        api_key: OpenRouter API key.
        temperature: Sampling temperature (0.0 for deterministic).
        max_new_tokens: Maximum tokens to generate.
        
    Returns:
        LlamaIndex LLM instance.
    """
    selected_model = model or config.LLM_MODEL
    key = api_key or config.OPENROUTER_API_KEY
    temp = config.TEMPERATURE if temperature is None else temperature
    tokens = max_new_tokens or config.MAX_NEW_TOKENS

    try:
        # Preferred: llama-index-llms-openrouter
        from llama_index.llms.openrouter import OpenRouter
        llm = OpenRouter(
            api_key=key,
            model=selected_model,
            temperature=temp,
            max_tokens=tokens,
            **kwargs
        )
        logger.info(f"Initialized OpenRouter LLM: {selected_model}")
    except (ImportError, Exception) as err:
        # Fallback: OpenRouter provides a 100% OpenAI-compatible endpoint
        logger.info(f"Using OpenAI compatible client for OpenRouter: {err}")
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(
            api_key=key,
            api_base=config.OPENROUTER_BASE_URL,
            model=selected_model,
            temperature=temp,
            max_tokens=tokens,
            **kwargs
        )

    # Set as global default LLM in LlamaIndex Settings
    Settings.llm = llm
    return llm


def create_embedding_model(model_name: Optional[str] = None) -> Any:
    """Creates an Embedding model for vector representations.
    
    Defaults to local HuggingFace embedding (BAAI/bge-small-en-v1.5) which
    runs locally without requiring extra paid credits.
    
    Args:
        model_name: HuggingFace or OpenAI embedding model name.
        
    Returns:
        LlamaIndex BaseEmbedding instance.
    """
    embed_name = model_name or config.EMBEDDING_MODEL

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=embed_name)
        logger.info(f"Initialized HuggingFaceEmbedding: {embed_name}")
    except (ImportError, Exception) as e:
        logger.warning(f"HuggingFace embedding initialization failed ({e}), attempting OpenAI embedding fallback.")
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(
            api_key=config.OPENROUTER_API_KEY,
            api_base=config.OPENROUTER_BASE_URL,
            model_name="text-embedding-3-small"
        )

    Settings.embed_model = embed_model
    return embed_model


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use dynamically.
    
    Args:
        new_model_id: New LLM model ID to use (e.g., 'meta-llama/llama-3.3-70b-instruct').
    """
    config.LLM_MODEL = new_model_id
    config.LLM_MODEL_ID = new_model_id
    config.config.LLM_MODEL = new_model_id
    config.config.LLM_MODEL_ID = new_model_id
    create_openrouter_llm(model=new_model_id)
    logger.info(f"Changed active LLM model to {new_model_id}")


# Backward-compatible aliases for legacy course code
create_watsonx_llm = create_openrouter_llm
create_watsonx_embedding = create_embedding_model


class LLMInterface:
    """High-level LLM and Embedding interface for RAG pipeline."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.llm_model = llm_model or config.LLM_MODEL
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.temperature = config.TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or config.MAX_NEW_TOKENS

        # Configure LLM and Embedding in Settings
        self.llm = create_openrouter_llm(
            model=self.llm_model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens
        )
        self.embed_model = create_embedding_model(model_name=self.embedding_model)
