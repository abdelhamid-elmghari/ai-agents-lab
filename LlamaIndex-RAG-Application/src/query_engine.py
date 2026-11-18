"""Module for querying indexed knowledge base and LinkedIn profile data."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    Settings,
)

try:
    from src.llm_interface import create_openrouter_llm
except ImportError:
    from modules.llm_interface import create_openrouter_llm

import config

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """Manages VectorStoreIndex persistence, loading, and querying."""
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        similarity_top_k: Optional[int] = None,
        llm: Optional[Any] = None
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else config.STORAGE_DIR
        self.similarity_top_k = similarity_top_k or config.SIMILARITY_TOP_K
        self.llm = llm or Settings.llm or create_openrouter_llm()
        self.index: Optional[VectorStoreIndex] = None

    def build_index(
        self,
        nodes: Optional[List[Any]] = None,
        documents: Optional[List[Any]] = None,
        persist: bool = True
    ) -> VectorStoreIndex:
        """Builds an index from nodes or documents and optionally persists it to disk."""
        if nodes is not None:
            self.index = VectorStoreIndex(nodes=nodes)
        elif documents is not None:
            self.index = VectorStoreIndex.from_documents(documents)
        else:
            raise ValueError("Must provide either 'nodes' or 'documents' to build_index.")

        if persist:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
            logger.info(f"Persisted vector index to {self.storage_dir}")

        return self.index

    def load_index(self) -> VectorStoreIndex:
        """Loads a persisted index from storage directory."""
        if not self.storage_dir.exists():
            raise FileNotFoundError(f"Storage directory '{self.storage_dir}' does not exist.")

        storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
        self.index = load_index_from_storage(storage_context)
        logger.info(f"Successfully loaded index from {self.storage_dir}")
        return self.index

    def query(self, query_str: str, qa_template: Optional[str] = None) -> Any:
        """Executes a RAG query against the indexed documents.
        
        Args:
            query_str: The user question.
            qa_template: Optional custom prompt template string.
            
        Returns:
            LlamaIndex Response object with response text and source nodes.
        """
        if self.index is None:
            self.load_index()

        template_str = qa_template or config.USER_QUESTION_TEMPLATE
        custom_qa_prompt = PromptTemplate(template_str)

        query_engine = self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            text_qa_template=custom_qa_prompt,
            llm=self.llm
        )
        return query_engine.query(query_str)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person's career or education using OpenRouter.
    
    Args:
        index: VectorStoreIndex containing profile or document data.
        
    Returns:
        String containing interesting facts.
    """
    llm = create_openrouter_llm()
    qa_prompt = PromptTemplate(config.INITIAL_FACTS_TEMPLATE)

    query_engine = index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K,
        text_qa_template=qa_prompt,
        llm=llm
    )
    prompt = "Based on the provided context, list 3 interesting facts about career or education."
    response = query_engine.query(prompt)
    return str(response)


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answers the user's question using the vector database and OpenRouter Llama LLM.
    
    Args:
        index: VectorStoreIndex containing the document data.
        user_query: The user's question.
        
    Returns:
        Response object containing the answer and source nodes.
    """
    llm = create_openrouter_llm()
    qa_prompt = PromptTemplate(config.USER_QUESTION_TEMPLATE)

    query_engine = index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K,
        text_qa_template=qa_prompt,
        llm=llm
    )
    return query_engine.query(user_query)
