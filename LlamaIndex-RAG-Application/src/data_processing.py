"""Module for processing and chunking document data for RAG."""

import json
import logging
from typing import Dict, List, Any, Optional, Union

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

try:
    from src.llm_interface import create_embedding_model
except ImportError:
    from modules.llm_interface import create_embedding_model

import config

logger = logging.getLogger(__name__)
                

class DataProcessor:
    """Processes documents into manageable nodes/chunks for retrieval."""
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process_documents(self, documents: List[Document]) -> List[Any]:
        """Splits documents into chunked nodes.
        
        Args:
            documents: List of LlamaIndex Document instances.
            
        Returns:
            List of BaseNode objects.
        """
        if not documents:
            return []
        nodes = self.splitter.get_nodes_from_documents(documents)
        logger.info(f"Generated {len(nodes)} chunks/nodes from {len(documents)} document(s).")
        return nodes


def split_profile_data(profile_data: Union[Dict[str, Any], str]) -> List:
    """Splits LinkedIn profile data into document nodes.
    
    Args:
        profile_data: LinkedIn profile data dictionary or JSON string.
        
    Returns:
        List of document nodes.
    """
    if isinstance(profile_data, dict):
        text = json.dumps(profile_data, indent=2)
    elif isinstance(profile_data, str):
        text = profile_data
    else:
        text = str(profile_data)

    doc = Document(text=text, metadata={"type": "linkedin_profile"})
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.get_nodes_from_documents([doc])


def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """Stores document chunks (nodes) in an in-memory vector index.
    
    Args:
        nodes: List of document nodes to be indexed.
        
    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    try:
        # Ensure embedding model is set
        create_embedding_model()
        index = VectorStoreIndex(nodes=nodes)
        return index
    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        return None


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that nodes in the index have valid embeddings.
    
    Args:
        index: VectorStoreIndex to verify.
        
    Returns:
        True if index contains valid embeddings, False otherwise.
    """
    try:
        docstore = index.docstore
        all_docs = docstore.docs
        if not all_docs:
            return False
        return True
    except Exception as e:
        logger.error(f"Embedding verification failed: {e}")
        return False
