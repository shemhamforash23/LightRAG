from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Literal,
    TypedDict,
    TypeVar,
)

import numpy as np
from dotenv import load_dotenv

from .types import KnowledgeGraph
from .utils import EmbeddingFunc

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


T = TypeVar("T")


@dataclass
class QueryParam:
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval.
    """

    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""

    only_need_prompt: bool = False
    """If True, only returns the generated prompt without producing a response."""

    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""

    stream: bool = False
    """If True, enables streaming output for real-time responses."""

    top_k: int = int(os.getenv("TOP_K", "60"))
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    max_token_for_text_unit: int = int(os.getenv("MAX_TOKEN_TEXT_CHUNK", "4000"))
    """Maximum number of tokens allowed for each retrieved text chunk."""

    max_token_for_global_context: int = int(os.getenv("MAX_TOKEN_RELATION_DESC", "4000"))
    """Maximum number of tokens allocated for relationship descriptions in global retrieval."""

    max_token_for_local_context: int = int(os.getenv("MAX_TOKEN_ENTITY_DESC", "4000"))
    """Maximum number of tokens allocated for entity descriptions in local retrieval."""

    hl_keywords: list[str] = field(default_factory=list)
    """List of high-level keywords to prioritize in retrieval."""

    ll_keywords: list[str] = field(default_factory=list)
    """List of low-level keywords to refine retrieval focus."""

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    history_turns: int = 3
    """Number of complete conversation turns (user-assistant pairs) to consider in the response context."""

    ids: list[str] | None = None
    """List of ids to filter the results."""

    model_func: Callable[..., object] | None = None
    """Optional override for the LLM model function to use for this specific query.
    If provided, this will be used instead of the global model function.
    This allows using different models for different query modes.
    """


@dataclass
class StorageNameSpace(ABC):
    namespace: str
    global_config: dict[str, Any]

    async def initialize(self):
        """Initialize the storage"""
        pass

    async def finalize(self):
        """Finalize the storage"""
        pass

    @abstractmethod
    async def index_done_callback(self) -> None:
        """Commit the storage operations after indexing"""


@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc
    cosine_better_than_threshold: float = field(default=0.2)
    meta_fields: set[str] = field(default_factory=set)

    @abstractmethod
    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Query the vector storage and retrieve top_k results."""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vectors in the storage."""

    @abstractmethod
    async def delete_entity(self, entity_name: str) -> None:
        """Delete a single entity by its name."""

    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations for a given entity."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with the specified IDs from the storage

        Args:
            ids: List of vector IDs to be deleted
        """

    @abstractmethod
    async def get_entities_by_source_id(self, source_id: str) -> list[dict[str, Any]]:
        """Get all entities that reference the given source_id

        Args:
            source_id: Source ID (usually a chunk ID) to find related entities

        Returns:
            List of entities that reference the source_id
        """

    @abstractmethod
    async def get_relations_by_source_id(self, source_id: str) -> list[dict[str, Any]]:
        """Get all relations that reference the given source_id

        Args:
            source_id: Source ID (usually a chunk ID) to find related relations

        Returns:
            List of relations that reference the source_id
        """

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        pass


@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get value by id"""

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get values by ids"""

    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return un-exist keys"""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data"""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete records with the specified IDs from the storage

        Args:
            ids: List of record IDs to be deleted
        """

    @abstractmethod
    async def get_all(self) -> dict[str, dict[str, Any]]:
        """Get all records from the storage

        Returns:
            Dictionary where keys are record IDs and values are record data
        """

    @abstractmethod
    async def get_chunks_by_doc_id(self, doc_id: str) -> dict[str, dict[str, Any]]:
        """Get all chunks associated with a document ID

        Args:
            doc_id: Document ID to get chunks for

        Returns:
            Dictionary where keys are chunk IDs and values are chunk data
        """


@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if an edge exists in the graph."""

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Get the degree of a node."""

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """Get the degree of an edge."""

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get a node by its id."""

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get an edge by its source and target node ids."""

    @abstractmethod
    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get all edges connected to a node."""

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Upsert a node into the graph."""

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert an edge into the graph."""

    @abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Delete a node from the graph."""

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a single node from the graph."""

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]) -> None:
        """Removes multiple nodes from the graph

        Args:
            nodes: List of node IDs to be deleted
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Removes multiple edges from the graph

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """

    @abstractmethod
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray[Any, Any], list[str]]:
        """Embed nodes using an algorithm."""

    @abstractmethod
    async def get_all_labels(self) -> list[str]:
        """Get all labels in the graph."""

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, min_degree: int = 0, inclusive: bool = False
    ) -> KnowledgeGraph:
        """Retrieve a subgraph of the knowledge graph starting from a given node."""


class DocStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    content: str
    """Original content of the document"""
    content_summary: str
    """First 100 chars of document content, used for preview"""
    content_length: int
    """Total length of document"""
    file_path: str
    """File path of the document"""
    status: DocStatus
    """Current processing status"""
    created_at: str
    """ISO format timestamp when document was created"""
    updated_at: str
    """ISO format timestamp when document was last updated"""
    chunks_count: int | None = None
    """Number of chunks after splitting, used for processing"""
    error: str | None = None
    """Error message if failed"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Base class for document status storage"""

    @abstractmethod
    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""

    @abstractmethod
    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete document status records with the specified IDs

        Args:
            ids: List of document IDs to be deleted
        """


class StoragesStatus(str, Enum):
    """Storages status"""

    NOT_CREATED = "not_created"
    CREATED = "created"
    INITIALIZED = "initialized"
    FINALIZED = "finalized"
