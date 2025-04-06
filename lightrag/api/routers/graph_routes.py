"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from lightrag.lightrag import LightRAG

from ..utils_api import get_combined_auth_dependency

router = APIRouter(tags=["graph"])


class EntityRequest(BaseModel):
    entity_type: str = Field(description="Type of the entity")
    description: str = Field(description="Description of the entity")
    source_id: str = Field(description="Source ID of the entity")


class RelationRequest(BaseModel):
    description: str = Field(description="Description of the relation")
    keywords: str = Field(description="Keywords of the relation")
    source_id: Optional[str] = Field(description="Source ID of the relation")
    weight: Optional[float] = Field(description="Weight of the relation")


class MergeEntitiesRequest(BaseModel):
    source_entities: List[str] = Field(
        description="List of source entities to merge",
    )
    target_entity: str = Field(
        description="Name of the target entity after merging",
    )
    merge_strategy: Optional[Dict[str, str]] = Field(
        description="Merge strategy for properties ('max', 'min', 'concat', 'first', 'last'). Example: {\"description\": \"concat\", \"weight\": \"max\"}",
        default=None,
    )


class StatusMessageResponse(BaseModel):
    message: str = Field(description="Status message")


class EntityResponse(BaseModel):
    entity_name: str = Field(description="Name of the entity")
    source_id: Optional[str] = Field(description="Source ID of the entity")
    graph_data: Optional[Dict[str, Any]] = Field(description="Graph data of the entity")


class RelationResponse(BaseModel):
    src_entity: str = Field(description="Source entity of the relation")
    tgt_entity: str = Field(description="Target entity of the relation")
    source_id: Optional[str] = Field(description="Source ID of the relation")
    graph_data: Optional[Dict[str, Any]] = Field(description="Graph data of the relation")


def create_graph_routes(rag: LightRAG, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def get_graph_labels():
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        return await rag.get_graph_labels()

    @router.get("/graphs", dependencies=[Depends(combined_auth)])
    async def get_knowledge_graph(
        label: str, max_depth: int = 3, min_degree: int = 0, inclusive: bool = False
    ):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        Maximum number of nodes is constrained by the environment variable `MAX_GRAPH_NODES` (default: 1000).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. min_degree does not affect nodes directly connected to the matching nodes
            2. Label matching nodes take precedence
            3. Followed by nodes directly connected to the matching nodes
            4. Finally, the degree of the nodes
        Maximum number of nodes is limited to env MAX_GRAPH_NODES(default: 1000)

        Args:
            label (str): Label to get knowledge graph for
            max_depth (int, optional): Maximum depth of graph. Defaults to 3.
            inclusive_search (bool, optional): If True, search for nodes that include the label. Defaults to False.
            min_degree (int, optional): Minimum degree of nodes. Defaults to 0.

        Returns:
            Dict[str, List[str]]: Knowledge graph for label
        """
        return await rag.get_knowledge_graph(
            node_label=label,
            max_depth=max_depth,
            inclusive=inclusive,
            min_degree=min_degree,
        )

    @router.delete(
        "/entities/{entity_name}",
        response_model=StatusMessageResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_entity(entity_name: str):
        """
        Delete a single entity and its relationships from the graph

        Args:
            entity_name: Name of the entity to delete
        """
        try:
            await rag.adelete_by_entity(entity_name)
            return StatusMessageResponse(message="Entity deleted successfully")
        except Exception as e:
            return StatusMessageResponse(message=f"Failed to delete entity: {e}")

    @router.delete(
        "/documents/{doc_id}",
        response_model=StatusMessageResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_by_doc_id(doc_id: str):
        """
        Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            await rag.adelete_by_doc_id(doc_id)
            return StatusMessageResponse(message="Document deleted successfully")
        except Exception as e:
            return StatusMessageResponse(message=f"Failed to delete document: {e}")

    @router.post(
        "/entities/{entity_name}",
        response_model=EntityResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_entity(entity_name: str, data: EntityRequest):
        """
        Creates a new entity in the knowledge graph and adds it to the vector database.

        Args:
            entity_name: Name of the new entity
            entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

        Returns:
            Dictionary containing created entity information
        """
        result = await rag.acreate_entity(entity_name, data.model_dump())
        return EntityResponse(**result)

    @router.put(
        "/entities/{entity_name}",
        response_model=EntityResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def edit_entity(entity_name: str, data: EntityRequest):
        """
        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.

        Args:
            entity_name: Name of the entity to edit
            data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}

        Returns:
            Dictionary containing updated entity information
        """
        result = await rag.aedit_entity(entity_name, data.model_dump())
        return EntityResponse(**result)

    @router.post(
        "/relations/{source}/{target}",
        response_model=RelationResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_relation(source: str, target: str, data: RelationRequest):
        """
        Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

        Args:
            source: Name of the source entity
            target: Name of the target entity
            data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

        Returns:
            Dictionary containing created relation information
        """
        result = await rag.acreate_relation(source, target, data.model_dump())
        return RelationResponse(**result)

    @router.put(
        "/relations/{source}/{target}",
        response_model=RelationResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def edit_relation(source: str, relation_type: str, target: str, data: RelationRequest):
        """
        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.

        Args:
            source: Name of the source entity
            relation_type: Type of the relation
            target: Name of the target entity
            data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}

        Returns:
            Dictionary containing updated relation information
        """
        result = await rag.aedit_relation(source, target, data.model_dump())
        return RelationResponse(**result)

    # Эндпоинт для слияния сущностей
    @router.post("/merge", response_model=EntityResponse, dependencies=[Depends(combined_auth)])
    async def merge_entities(data: MergeEntitiesRequest):
        """
        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
                Supported strategies:
                - "concatenate": Concatenate all values (for text fields)
                - "keep_first": Keep the first non-empty value
                - "keep_last": Keep the last non-empty value
                - "join_unique": Join all unique values (for fields separated by delimiter)            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        result = await rag.amerge_entities(
            source_entities=data.source_entities,
            target_entity=data.target_entity,
            merge_strategy=data.merge_strategy,
        )
        return EntityResponse(**result)

    return router
