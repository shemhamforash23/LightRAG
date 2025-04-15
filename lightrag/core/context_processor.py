from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Hashable, List, Sequence, Tuple, Union

from lightrag.base import EdgeData, NodeData, QueryParam
from lightrag.utils import logger, truncate_list_by_token_size


def _get_unique_identifier(
    item: Union[NodeData, EdgeData, Dict[str, Any]], id_fields: List[str]
) -> Hashable:
    """Generate a unique identifier for an item (dict or dataclass) based on specified fields."""
    try:
        if isinstance(item, dict):
            ids = tuple(item.get(field) for field in id_fields)
        else:  # Assume dataclass or object with attributes
            ids = tuple(getattr(item, field, None) for field in id_fields)
        # Handle potential non-hashable types like lists/dicts if necessary
        # For simplicity, assuming fields yield hashable types or None
        return ids
    except AttributeError as e:
        logger.error(f"Error accessing ID fields {id_fields} for item {item}: {e}")
        # Fallback identifier or raise error
        return json.dumps(item, sort_keys=True)  # Less efficient fallback


def _dedup_and_rank(
    items: Sequence[Union[NodeData, EdgeData, Dict[str, Any]]], id_fields: List[str]
) -> List[Union[NodeData, EdgeData, Dict[str, Any]]]:
    """Deduplicates items based on a composite ID and ranks them by vdb_score."""
    unique_items: Dict[Hashable, Union[NodeData, EdgeData, Dict[str, Any]]] = {}
    for i, item in enumerate(items):
        try:
            unique_id = _get_unique_identifier(item, id_fields)
            existing_item = unique_items.get(unique_id)

            # Determine the vdb_score safely
            current_score = getattr(item, "vdb_score", None)
            if current_score is None and isinstance(item, dict):
                current_score = item.get("vdb_score")

            if existing_item:
                existing_score = getattr(existing_item, "vdb_score", None)
                if existing_score is None and isinstance(existing_item, dict):
                    existing_score = existing_item.get("vdb_score")

                # Keep the item with the higher score
                if current_score is not None and (
                    existing_score is None or current_score > existing_score
                ):
                    unique_items[unique_id] = item
                else:
                    pass
            else:
                unique_items[unique_id] = item
        except Exception as e:
            logger.exception(
                f"Error processing item {i} in _dedup_and_rank: {item}. Error: {e}"
            )
            # Decide whether to skip the item or re-raise the exception
            # For now, let's log and skip to potentially process other items
            continue

    # Prepare for sorting
    items_to_sort = list(unique_items.values())

    try:
        # Sort items by vdb_score in descending order, handling None scores
        ranked_items = sorted(
            unique_items.values(),
            key=lambda x:
            # Get score based on the type of the item
            (
                # For NodeData and EdgeData, access vdb_score attribute directly
                getattr(x, "vdb_score", 0.0)
                if isinstance(x, (NodeData, EdgeData))
                # For dictionaries, use get method
                else x.get("vdb_score", 0.0)
                if isinstance(x, dict)
                # Default to 0.0 if score is None or not present
                else 0.0
            )
            or 0.0,  # Convert None to 0.0 for comparison
            reverse=True,
        )
    except TypeError as e:
        logger.exception(f"Error during sorting in _dedup_and_rank: {e}")
        # Return unsorted items or an empty list in case of sorting error
        return items_to_sort

    return ranked_items


# Token key extraction functions adjusted for dataclasses


def _node_token_key(node: Union[NodeData, Dict[str, Any]]) -> str:
    """Extracts the 'description' for token calculation from NodeData or dict."""
    if isinstance(node, dict):
        # For dictionaries, safely get description with empty string as default
        return node.get("description", "") or ""
    # For NodeData objects, check for None and return empty string in that case
    return node.description or ""


def _edge_token_key(edge: Union[EdgeData, Dict[str, Any]]) -> str:
    """Extracts the 'description' for token calculation from EdgeData or dict."""
    if isinstance(edge, dict):
        # For dictionaries, safely get description with empty string as default
        return edge.get("description", "") or ""
    # For EdgeData objects, check for None and return empty string in that case
    return edge.description or ""


def _text_unit_token_key(text_unit: Dict[str, Any]) -> str:
    """Extracts the 'content' for token calculation from TextChunkSchema (as dict)."""
    return text_unit.get("content", "")


def process_context_candidates(
    nodes: List[NodeData],  # Use NodeData
    edges: List[EdgeData],  # Use EdgeData
    text_units: List[Dict[str, Any]],  # Keep as Dict for now
    query_param: QueryParam,
    node_id_fields: List[str] = ["entity_name"],
    edge_id_fields: List[str] = ["source", "target"],  # Simplified default edge ID
    text_unit_id_fields: List[str] = ["id"],  # Assuming 'id' exists and is unique
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
]:  # Return Dicts
    """
    Processes candidate lists for nodes (NodeData), edges (EdgeData), and text units (Dict)
    by deduplicating, ranking (based on 'vdb_score'), and truncating based on token limits.

    Returns processed lists as dictionaries.
    """
    logger.info(
        f"Starting context processing for {len(nodes)} nodes, {len(edges)} edges, {len(text_units)} text units."
    )

    # 1. Deduplicate and Rank
    ranked_nodes = _dedup_and_rank(nodes, node_id_fields)

    ranked_edges = _dedup_and_rank(edges, edge_id_fields)

    ranked_text_units = _dedup_and_rank(text_units, text_unit_id_fields)

    # 2. Truncate based on token limits
    truncated_nodes = truncate_list_by_token_size(
        ranked_nodes, query_param.max_token_for_local_context, _node_token_key
    )

    truncated_edges = truncate_list_by_token_size(
        ranked_edges, query_param.max_token_for_global_context, _edge_token_key
    )

    truncated_text_units = truncate_list_by_token_size(
        ranked_text_units, query_param.max_token_for_text_unit, _text_unit_token_key
    )

    # 3. Convert back to List[Dict] for consistent output
    # Ensure asdict works correctly for dataclasses
    try:
        processed_nodes_dict = [asdict(node) for node in truncated_nodes]

        processed_edges_dict = [asdict(edge) for edge in truncated_edges]

        # Text units are already dicts, but ensure they are if _dedup_and_rank changed them
        processed_text_units_dict = [
            unit
            if isinstance(unit, dict)
            else asdict(unit)
            if hasattr(unit, "__dataclass_fields__")
            else vars(unit)
            for unit in truncated_text_units
        ]

    except Exception as e:
        logger.exception(
            f"Error during final conversion to dictionaries in process_context_candidates: {e}"
        )
        # Return empty lists in case of conversion error
        return [], [], []

    logger.info(
        f"Finished context processing. Returning {len(processed_nodes_dict)} nodes, {len(processed_edges_dict)} edges, {len(processed_text_units_dict)} text units."
    )
    return processed_nodes_dict, processed_edges_dict, processed_text_units_dict
