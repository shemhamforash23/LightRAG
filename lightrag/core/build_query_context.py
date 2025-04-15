from __future__ import annotations

import asyncio
import csv
import io
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    EdgeData,
    NodeData,
    QueryParam,
    TextChunkSchema,
)
from lightrag.core.context_processor import process_context_candidates
from lightrag.core.find_related import (
    find_most_related_text_unit_from_entities,
    find_related_text_unit_from_relationships,
)
from lightrag.core.get_graph_data import get_edge_data, get_node_data
from lightrag.utils import logger


# Helper function to format list of dicts to a CSV string
def _format_data_to_csv_string(data: List[Dict[str, Any]]) -> str:
    """Converts a list of dictionaries to a CSV formatted string with an index column."""
    if not data:
        return ""

    # Use keys from the first item as header, assuming consistency
    # Prepend 'index' to the header
    header = ["index"] + list(data[0].keys())

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(header)
    for i, item in enumerate(data):
        # Create the row values in the same order as the header
        # Add the index (i) as the first element
        row = [i] + [item.get(key, "") for key in header[1:]]  # Skip 'index' in lookup
        writer.writerow(row)

    return output.getvalue().strip()  # Return CSV string without trailing newline


# Async helper to act as a placeholder returning an empty list for gather
async def _return_empty_list_async() -> List:
    return []


# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


async def build_query_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> Optional[str]:
    """
    Builds query context by retrieving relevant graph data and related text sources,
    formats the data into CSV strings, combines contexts using utils.process_combine_contexts,
    and returns a single formatted string wrapped in CSV markdown blocks or None.

    Args:
        ll_keywords: Low-level keywords string for local context retrieval (entities).
        hl_keywords: High-level keywords string for global context retrieval (relationships).
        knowledge_graph_inst: Knowledge graph storage instance.
        entities_vdb: Vector database for entity embeddings.
        relationships_vdb: Vector database for relationship embeddings.
        text_chunks_db: Key-value storage for text chunks.
        query_param: Query parameters including mode (local, global, hybrid).

    Returns:
        A formatted string containing combined entities, relationships, and sources
        within CSV markdown blocks, or None if no relevant context is found.
    """
    # Variables to hold raw candidate lists
    ll_entities_list: List[NodeData] = []
    hl_relations_list: List[EdgeData] = []
    ll_sources_list: List[Dict[str, Any]] = []
    hl_sources_list: List[Dict[str, Any]] = []

    # Variables to hold processed lists
    processed_nodes: List[Dict[str, Any]] = []
    processed_edges: List[Dict[str, Any]] = []
    processed_sources: List[Dict[str, Any]] = []

    logger.info(
        f"Process {os.getpid()} building query context in mode: {query_param.mode}..."
    )

    if query_param.mode == "local":
        ll_entities_list = await get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
        if ll_entities_list:
            # Note: find_most_related_text_unit_from_entities returns List[TextChunkSchema]
            # process_context_candidates expects List[Dict], ensure compatibility or adapt processor
            ll_sources_list_raw: (
                List[TextChunkSchema] | None
            ) = await find_most_related_text_unit_from_entities(
                knowledge_graph_inst=knowledge_graph_inst,
                node_datas=ll_entities_list,
                query_param=query_param,
                text_chunks_db=text_chunks_db,
                chunks_vdb=entities_vdb,
                query_text=ll_keywords,
            )
            # Assuming TextChunkSchema can be treated as dict or convert it:
            ll_sources_list = (
                [dict(chunk) for chunk in ll_sources_list_raw]
                if ll_sources_list_raw
                else []
            )

        # Process the collected candidates
        processed_nodes, processed_edges, processed_sources = (
            process_context_candidates(
                nodes=ll_entities_list,
                edges=[],
                text_units=ll_sources_list,
                query_param=query_param,
            )
        )

    elif query_param.mode == "global":
        hl_relations_list = await get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
        if hl_relations_list:
            hl_sources_list_raw: (
                List[TextChunkSchema] | None
            ) = await find_related_text_unit_from_relationships(
                knowledge_graph_inst=knowledge_graph_inst,
                edge_datas=hl_relations_list,
                query_param=query_param,
                text_chunks_db=text_chunks_db,
                chunks_vdb=relationships_vdb,
                query_text=hl_keywords,
            )
            hl_sources_list = (
                [dict(chunk) for chunk in hl_sources_list_raw]
                if hl_sources_list_raw
                else []
            )

        processed_nodes, processed_edges, processed_sources = (
            process_context_candidates(
                nodes=[],
                edges=hl_relations_list,
                text_units=hl_sources_list,
                query_param=query_param,
            )
        )

    elif query_param.mode == "hybrid":
        # 1. Fetch primary data (nodes and edges) concurrently
        ll_entities_list, hl_relations_list = await asyncio.gather(
            get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            ),
            get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            ),
        )

        # 2. Fetch related sources based on primary data concurrently
        source_tasks = []
        if ll_entities_list:
            source_tasks.append(
                find_most_related_text_unit_from_entities(
                    knowledge_graph_inst=knowledge_graph_inst,
                    node_datas=ll_entities_list,
                    query_param=query_param,
                    text_chunks_db=text_chunks_db,
                    chunks_vdb=entities_vdb,
                    query_text=ll_keywords,
                )
            )
        else:
            source_tasks.append(_return_empty_list_async())

        if hl_relations_list:
            source_tasks.append(
                find_related_text_unit_from_relationships(
                    knowledge_graph_inst=knowledge_graph_inst,
                    edge_datas=hl_relations_list,
                    query_param=query_param,
                    text_chunks_db=text_chunks_db,
                    chunks_vdb=relationships_vdb,
                    query_text=hl_keywords,
                )
            )
        else:
            source_tasks.append(_return_empty_list_async())

        ll_sources_list_raw, hl_sources_list_raw = await asyncio.gather(*source_tasks)
        ll_sources_list = (
            [dict(chunk) for chunk in ll_sources_list_raw]
            if ll_sources_list_raw
            else []
        )
        hl_sources_list = (
            [dict(chunk) for chunk in hl_sources_list_raw]
            if hl_sources_list_raw
            else []
        )

        # 3. Process all collected candidates
        processed_nodes, processed_edges, processed_sources = (
            process_context_candidates(
                nodes=ll_entities_list,  # Primarily LL nodes
                edges=hl_relations_list,  # Primarily HL edges
                text_units=ll_sources_list + hl_sources_list,  # Combine sources
                query_param=query_param,
            )
        )

    else:
        logger.warning(f"Unsupported query mode: {query_param.mode}")
        return None

    # Format the *processed* lists into CSV strings
    combined_entities_csv = _format_data_to_csv_string(processed_nodes)
    combined_relations_csv = _format_data_to_csv_string(processed_edges)
    combined_sources_csv = _format_data_to_csv_string(processed_sources)

    # Check if any context was found after processing
    def is_empty_or_header_only(csv_str):
        return not csv_str or len(csv_str.splitlines()) <= 1

    if (
        is_empty_or_header_only(combined_entities_csv)
        and is_empty_or_header_only(combined_relations_csv)
        and is_empty_or_header_only(combined_sources_csv)
    ):
        logger.info("No relevant context found after processing and combining.")
        return None

    # Format the final result using the combined CSV strings within markdown blocks
    entities_block = (
        combined_entities_csv
        if not is_empty_or_header_only(combined_entities_csv)
        else "No entities found."
    )
    relations_block = (
        combined_relations_csv
        if not is_empty_or_header_only(combined_relations_csv)
        else "No relationships found."
    )
    sources_block = (
        combined_sources_csv
        if not is_empty_or_header_only(combined_sources_csv)
        else "No sources found."
    )

    result = f"""
------Entities-----
```csv
{entities_block}
```
------Relationships-----
```csv
{relations_block}
```
------Sources-----
```csv
{sources_block}
```
""".strip()
    return result
