from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    EdgeData,
    NodeData,
    QueryParam,
)
from lightrag.utils import (
    logger,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


async def get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> list[EdgeData]:
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )

    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, ids=query_param.ids
    )

    if not results:
        return []

    edge_tuples_to_fetch = [(r["src_id"], r["tgt_id"]) for r in results]

    raw_edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(src, tgt) for src, tgt in edge_tuples_to_fetch]
    )

    combined_edge_data: list[EdgeData] = []
    for vdb_result, graph_edge_data_dict in zip(results, raw_edge_datas):
        if graph_edge_data_dict is not None:
            edge_instance = EdgeData(
                source=vdb_result["src_id"],
                target=vdb_result["tgt_id"],
                description=graph_edge_data_dict.get("description"),
                keywords=graph_edge_data_dict.get("keywords"),
                source_id=graph_edge_data_dict.get("source_id"),
                weight=float(graph_edge_data_dict.get("weight", 0.0) or 0.0),
                vdb_score=vdb_result.get("score", 0.0),
                rank=None,
                created_at=graph_edge_data_dict.get("created_at"),
                file_path=graph_edge_data_dict.get("file_path"),
            )
            combined_edge_data.append(edge_instance)
        else:
            logger.warning(
                f"Edge ({vdb_result['src_id']}, {vdb_result['tgt_id']}) found in VDB but not in Graph Storage."
            )

    logger.info(f"_get_edge_data found {len(combined_edge_data)} edges.")
    logger.debug(f"_get_edge_data content: {combined_edge_data}")
    return combined_edge_data


async def get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> list[NodeData]:
    """
    Retrieves node data based on a query.

    Args:
        query: The query string to search for entities.
        knowledge_graph_inst: The knowledge graph storage instance.
        entities_vdb: The vector database for entities.
        text_chunks_db: The key-value storage for text chunks.
        query_param: The query parameters.

    Returns:
        A list of dictionaries containing node data.
    """
    # 1. Get similar entities from VDB (assuming results include 'score')
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )
    entities_vdb_results = await entities_vdb.query(
        query, top_k=query_param.top_k, ids=query_param.ids
    )

    if not entities_vdb_results:
        return []

    # Prepare list of entity names to fetch
    entity_names_to_fetch: list[str] = [r["entity_name"] for r in entities_vdb_results]

    # 2. Get entity information from Graph Storage
    # Assuming get_node returns None if node not found
    graph_node_results = await asyncio.gather(
        *[knowledge_graph_inst.get_node(name) for name in entity_names_to_fetch]
    )

    # 3. Combine VDB results (scores) with graph data
    combined_node_data: list[NodeData] = []
    for vdb_result, graph_node_data_dict in zip(
        entities_vdb_results, graph_node_results
    ):
        if graph_node_data_dict is not None:
            node_instance = NodeData(
                entity_name=vdb_result["entity_name"],
                description=graph_node_data_dict.get("description"),
                source_id=graph_node_data_dict.get("source_id"),
                vdb_score=vdb_result.get("score", 0.0),
                created_at=graph_node_data_dict.get("created_at"),
                file_path=graph_node_data_dict.get("file_path"),
                entity_type=graph_node_data_dict.get("entity_type"),
            )
            combined_node_data.append(node_instance)
        else:
            logger.warning(
                f"Node '{vdb_result['entity_name']}' found in VDB but not in Graph Storage."
            )

    # 4. Return the raw list of dictionaries
    logger.info(f"get_node_data found {len(combined_node_data)} nodes.")
    logger.debug(f"get_node_data content: {combined_node_data}")
    return combined_node_data
