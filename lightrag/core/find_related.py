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
    TextChunkSchema,
)
from lightrag.prompt import GRAPH_FIELD_SEP
from lightrag.utils import (
    logger,
    split_string_by_multi_markers,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


async def find_most_related_entities_from_relationships(
    edge_datas: list[EdgeData],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
) -> list[NodeData]:
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e.source not in seen:
            entity_names.append(e.source)
            seen.add(e.source)
        if e.target not in seen:
            entity_names.append(e.target)
            seen.add(e.target)

    node_dicts, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    # Combine and create NodeData instances
    node_datas_result: list[NodeData] = []

    # Find max degree for normalization
    max_degree = (
        max(d for d in node_degrees if d is not None)
        if node_degrees and any(d is not None for d in node_degrees)
        else 1
    )

    for k, n_dict, d in zip(entity_names, node_dicts, node_degrees):
        if n_dict is not None:
            # Calculate score based on node degree (normalized to 0.5-1.0 range)
            # Higher degree nodes get higher scores
            degree_score = (
                0.5 + (0.5 * (d / max_degree))
                if d is not None and max_degree > 0
                else 0.5
            )

            node_instance = NodeData(
                entity_name=k,
                description=n_dict.get("description"),
                source_id=n_dict.get("source_id"),
                entity_type=n_dict.get("entity_type"),
                created_at=n_dict.get("created_at"),
                file_path=n_dict.get("file_path"),
                vdb_score=degree_score,  # Use degree-based score instead of 0.0
                # rank=str(d) # Should rank be part of NodeData?
            )
            # Currently, rank is not part of NodeData, consider if it should be or handle differently
            node_datas_result.append(node_instance)

    len_node_datas = len(node_datas_result)
    logger.debug(f"Found {len_node_datas} related entities")

    return node_datas_result


async def find_related_text_unit_from_relationships(
    edge_datas: list[EdgeData],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    chunks_vdb: BaseVectorStorage,
    query_text: str,
) -> list[TextChunkSchema]:
    text_units = [
        split_string_by_multi_markers(dp.source_id, [GRAPH_FIELD_SEP])
        for dp in edge_datas
        if dp.source_id is not None
    ]
    all_text_units_lookup = {}

    # Собираем все уникальные ID чанков
    all_chunk_ids = set()
    for unit_list in text_units:
        for c_id in unit_list:
            all_chunk_ids.add(c_id)

    # Если есть векторная база данных, получаем скоры для чанков
    chunk_scores = {}
    if chunks_vdb and all_chunk_ids:
        try:
            # Используем query_text для получения релевантных скоров
            # и устанавливаем top_k равным количеству ID, чтобы получить все скоры
            logger.debug(f"Querying chunks_vdb with query: {query_text}")
            vector_results = await chunks_vdb.query(
                query=query_text,  # Используем переданный текст запроса
                top_k=len(all_chunk_ids),  # Получаем скоры для всех ID
                ids=list(all_chunk_ids),
            )
            if vector_results:
                # Создаем словарь ID -> скор
                for result in vector_results:
                    if "id" in result and "score" in result:
                        chunk_scores[result["id"]] = result["score"]
        except Exception as e:
            logger.error(f"Error querying chunks_vdb by ids: {e}")

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                # Всегда устанавливаем скор из chunk_scores или значение по умолчанию
                # Игнорируем скор, который может быть в chunk_data
                if c_id in chunk_scores:
                    chunk_data["vdb_score"] = chunk_scores[c_id]
                else:
                    chunk_data["vdb_score"] = (
                        0.5  # Значение по умолчанию, если нет скора из векторной базы
                    )
                    logger.debug(f"Setting default score for chunk {c_id}")
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    logger.debug(f"Found {len(valid_text_units)} valid text chunks")

    text_chunks: list[TextChunkSchema] = [t["data"] for t in valid_text_units]

    return text_chunks


async def find_most_related_text_unit_from_entities(
    node_datas: list[NodeData],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    chunks_vdb: BaseVectorStorage,
    query_text: str,
) -> list[TextChunkSchema]:
    text_units = [
        split_string_by_multi_markers(dp.source_id, [GRAPH_FIELD_SEP])
        for dp in node_datas
        if dp.source_id is not None
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp.entity_name) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = set(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v
    }

    all_text_units_lookup = {}

    # Собираем все уникальные ID чанков
    all_chunk_ids = set()
    for unit_list in text_units:
        for c_id in unit_list:
            all_chunk_ids.add(c_id)

    # Если есть векторная база данных, получаем скоры для чанков
    chunk_scores = {}
    if chunks_vdb and all_chunk_ids:
        try:
            # Используем query_text для получения релевантных скоров
            # и устанавливаем top_k равным количеству ID, чтобы получить все скоры
            vector_results = await chunks_vdb.query(
                query=query_text,  # Используем переданный текст запроса
                top_k=len(all_chunk_ids),  # Получаем скоры для всех ID
                ids=list(all_chunk_ids),
            )
            if vector_results:
                # Создаем словарь ID -> скор
                for result in vector_results:
                    if "id" in result and "score" in result:
                        chunk_scores[result["id"]] = result["score"]
        except Exception as e:
            logger.error(f"Error querying chunks_vdb by ids: {e}")

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                # Всегда устанавливаем скор из chunk_scores или значение по умолчанию
                # Игнорируем скор, который может быть в chunk_data
                if c_id in chunk_scores:
                    chunk_data["vdb_score"] = chunk_scores[c_id]
                else:
                    chunk_data["vdb_score"] = (
                        0.5  # Значение по умолчанию, если нет скора из векторной базы
                    )
                    logger.debug(f"Setting default score for chunk {c_id}")
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    logger.debug(f"Found {len(valid_text_units)} valid text chunks from entities")

    text_chunks: list[TextChunkSchema] = [t["data"] for t in valid_text_units]

    return text_chunks


async def find_most_related_edges_from_entities(
    node_datas: list[NodeData],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
) -> list[EdgeData]:
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp.entity_name) for dp in node_datas]
    )
    all_edges_tuples = []
    seen = set()

    for this_edges in all_related_edges:
        if this_edges is None:
            continue
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges_tuples.append(e)

    all_edges_pack_dict, all_edges_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges_tuples]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges_tuples]
        ),
    )

    all_edges_data: list[EdgeData] = []

    # Find max weight and degree for normalization
    max_weight = (
        max(
            float(v_dict.get("weight", 0.0) or 0.0)
            for v_dict in all_edges_pack_dict
            if v_dict is not None
        )
        if all_edges_pack_dict
        and any(v_dict is not None for v_dict in all_edges_pack_dict)
        else 1.0
    )
    max_degree = (
        max(d for d in all_edges_degree if d is not None)
        if all_edges_degree and any(d is not None for d in all_edges_degree)
        else 1
    )

    for k_tuple, v_dict, d in zip(
        all_edges_tuples, all_edges_pack_dict, all_edges_degree
    ):
        if v_dict is not None:
            # Calculate weight component (normalized to 0-0.5 range)
            weight = float(v_dict.get("weight", 0.0) or 0.0)
            weight_score = 0.5 * (weight / max_weight) if max_weight > 0 else 0.0

            # Calculate degree component (normalized to 0-0.5 range)
            degree_score = (
                0.5 * (d / max_degree) if d is not None and max_degree > 0 else 0.0
            )

            # Combined score (0.0-1.0 range)
            combined_score = weight_score + degree_score

            edge_instance = EdgeData(
                source=k_tuple[0],
                target=k_tuple[1],
                description=v_dict.get("description"),
                keywords=v_dict.get("keywords"),
                source_id=v_dict.get("source_id"),
                weight=float(v_dict.get("weight", 0.0) or 0.0),
                created_at=v_dict.get("created_at"),
                file_path=v_dict.get("file_path"),
                vdb_score=combined_score,  # Use combined score instead of 0.0
                rank=d,
            )
            all_edges_data.append(edge_instance)

    # Sorting needs to access attributes of EdgeData objects
    all_edges_data = sorted(
        all_edges_data,
        key=lambda x: (
            x.rank if x.rank is not None else 0,
            x.weight if x.weight is not None else 0.0,
        ),
        reverse=True,
    )
    logger.debug(f"Found {len(all_edges_data)} related edges")

    return all_edges_data
