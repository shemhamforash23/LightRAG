from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic.v1 import BaseModel

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    EdgeData,
    NodeData,
    QueryParam,
)
from lightrag.core.context_processor import process_context_candidates
from lightrag.core.find_related import (
    find_most_related_edges_from_entities,
    find_most_related_entities_from_relationships,
    find_most_related_text_unit_from_entities,
    find_related_text_unit_from_relationships,
)
from lightrag.core.get_graph_data import (
    get_edge_data,
    get_node_data,
)
from lightrag.core.keywords import extract_keywords_only
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    CacheData,
    compute_args_hash,
    encode_string_by_tiktoken,
    get_conversation_turns,
    handle_cache,
    logger,
    save_to_cache,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, Any],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> dict[str, str] | str:
    """
    Refactored hybrid retrieval combining knowledge graph and vector search.
    Fetches raw candidates, finds related items, combines, ranks, truncates, and formats context.
    """
    # 1. Cache handling (Keep as is for now)
    use_model_func: Callable = (
        query_param.model_func
        if query_param.model_func
        else global_config["llm_model_func"]
    )
    args_hash = compute_args_hash("mix", query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 2. Extract Keywords
    # Extract keywords using extract_keywords_only function which already supports conversation history
    try:
        hl_keywords_list, ll_keywords_list = await extract_keywords_only(
            query, query_param, global_config, hashing_kv
        )
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        hl_keywords_list, ll_keywords_list = [], []

    ll_keywords_str = ", ".join(ll_keywords_list) if ll_keywords_list else ""
    hl_keywords_str = ", ".join(hl_keywords_list) if hl_keywords_list else ""

    # 3. Fetch Initial Candidates in Parallel using a task dictionary
    initial_nodes: list[NodeData] = []
    initial_edges: list[EdgeData] = []
    initial_chunk_results: list[dict[str, Any]] = []

    initial_fetch_tasks: dict[str, asyncio.Task] = {}

    # Task for fetching nodes based on low-level keywords
    if ll_keywords_str:
        initial_fetch_tasks["nodes"] = asyncio.create_task(
            get_node_data(
                query=ll_keywords_str,
                knowledge_graph_inst=knowledge_graph_inst,
                entities_vdb=entities_vdb,
                text_chunks_db=text_chunks_db,
                query_param=query_param,
            )
        )

    # Task for fetching edges based on high-level keywords
    if hl_keywords_str:
        initial_fetch_tasks["edges"] = asyncio.create_task(
            get_edge_data(
                keywords=hl_keywords_str,
                relationships_vdb=relationships_vdb,
                knowledge_graph_inst=knowledge_graph_inst,
                text_chunks_db=text_chunks_db,
                query_param=query_param,
            )
        )

    # Task for fetching chunks based on original query (possibly augmented with history)
    augmented_query = query
    if history_context:
        augmented_query = f"{history_context}\n{query}"

    async def fetch_chunk_vectors() -> list:
        try:
            results = await chunks_vdb.query(
                augmented_query,
                top_k=query_param.top_k,
                ids=query_param.ids,
            )
            return results if results else []
        except Exception as e:
            logger.error(f"Error querying chunks_vdb: {e}")
            return []

    initial_fetch_tasks["chunks"] = asyncio.create_task(fetch_chunk_vectors())

    # Gather results for initial fetch tasks
    if initial_fetch_tasks:
        try:
            initial_task_keys = list(initial_fetch_tasks.keys())
            initial_results = await asyncio.gather(*initial_fetch_tasks.values())
            initial_results_map = dict(zip(initial_task_keys, initial_results))

            # Assign results safely
            initial_nodes = initial_results_map.get("nodes", []) or []
            initial_edges = initial_results_map.get("edges", []) or []
            initial_chunk_results = initial_results_map.get("chunks", []) or []

        except Exception as e:
            logger.error(f"Error during initial data fetching: {e}")

    logger.info(
        f"Fetched initial candidates: {len(initial_nodes)} nodes, {len(initial_edges)} edges, {len(initial_chunk_results)} chunks."
    )

    # 4. Fetch Full Data for Chunks
    full_chunks_data: list[dict[str, Any]] = []
    if initial_chunk_results:
        chunk_ids = [r["id"] for r in initial_chunk_results if "id" in r]
        chunk_scores = {
            r["id"]: r.get("score", 0.0) for r in initial_chunk_results if "id" in r
        }
        try:
            chunk_contents = await text_chunks_db.get_by_ids(chunk_ids)
            # logger.debug(
            #     f"Raw data from text_chunks_db.get_by_ids for IDs {chunk_ids[:3]}...: {chunk_contents[:3]}"
            # )
            for chunk in chunk_contents:
                # logger.debug(f"Chunk keys: {list(chunk.keys())}")
                if chunk is not None and "id" in chunk and "content" in chunk:
                    chunk_id = chunk["id"]
                    chunk["vdb_score"] = chunk_scores.get(chunk_id, 0.0)
                    full_chunks_data.append(chunk)
            logger.info(f"Fetched full data for {len(full_chunks_data)} chunks.")
        except Exception as e:
            logger.error(f"Error fetching full chunk data: {e}")

    # 5. Find Related Items in Parallel
    related_nodes: list[NodeData] = []
    related_edges: list[EdgeData] = []
    related_text_units_from_nodes: list[dict[str, Any]] = []
    related_text_units_from_edges: list[dict[str, Any]] = []

    tasks_to_run: dict[str, asyncio.Task] = {}

    # Tasks related to initial nodes
    if initial_nodes:
        tasks_to_run["edges_from_nodes"] = asyncio.create_task(
            find_most_related_edges_from_entities(
                node_datas=initial_nodes,
                query_param=query_param,
                knowledge_graph_inst=knowledge_graph_inst,
            )
        )
        tasks_to_run["text_from_nodes"] = asyncio.create_task(
            find_most_related_text_unit_from_entities(
                node_datas=initial_nodes,
                query_param=query_param,
                text_chunks_db=text_chunks_db,
                knowledge_graph_inst=knowledge_graph_inst,
                chunks_vdb=chunks_vdb,
                query_text=query,
            )
        )

    # Tasks related to initial edges
    if initial_edges:
        tasks_to_run["nodes_from_edges"] = asyncio.create_task(
            find_most_related_entities_from_relationships(
                edge_datas=initial_edges,
                query_param=query_param,
                knowledge_graph_inst=knowledge_graph_inst,
            )
        )
        tasks_to_run["text_from_edges"] = asyncio.create_task(
            find_related_text_unit_from_relationships(
                edge_datas=initial_edges,
                query_param=query_param,
                text_chunks_db=text_chunks_db,
                knowledge_graph_inst=knowledge_graph_inst,
                chunks_vdb=chunks_vdb,
                query_text=query,
            )
        )

    if tasks_to_run:
        try:
            # Gather results only for the tasks that were actually created
            task_keys = list(tasks_to_run.keys())
            results = await asyncio.gather(*tasks_to_run.values())
            # Map results back based on the order in tasks_to_run.values()
            results_map = dict(zip(task_keys, results))

            # Assign results safely using .get() with default empty list
            related_edges = results_map.get("edges_from_nodes", []) or []
            related_text_from_nodes_raw = results_map.get("text_from_nodes", []) or []
            related_nodes = results_map.get("nodes_from_edges", []) or []
            related_text_from_edges_raw = results_map.get("text_from_edges", []) or []

            # Initialize final lists for related text units
            related_text_units_from_nodes = []
            related_text_units_from_edges = []

            # Safely convert TextChunkSchema lists to Dict lists using .dict()
            if related_text_from_nodes_raw:
                logger.debug(
                    f"Converting {len(related_text_from_nodes_raw)} items from related_text_from_nodes_raw"
                )
                for i, chunk in enumerate(related_text_from_nodes_raw):
                    try:
                        if isinstance(
                            chunk, BaseModel
                        ):  # Check if it's a Pydantic model
                            related_text_units_from_nodes.append(chunk.dict())
                        elif isinstance(chunk, dict):  # If it's already a dict
                            related_text_units_from_nodes.append(chunk)
                        else:
                            logger.warning(
                                f"Item {i} in related_text_from_nodes_raw is not a Pydantic model or dict: {type(chunk)}. Attempting conversion via asdict/vars."
                            )
                            # Fallback attempts
                            try:
                                related_text_units_from_nodes.append(asdict(chunk))
                            except TypeError:
                                try:
                                    related_text_units_from_nodes.append(vars(chunk))
                                except TypeError:
                                    logger.error(
                                        f"Could not convert item {i} from related_text_from_nodes_raw: {chunk}"
                                    )
                    except Exception as conversion_e:
                        logger.error(
                            f"Failed to convert item {i} from related_text_from_nodes_raw: {chunk}. Error: {conversion_e}"
                        )

            if related_text_from_edges_raw:
                logger.debug(
                    f"Converting {len(related_text_from_edges_raw)} items from related_text_from_edges_raw"
                )
                for i, chunk in enumerate(related_text_from_edges_raw):
                    try:
                        if isinstance(chunk, BaseModel):
                            related_text_units_from_edges.append(chunk.dict())
                        elif isinstance(chunk, dict):
                            related_text_units_from_edges.append(chunk)
                        else:
                            logger.warning(
                                f"Item {i} in related_text_from_edges_raw is not a Pydantic model or dict: {type(chunk)}. Attempting conversion via asdict/vars."
                            )
                            # Fallback attempts
                            try:
                                related_text_units_from_edges.append(asdict(chunk))
                            except TypeError:
                                try:
                                    related_text_units_from_edges.append(vars(chunk))
                                except TypeError:
                                    logger.error(
                                        f"Could not convert item {i} from related_text_from_edges_raw: {chunk}"
                                    )
                    except Exception as conversion_e:
                        logger.error(
                            f"Failed to convert item {i} from related_text_from_edges_raw: {chunk}. Error: {conversion_e}"
                        )

            logger.info(
                f"Related items fetch results: {len(related_nodes)} nodes, {len(related_edges)} edges, {len(related_text_units_from_nodes)} text units from nodes, {len(related_text_units_from_edges)} text units from edges."
            )
        except Exception as e:
            # Log the full traceback for better debugging
            logger.exception(
                f"Error occurred during fetching or processing related items: {e}"
            )
            # Ensure lists are initialized even if fetching fails to avoid NameErrors later
            related_nodes, related_edges = [], []
            related_text_from_nodes_raw, related_text_from_edges_raw = [], []

    # 6. Combine All Candidates
    all_nodes = initial_nodes + related_nodes
    all_edges = initial_edges + related_edges
    all_text_units = (
        full_chunks_data + related_text_units_from_nodes + related_text_units_from_edges
    )
    logger.info(
        f"Combined totals: {len(all_nodes)} nodes, {len(all_edges)} edges, {len(all_text_units)} text units before dedup."
    )

    # 7. Deduplicate, Rank, and Truncate using the centralized processor
    # Define keys for deduplication - ensure these match the processor defaults or pass explicitly
    node_id_fields = ["entity_name"]
    edge_id_fields = ["source", "target", "description"]
    # IMPORTANT: Confirm 'id' field exists and is unique for text units in your TextChunkSchema/dict structure
    text_unit_id_fields = ["id"]

    processed_nodes, processed_edges, processed_text_units = process_context_candidates(
        nodes=all_nodes,
        edges=all_edges,
        text_units=all_text_units,
        query_param=query_param,
        node_id_fields=node_id_fields,
        edge_id_fields=edge_id_fields,
        text_unit_id_fields=text_unit_id_fields,
    )

    # 8. Format the final context for LLM
    nodes_context = "\n".join(
        [
            f"{i + 1}. {item.get('entity_name', '')}: {item.get('description', '')}"
            for i, item in enumerate(processed_nodes)
        ]
    )
    edges_context = "\n".join(
        [
            f"{i + 1}. {item.get('source', '')} -> {item.get('target', '')}: {item.get('description', '')}"
            for i, item in enumerate(processed_edges)
        ]
    )
    text_units_context = "\n".join(
        [
            f"{i + 1}. {item.get('content', '')}"
            for i, item in enumerate(processed_text_units)
        ]
    )

    if not nodes_context and not edges_context and not text_units_context:
        return PROMPTS["fail_response"]

    # 4. Merge contexts (using placeholders for now)
    if query_param.only_need_context:
        return {
            "kg_context": nodes_context + "\n" + edges_context,
            "vector_context": text_units_context,
        }

    # 5. Construct hybrid prompt
    sys_prompt = (
        system_prompt
        if system_prompt
        else PROMPTS["mix_rag_response"].format(
            kg_context=nodes_context + "\n" + edges_context
            if nodes_context or edges_context
            else "No relevant knowledge graph information found",
            vector_context=text_units_context
            if text_units_context
            else "No relevant text information found",
            response_type=query_param.response_type,
            history=history_context,
        )
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[mix_kg_vector_query]Prompt Tokens: {len_of_prompts}")

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # Clean up response content
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        # 7. Save cache - Only cache after collecting complete response
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode="mix",
                cache_type="query",
            ),
        )

    return response
