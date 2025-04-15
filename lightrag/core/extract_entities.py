from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Callable

from dotenv import load_dotenv

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from lightrag.core.merge_upsert import (
    merge_edges_then_upsert,
    merge_nodes_then_upsert,
)
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    CacheData,
    clean_str,
    compute_args_hash,
    compute_mdhash_id,
    handle_cache,
    is_float_regex,
    logger,
    pack_user_ass_to_openai_messages,
    save_to_cache,
    split_string_by_multi_markers,
    statistic_data,
    verbose_debug,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, Any],
    pipeline_status: dict,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
) -> None:
    use_llm_func: Callable | None = global_config["llm_model_func"]
    if not use_llm_func:
        logger.error(
            "LLM function 'llm_model_func' not found in global_config. Cannot generate entity relations."
        )
        return
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    enable_llm_cache_for_entity_extract: bool = global_config[
        "enable_llm_cache_for_entity_extract"
    ]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples_formatted = examples

    for key, value in example_context_base.items():
        placeholder = "{" + key + "}"
        examples_formatted = examples_formatted.replace(placeholder, str(value))

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples_formatted,
        language=language,
    )

    continue_prompt_template = PROMPTS["entity_continue_extraction"]
    continue_prompt = continue_prompt_template

    for key, value in context_base.items():
        placeholder = "{" + key + "}"
        continue_prompt = continue_prompt.replace(placeholder, str(value))

    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _user_llm_func_with_cache(
        input_text: str, history_messages: list[dict[str, str]] | None = None
    ) -> str:
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            if history_messages:
                history = json.dumps(history_messages, ensure_ascii=False)
                _prompt = history + "\n" + input_text
            else:
                _prompt = input_text

            # TODO： add cache_type="extract"
            arg_hash = compute_args_hash(_prompt)
            cached_return, _1, _2, _3 = await handle_cache(
                llm_response_cache,
                arg_hash,
                _prompt,
                "default",
                cache_type="extract",
            )
            if cached_return:
                logger.debug(f"Found cache for {arg_hash}")
                statistic_data["llm_cache"] += 1
                return cached_return
            statistic_data["llm_call"] += 1
            res: str
            if history_messages:
                res = await use_llm_func(input_text, history_messages=history_messages)
            else:
                res = await use_llm_func(input_text)
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type="extract",
                ),
            )
            return res

        if history_messages:
            return await use_llm_func(input_text, history_messages=history_messages)
        else:
            return await use_llm_func(input_text)

    async def _process_extraction_result(
        result: str, chunk_key: str, file_path: str = "unknown_source"
    ):
        """Process a single extraction result (either initial or gleaning)
        Args:
            result (str): The extraction result to process
            chunk_key (str): The chunk key for source tracking
            file_path (str): The file path for citation
        Returns:
            tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
        """
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue
            record = match.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        return maybe_nodes, maybe_edges

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        """
        nonlocal processed_chunks
        chunk_key: str = chunk_key_dp[0]
        chunk_dp: TextChunkSchema = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path: str = chunk_dp.get("file_path", "unknown_source")  # type: ignore

        # Get initial extraction
        hint_prompt_template = entity_extract_prompt
        hint_prompt = hint_prompt_template

        # Apply safe formatting
        for key, value in context_base.items():
            placeholder = "{" + key + "}"
            hint_prompt = hint_prompt.replace(placeholder, str(value))

        # Now safely add the input_text
        hint_prompt = hint_prompt.replace("{input_text}", content)

        final_result = await _user_llm_func_with_cache(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result, chunk_key, file_path
        )

        # Process additional gleaning results
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(
                continue_prompt, history_messages=history
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # Process gleaning result separately with file path
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result, chunk_key, file_path
            )

            # Merge results
            for entity_name, entities in glean_nodes.items():
                maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items():
                maybe_edges[edge_key].extend(edges)

            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await _user_llm_func_with_cache(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"  Chk {processed_chunks}/{total_chunks}: extracted {entities_count} Ent + {relations_count} Rel (deduplicated)"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
        return dict(maybe_nodes), dict(maybe_edges)

    tasks = [_process_single_content(c) for c in ordered_chunks]
    results = await asyncio.gather(*tasks)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    from lightrag.kg.shared_storage import get_graph_db_lock

    graph_db_lock = get_graph_db_lock(enable_logging=False)

    # Ensure that nodes and edges are merged and upserted atomically
    async with graph_db_lock:
        all_entities_data = await asyncio.gather(
            *[
                merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        )

        all_relationships_data = await asyncio.gather(
            *[
                merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        )

    if not (all_entities_data or all_relationships_data):
        log_message = "Didn't extract any entities and relationships."
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
        return

    if not all_entities_data:
        log_message = "Didn't extract any entities"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
    if not all_relationships_data:
        log_message = "Didn't extract any relationships"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    log_message = f"Extracted {len(all_entities_data)} entities + {len(all_relationships_data)} relationships (deduplicated)"
    logger.info(log_message)
    if pipeline_status is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)
    verbose_debug(
        f"New entities:{all_entities_data}, relationships:{all_relationships_data}"
    )
    verbose_debug(f"New relationships:{all_relationships_data}")

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "entity_name": dp["entity_name"],
                "entity_type": dp["entity_type"],
                "content": f"{dp['entity_name']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "file_path": dp.get("file_path", "unknown_source"),
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "keywords": dp["keywords"],
                "content": f"{dp['src_id']}\t{dp['tgt_id']}\n{dp['keywords']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "file_path": dp.get("file_path", "unknown_source"),
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    # Clean and validate entity name
    entity_name = clean_str(record_attributes[1]).strip('"')
    if not entity_name.strip():
        logger.warning(
            f"Entity extraction error: empty entity name in: {record_attributes}"
        )
        return None

    # Clean and validate entity type
    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(
            f"Entity extraction error: invalid entity type in: {record_attributes}"
        )
        return None

    # Clean and validate description
    entity_description = clean_str(record_attributes[3]).strip('"')
    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        file_path=file_path,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1]).strip('"')
    target = clean_str(record_attributes[2]).strip('"')
    edge_description = clean_str(record_attributes[3]).strip('"')
    edge_keywords = clean_str(record_attributes[4]).strip('"')
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"'))
        if is_float_regex(record_attributes[-1])
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        file_path=file_path,
    )
