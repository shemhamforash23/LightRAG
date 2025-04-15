from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from dotenv import load_dotenv

from lightrag.base import (
    BaseGraphStorage,
)
from lightrag.prompt import GRAPH_FIELD_SEP, PROMPTS
from lightrag.utils import (
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    logger,
    split_string_by_multi_markers,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


async def merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_file_paths.extend(
            split_string_by_multi_markers(already_node["file_path"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    file_path = GRAPH_FIELD_SEP.join(
        set([dp["file_path"] for dp in nodes_data] + already_file_paths)
    )

    logger.debug(f"file_path: {file_path}")
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict[str, Any],
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 0.0 if missing
            weight_value = already_edge.get("weight", 0.0)
            # Convert string weight to float if necessary
            if isinstance(weight_value, str):
                try:
                    weight_value = float(weight_value)
                except (ValueError, TypeError):
                    # If conversion fails, use default value
                    logger.warning(
                        f"Failed to convert weight '{weight_value}' to float, using 0.0"
                    )
                    weight_value = 0.0
            already_weights.append(weight_value)

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    split_string_by_multi_markers(
                        already_edge["source_id"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    split_string_by_multi_markers(
                        already_edge["file_path"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.append(already_edge["description"])

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    # Process edges_data with None checks and ensure all weights are floats
    edge_weights = []
    for dp in edges_data:
        weight_value = dp.get("weight", 0.0)
        # Convert string weight to float if necessary
        if isinstance(weight_value, str):
            try:
                weight_value = float(weight_value)
            except (ValueError, TypeError):
                # If conversion fails, use default value
                logger.warning(
                    f"Failed to convert weight '{weight_value}' to float, using 0.0"
                )
                weight_value = 0.0
        edge_weights.append(weight_value)

    # Sum all weights
    weight = sum(edge_weights + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["description"] for dp in edges_data if dp.get("description")]
                + already_description
            )
        )
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["keywords"] for dp in edges_data if dp.get("keywords")]
                + already_keywords
            )
        )
    )
    source_id = GRAPH_FIELD_SEP.join(
        set(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
        )
    )
    file_path = GRAPH_FIELD_SEP.join(
        set(
            [dp["file_path"] for dp in edges_data if dp.get("file_path")]
            + already_file_paths
        )
    )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "entity_id": need_insert_id,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKNOWN",
                    "file_path": file_path,
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    relationship_type = await _generate_relationship_type(
        src_id, tgt_id, description, keywords, knowledge_graph_inst, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        relationship_type=relationship_type,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        relationship_type=relationship_type,
    )

    return edge_data


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: Callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _generate_relationship_type(
    src_id: str,
    tgt_id: str,
    relationship_description: str,
    relationship_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict[str, Any],
) -> str:
    """Generates a relationship type using LLM based on source/target entity info and relationship context."""
    try:
        # 1. Get node data for source and target entities
        src_node_data = await knowledge_graph_inst.get_node(src_id)
        tgt_node_data = await knowledge_graph_inst.get_node(tgt_id)

        # Handle cases where nodes might not exist (though they should ideally exist before edge creation)
        if not src_node_data:
            logger.warning(
                f"Source node {src_id} not found for relationship type generation."
            )
            src_node_data = {}
        if not tgt_node_data:
            logger.warning(
                f"Target node {tgt_id} not found for relationship type generation."
            )
            tgt_node_data = {}

        # 2. Extract names and descriptions (provide defaults if missing)
        # Assuming entity_id might be the name or there's a 'name' field
        src_name = src_node_data.get("name", src_node_data.get("entity_id", src_id))
        tgt_name = tgt_node_data.get("name", tgt_node_data.get("entity_id", tgt_id))
        src_desc = src_node_data.get("description", "No description available.")
        tgt_desc = tgt_node_data.get("description", "No description available.")

        # 3. Get the prompt template and default types from loaded PROMPTS
        prompt_template = PROMPTS.get("identify_relationship_type")
        default_types = PROMPTS.get("DEFAULT_RELATIONSHIP_TYPES", [])

        if not prompt_template:
            logger.error(
                "'identify_relationship_type' prompt template not found. Falling back to default."
            )
            return "RELATED_TO"

        # 4. Format the prompt
        formatted_prompt = prompt_template.format(
            source_entity_name=src_name,
            source_entity_description=src_desc,
            target_entity_name=tgt_name,
            target_entity_description=tgt_desc,
            relationship_description=relationship_description
            or "No description provided.",
            relationship_keywords=relationship_keywords or "No keywords provided.",
            default_relationship_types="\n".join([f"- {t}" for t in default_types]),
        )

        # 5. Call the LLM using the function from global_config
        use_llm_func: Callable | None = global_config.get("llm_model_func")
        if not use_llm_func:
            logger.error(
                "LLM function 'llm_model_func' not found in global_config. Cannot generate relationship type."
            )
            return "RELATED_TO"  # Fallback if LLM function is missing

        llm_response_str = await use_llm_func(formatted_prompt, max_tokens=50)

        # 6. Process the response
        if llm_response_str:
            # Assuming llm_response_str contains the raw string output
            generated_type = llm_response_str.strip()
            # Basic validation: check if it's UPPER_SNAKE_CASE (simple check)
            if (
                generated_type
                and generated_type.isupper()
                and " " not in generated_type
                and all(c.isalnum() or c == "_" for c in generated_type)
            ):
                logger.info(
                    f"Generated relationship type for ({src_id}, {tgt_id}): {generated_type}"
                )
                return generated_type
            else:
                logger.warning(
                    f"LLM generated invalid relationship type format: '{generated_type}'. Falling back."
                )
        else:
            logger.warning(
                "LLM did not return a valid response for relationship type generation."
            )

    except Exception as e:
        logger.error(
            f"Error during relationship type generation for ({src_id}, {tgt_id}): {e}",
            exc_info=True,
        )

    # Fallback in case of errors or invalid format
    logger.warning(
        f"Falling back to default relationship type for ({src_id}, {tgt_id}): RELATED_TO"
    )
    return "RELATED_TO"
