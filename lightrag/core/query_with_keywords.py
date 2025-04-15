from lightrag.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, QueryParam
from lightrag.core.keywords import extract_keywords_only
from lightrag.core.kg_query import kg_query_with_keywords
from lightrag.core.mix_kg_vector_query import mix_kg_vector_query
from lightrag.core.naive_query import naive_query


async def query_with_keywords(
    query: str,
    prompt: str,
    param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> dict[str, str] | str:
    """
    Extract keywords from the query and then use them for retrieving information.

    1. Extracts high-level and low-level keywords from the query
    2. Formats the query with the extracted keywords and prompt
    3. Uses the appropriate query method based on param.mode

    Args:
        query: The user's query
        prompt: Additional prompt to prepend to the query
        param: Query parameters
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entities vector database
        relationships_vdb: Relationships vector database
        chunks_vdb: Document chunks vector database
        text_chunks_db: Text chunks storage
        global_config: Global configuration
        hashing_kv: Cache storage

    Returns:
        Query response or async iterator
    """
    # Extract keywords
    hl_keywords, ll_keywords = await extract_keywords_only(
        text=query,
        param=param,
        global_config=global_config,
        hashing_kv=hashing_kv,
    )

    param.hl_keywords = hl_keywords
    param.ll_keywords = ll_keywords

    # Create a new string with the prompt and the keywords
    ll_keywords_str = ", ".join(ll_keywords)
    hl_keywords_str = ", ".join(hl_keywords)
    formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"

    # Use appropriate query method based on mode
    if param.mode in ["local", "global", "hybrid"]:
        return await kg_query_with_keywords(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    elif param.mode == "naive":
        return await naive_query(
            formatted_question,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    elif param.mode == "mix":
        return await mix_kg_vector_query(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    else:
        raise ValueError(f"Unknown mode {param.mode}")
