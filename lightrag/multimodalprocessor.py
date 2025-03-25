import re
import json
from typing import Optional
from typing import Any, AsyncIterator, List, Dict, Optional, Tuple, cast
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
)
import asyncio
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    statistic_data,
    get_conversation_turns,
    verbose_debug,
)

class MultiModalProcessor:
    def __init__(
        self,
        modal_caption_func,  # 多模态内容生成caption的函数
        text_chunks_db: BaseKVStorage,
        chunks_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        knowledge_graph_inst: BaseGraphStorage,
        embedding_func,
        llm_model_func,
        global_config: dict,
        hashing_kv: Optional[BaseKVStorage] = None,
    ):
        self.modal_caption_func = modal_caption_func
        self.text_chunks_db = text_chunks_db
        self.chunks_vdb = chunks_vdb
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
        self.knowledge_graph_inst = knowledge_graph_inst
        self.embedding_func = embedding_func
        self.llm_model_func = llm_model_func
        self.global_config = global_config
        self.hashing_kv = hashing_kv

    async def process_multimodal_content(
        self,
        modal_content,  # 多模态内容（图像、视频等）
        content_type: str,  # 内容类型，如"image", "video"等
        entity_name: str = None,  # 可选的实体名称
        top_k: int = 10,
        better_than_threshold: float = 0.6,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process multimodal content and create a single entity in the knowledge graph"""
        
        # 1. Generate initial caption
        initial_caption = await self._generate_initial_caption(modal_content, content_type)
        logger.info(f"Initial caption generated: {initial_caption}")
        
        # 2. Retrieve related text chunks
        related_chunks = await self._retrieve_related_chunks(initial_caption, top_k, better_than_threshold)
        
        # 3. Generate enhanced caption
        enhanced_caption = await self._generate_enhanced_caption(initial_caption, related_chunks)
        logger.info(f"Enhanced caption generated: {enhanced_caption}")
        
        # 4. Create entity ID if not provided
        if not entity_name:
            entity_name = f"{content_type}_{compute_mdhash_id(str(modal_content))[:8]}"
        
        # 5. Update knowledge graph with the single entity
        source_id = compute_mdhash_id(str(modal_content))
        
        # Create node data for the entity
        node_data = {
            "entity_type": content_type.upper(),  # Use content type as entity type
            "description": enhanced_caption,
            "source_id": source_id,
            "file_path": "multimodal_content"  # Optional, can be customized
        }
        
        # Add entity to knowledge graph
        await self.knowledge_graph_inst.upsert_node(entity_name, node_data)
        
        # 6. Find potential relationships with existing entities
        relationships = await self._find_related_entities(entity_name, enhanced_caption)
        
        # 7. Add relationships to knowledge graph
        for relation in relationships:
            edge_data = {
                "weight": relation["weight"],
                "keywords": relation["keywords"],
                "description": relation["description"],
                "source_id": source_id
            }
            await self.knowledge_graph_inst.upsert_edge(relation["source"], relation["target"], edge_data)
        
        print("insert done")
        await self._insert_done()

        return enhanced_caption, {
            "entity_name": entity_name, 
            "entity_type": content_type.upper(),
            "description": enhanced_caption,
            "relationships": relationships
        }

    async def _insert_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.knowledge_graph_inst,
                ]
            ]
        )

    async def _generate_initial_caption(self, modal_content, content_type: str) -> str:
        """Generate initial caption using modality-specific LLM"""
        # Cache handling
        content_hash = compute_mdhash_id(str(modal_content))
        args_hash = f"modal_caption_{content_hash}"
        
        cached_result, quantized, min_val, max_val = await handle_cache(
            self.hashing_kv, args_hash, str(content_type), mode="modal_caption", cache_type="modal"
        )
        
        if cached_result:
            return cached_result
        
        # Generate caption
        caption = await self.modal_caption_func(modal_content, content_type)
        
        # Save to cache
        if self.hashing_kv:
            await save_to_cache(
                self.hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=caption,
                    prompt=str(content_type),
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode="modal_caption",
                    cache_type="modal",
                ),
            )
        
        return caption


    async def _retrieve_related_chunks(self, caption: str, top_k: int, better_than_threshold: float) -> List[Dict[str, Any]]:
        """Retrieve relevant text chunks based on caption"""
        # Use vector database to retrieve related text chunks
        results = await self.chunks_vdb.query(
            caption, top_k=top_k,
        )
        
        if not results:
            return None

        return results


    async def _generate_enhanced_caption(self, initial_caption: str, related_chunks: List[Dict[str, Any]]) -> str:
        """Combine original text and initial caption to generate enhanced caption"""
        # Build prompt
        chunks_text = "\n\n".join([chunk["content"] for chunk in related_chunks])
        prompt = f"""Based on the following information, generate a detailed description:
        
    Initial description:
    {initial_caption}

    Related text content:
    {chunks_text}

    Please generate a more detailed and accurate description by combining the initial description with related text content. 
    The description should include key entities, relationships, and important details."""

        # Use LLM to generate enhanced caption
        enhanced_caption = await self.llm_model_func(prompt)
        return enhanced_caption


    async def _find_related_entities(self, entity_name: str, enhanced_caption: str) -> List[Dict[str, Any]]:
        """Find relationships between this entity and existing entities in the knowledge graph using vector similarity"""
        
        # 1. Use entity vector database to find related entities based on enhanced caption
        top_k = 10
        
        # Query the entity vector database using the enhanced caption
        related_entities = await self.entities_vdb.query(
            enhanced_caption, 
            top_k=top_k,
        )
        
        if not related_entities:
            return []
        
        # 2. Filter out the current entity if it already exists
        related_entities = [e for e in related_entities if e["entity_name"] != entity_name]
        
        if not related_entities:
            return []
        
        # 3. Get full entity information from knowledge graph
        entity_names = [entity["entity_name"] for entity in related_entities]
        
        # Build context for each related entity
        entity_contexts = []
        entity_data = {}
        
        for ent_name in entity_names:
            # Get node data from knowledge graph
            node_data = await self.knowledge_graph_inst.get_node(ent_name)
            if node_data:
                entity_data[ent_name] = node_data
                ent_desc = node_data.get("description", "")
                if ent_desc:
                    entity_contexts.append(f"Entity: {ent_name}\nDescription: {ent_desc[:200]}...")
        
        if not entity_contexts:
            return []
        
        entity_contexts_str = "\n\n".join(entity_contexts)
        # 4. Build prompt to find relationships using LLM
        prompt = f"""Analyze potential relationships between the new entity and existing entities.

New entity: {entity_name}
New entity description: {enhanced_caption}

Existing entities:
{entity_contexts_str}

For each potential relationship between the new entity and an existing entity, provide:
1. Source entity name
2. Target entity name
3. Relationship description (detailed explanation of how the entities are related)
4. Relationship keywords (core concepts of the relationship)
5. Relationship strength (a value between 0.0 and 1.0 indicating strength)

Only create relationships where there's a clear connection.
Return results in JSON format:
[
{{
    "source": "source entity name",
    "target": "target entity name",
    "description": "relationship description",
    "keywords": "relationship keywords",
    "weight": "relationship strength"
}}
]"""
        
        # 5. Use LLM to find relationships
        # relationships_json = await self.llm_model_func(prompt)
        # # 6. Parse JSON result
        # relationships = re.search(r"\{.*\}", relationships_json, re.DOTALL)
        # print(relationships.group(0))
        # relationships = json.loads(relationships.group(0))
        # print(relationships)
        relationships_json = """[
    {
        "source": "table_41a6f0d1",
        "target": "Our Company",
        "description": "The product catalog overview provides a comprehensive listing of products that are a core part of the offerings made by Our Company, connecting the catalog to the broader business entity responsible for these products.",
        "keywords": "product offerings, business entity",
        "weight": 0.9
    },
    {
        "source": "table_41a6f0d1",
        "target": "Electronics Category",
        "description": "The catalog includes an extensive section on electronic products, which directly relates to the existing Electronics Category entity, as it encompasses the detailed product offerings within this category.",
        "keywords": "electronic products, product catalog",
        "weight": 0.8
    },
    {
        "source": "table_41a6f0d1",
        "target": "Accessories",
        "description": "The catalog features a range of accessories that enhance user productivity, directly aligning with the existing Accessories entity, which describes similar products.",
        "keywords": "accessories, product catalog",
        "weight": 0.8
    },
    {
        "source": "table_41a6f0d1",
        "target": "Laptop Pro",
        "description": "The product catalog highlights the Laptop Pro, which is a specific product listed in the existing entities, establishing a direct relationship between the catalog and the individual product.",
        "keywords": "high-performance laptop, product catalog",
        "weight": 0.85
    },
    {
        "source": "table_41a6f0d1",
        "target": "Ultra HD Monitor",
        "description": "The catalog features the Ultra HD Monitor, aligning it with the existing entity that provides details about this specific product, thus forming a connection between them.",
        "keywords": "monitor, product catalog",
        "weight": 0.85
    },
    {
        "source": "table_41a6f0d1",
        "target": "Wireless Mouse",
        "description": "The catalog includes the Wireless Mouse, which is referenced in the existing entity, creating a relationship between the product listing and the specific mouse product.",
        "keywords": "mouse, product catalog",
        "weight": 0.75
    },
    {
        "source": "table_41a6f0d1",
        "target": "USB-C Dock",
        "description": "The product catalog mentions the USB-C Dock, linking it to the existing entity that details this accessory, thereby establishing a clear relationship between them.",
        "keywords": "dock, product catalog",
        "weight": 0.75
    },
    {
        "source": "table_41a6f0d1",
        "target": "Smart Watch",
        "description": "The catalog features a Smart Watch collection, relating it to the existing Smart Watch entity that describes the functions and features, creating a connection between product offerings.",
        "keywords": "wearables, product catalog",
        "weight": 0.7
    },
    {
        "source": "table_41a6f0d1",
        "target": "External SSD",
        "description": "The External SSD is included in the product catalog, which correlates with the existing entity that elaborates on this storage solution, thus forming a direct link.",
        "keywords": "storage, product catalog",
        "weight": 0.8
    }
]"""
        relationships = json.loads(relationships_json)
        print(relationships)
        return relationships
