import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./product_data"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="sk-ZDiJP6MOI3yOr6iL7vOOJ7ohwdhbbuL2jcZe3KDmYMq6nWQ2",
        base_url="https://api.nuwaapi.com/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key="sk-ZDiJP6MOI3yOr6iL7vOOJ7ohwdhbbuL2jcZe3KDmYMq6nWQ2",
        base_url="https://api.nuwaapi.com/v1",
    )


async def modal_caption_func(content, content_type):
    """Function to generate captions for multimodal content"""
    # For markdown tables, we can just pass the content to the LLM to generate a caption
    if content_type == "table":
        prompt = f"""The following is a markdown table. Please generate a brief caption that describes the content and structure of this table:

{content}

Caption:"""
        return await llm_model_func(prompt)
    
    # For other types, we would need specific handlers
    return f"Default caption for {content_type}"


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        modal_caption_func=modal_caption_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


# Create sample markdown table
def create_sample_table():
    markdown_table = """
| Product ID | Product Name | Category | Price | Stock |
|------------|--------------|----------|-------|-------|
| P001 | Laptop Pro | Electronics | $1299.99 | 45 |
| P002 | Ultra HD Monitor | Electronics | $349.99 | 78 |
| P003 | Wireless Mouse | Accessories | $24.99 | 156 |
| P004 | Ergonomic Keyboard | Accessories | $89.99 | 62 |
| P005 | Smart Watch | Wearables | $199.99 | 33 |
| P006 | Bluetooth Speaker | Audio | $59.99 | 92 |
| P007 | External SSD 1TB | Storage | $129.99 | 24 |
| P008 | USB-C Dock | Accessories | $79.99 | 18 |
"""
    return markdown_table


# Create sample product text for context
def create_sample_product_text():
    return """
# Product Catalog Information

Our company offers a wide range of electronic products and accessories designed for professionals and everyday users alike.

## Electronics
Our electronics category includes high-performance laptops and monitors. The Laptop Pro model features the latest processor technology, 16GB RAM, and a 512GB SSD, making it perfect for professionals requiring computing power. Our Ultra HD Monitor delivers exceptional color accuracy and resolution for graphic designers and video editors.

## Accessories
We provide a variety of accessories to enhance your computing experience. Our Wireless Mouse offers precise tracking and long battery life. The Ergonomic Keyboard is designed to reduce wrist strain during long work sessions. The USB-C Dock expands connectivity options for your devices with multiple ports.

## Wearables
Our Smart Watch collection combines style with functionality, offering health tracking, notifications, and long battery life.

## Audio
The Bluetooth Speaker in our catalog delivers rich, room-filling sound with 10 hours of battery life and water resistance for outdoor use.

## Storage
Our External SSD offers 1TB of portable storage with fast transfer speeds for backing up important files or expanding your device capacity.

Inventory levels are updated weekly, with restocking performed for items with less than 20 units in stock. Products in the electronics category typically have a 12-month warranty, while accessories come with a 6-month warranty.
"""


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        
        # Create sample product text file and insert into RAG
        with open("./product_info.txt", "w", encoding="utf-8") as f:
            f.write(create_sample_product_text())
            
        with open("./product_info.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())
        print("Base product text inserted into RAG")
        
        # Create sample markdown table
        markdown_table = create_sample_table()
        
        # Process the table as a multimodal entity
        print("Processing markdown table as multimodal content...")
        enhanced_caption, entity_info = await rag.process_multimodal(
            markdown_table, 
            content_type="table",
            entity_name="ProductCatalogTable",
            top_k=5,
            better_than_threshold=0.6
        )
        
        print("\nEnhanced Caption:")
        print("----------------")
        print(enhanced_caption)
        
        print("\nEntity Information:")
        print("------------------")
        print(f"Entity Name: {entity_info['entity_name']}")
        print(f"Entity Type: {entity_info['entity_type']}")
        
        if entity_info['relationships']:
            print("\nRelationships:")
            print("-------------")
            for rel in entity_info['relationships']:
                print(f"Source: {rel['source']} -> Target: {rel['target']}")
                print(f"Description: {rel['description']}")
                print(f"Keywords: {rel['keywords']}")
                print(f"Weight: {rel['weight']}")
                print()
        
        # Perform a query about the table and product information
        print("Performing queries...")
        
        # Query about specific products in the table
        print("\nQuery 1: What electronic products are in our catalog?")
        result1 = await rag.aquery(
            "What electronic products are in our catalog?", 
            param=QueryParam(mode="mix")
        )
        print(result1)
        
        # Query combining table and context information
        print("\nQuery 2: What accessories do we offer and what are their prices?")
        result2 = await rag.aquery(
            "What accessories do we offer and what are their prices?", 
            param=QueryParam(mode="mix")
        )
        print(result2)
        
        # Query about stock levels and inventory policies
        print("\nQuery 3: Which products have low stock levels and what is our restocking policy?")
        result3 = await rag.aquery(
            "Which products have low stock levels and what is our restocking policy?", 
            param=QueryParam(mode="mix")
        )
        print(result3)
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 