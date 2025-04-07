from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict

import dotenv

# Настройка логгера
dotenv.load_dotenv()
logger = logging.getLogger("prompts")
logger.setLevel(logging.DEBUG)

# Добавляем обработчик для вывода в консоль, если его еще нет
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: Dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Общие промпты, которые не зависят от типа анализируемого контента
PROMPTS["entity_if_loop_extraction"] = """
----Goal---'

It appears some entities may have still been missed.

----Output---

Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

# Промпт для ответа при ошибке
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

# Промпт для ответа на основе базы знаний
PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by Knowledge Base."""

# Промпт для извлечения ключевых слов
PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

# Примеры для извлечения ключевых слов
PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does dependency injection work in Angular?"
################
Output:
{
  "high_level_keywords": ["Dependency Injection", "Angular", "Framework Architecture"],
  "low_level_keywords": ["Injector", "Providers", "@Injectable", "Services", "Injection Hierarchy", "Tokens"]
}
#############################""",
    """Example 2:

Query: "What design patterns are used in React for state management?"
################
Output:
{
  "high_level_keywords": ["Design Patterns", "React", "State Management"],
  "low_level_keywords": ["Redux", "Context API", "Hooks", "Flux", "MobX", "Component State"]
}
#############################""",
    """Example 3:

Query: "How to implement multithreading in Python for data processing?"
################
Output:
{
  "high_level_keywords": ["Multithreading", "Python", "Parallel Processing"],
  "low_level_keywords": ["threading", "multiprocessing", "asyncio", "GIL", "Thread Pool", "Locks", "Semaphores"]
}
#############################""",
    """Example 4:

Query: "Explain SOLID principles in object-oriented programming"
################
Output:
{
  "high_level_keywords": ["SOLID", "Object-Oriented Programming", "Design Principles"],
  "low_level_keywords": ["Single Responsibility Principle", "Open/Closed Principle", "Liskov Substitution Principle", "Interface Segregation Principle", "Dependency Inversion Principle"]
}
#############################""",
    """Example 5:

Query: "What methods exist for optimizing JavaScript performance?"
################
Output:
{
  "high_level_keywords": ["Performance Optimization", "JavaScript", "Web Development"],
  "low_level_keywords": ["Minification", "Lazy Loading", "Memoization", "Virtual DOM", "Profiling", "Event Loop", "Garbage Collection"]
}
#############################""",
]

# Промпт для ответа на основе документов
PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so.
- Do not make anything up.
- Do not include information not provided by the Document Chunks."""

# Промпт для проверки схожести запросов
PROMPTS["similarity_check"] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

# Промпт для ответа на основе смешанных источников
PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sections focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""

# --- Логика загрузки промптов на основе выбранного режима ---

# Сопоставление режимов с модулями в пакете 'lightrag.prompts'
PROMPT_MODULE_MAP: Dict[str, str] = {
    "code": ".prompts.code_prompts",
    "research": ".prompts.research_prompts",
}
DEFAULT_PROMPT_MODE = "code"


def load_mode_prompts() -> None:
    """
    Загружает соответствующий набор промптов в глобальный словарь PROMPTS
    путем динамического импорта модуля, указанного переменной окружения
    LIGHTRAG_PROMPT_MODE из пакета 'lightrag.prompts'.
    """
    global PROMPTS
    mode = os.environ.get("LIGHTRAG_PROMPT_MODE", DEFAULT_PROMPT_MODE).lower()
    module_name = PROMPT_MODULE_MAP.get(mode)

    if not module_name:
        logger.warning(
            f"Invalid LIGHTRAG_PROMPT_MODE '{mode}'. "
            f"Falling back to default mode '{DEFAULT_PROMPT_MODE}'."
        )
        module_name = PROMPT_MODULE_MAP[DEFAULT_PROMPT_MODE]
        mode = DEFAULT_PROMPT_MODE
    else:
        logger.info(f"Using prompt mode: '{mode}'")

    try:
        # Динамически импортируем модуль относительно пакета 'lightrag'
        # Параметр 'package' здесь крайне важен
        prompt_module = importlib.import_module(module_name, package="lightrag")

        if hasattr(prompt_module, "PROMPTS") and isinstance(prompt_module.PROMPTS, dict):
            # Обновляем глобальный словарь PROMPTS соответствующими промптами из модуля
            # Сохраняя при этом базовые промпты, определенные выше
            mode_prompts = prompt_module.PROMPTS
            PROMPTS.update(mode_prompts)
            logger.debug(
                f"Loaded {len(mode_prompts)} prompts from module '{module_name}' for mode '{mode}'."
            )
        else:
            logger.error(
                f"Prompt module '{module_name}' (mode: {mode}) does not contain a valid 'PROMPTS' dictionary. Only base prompts are available."
            )

    except ImportError as e:
        logger.error(
            f"Could not import prompt module '{module_name}' for mode '{mode}': {e}. Only base prompts are available."
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading prompts for mode '{mode}': {e}. Only base prompts are available."
        )


# --- Загружаем промпты при первом импорте модуля ---
load_mode_prompts()
