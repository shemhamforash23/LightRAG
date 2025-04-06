from __future__ import annotations

from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "module",
    "class",
    "function",
    "method",
    "variable",
    "interface",
    "library",
    "framework",
]

PROMPTS["entity_extraction"] = """---Goal---
Given a code snippet or documentation that is potentially relevant to this activity and a list of entity types, identify all code entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all code entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. Preserve the exact casing of the entity name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's purpose, functionality, and implementation details
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other (e.g., inheritance, composition, dependency, function call)
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details (e.g., "imports", "extends", "implements", "calls", "uses")
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire code. These should capture the overarching ideas, patterns, or architecture present in the code.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [class, method, function, module, library]
Text:
```python
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        self.data = np.loadtxt(self.data_path, delimiter=',')
        return self.data
    
    def preprocess(self, normalize=True):
        if self.data is None:
            self.load_data()
        
        if normalize:
            self.data = (self.data - np.mean(self.data, axis=0)) / np.std(self.data, axis=0)
        
        return self.data

def split_dataset(data, test_size=0.2, random_state=42):
    X = data[:, :-1]
    y = data[:, -1]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
```

Output:
("entity"{tuple_delimiter}"numpy"{tuple_delimiter}"library"{tuple_delimiter}"NumPy is a library for numerical computing in Python, providing support for arrays, matrices, and mathematical functions."){record_delimiter}
("entity"{tuple_delimiter}"sklearn.model_selection"{tuple_delimiter}"module"{tuple_delimiter}"A module from scikit-learn that provides utilities for splitting datasets into training and testing sets."){record_delimiter}
("entity"{tuple_delimiter}"train_test_split"{tuple_delimiter}"function"{tuple_delimiter}"A function from sklearn.model_selection that splits arrays or matrices into random train and test subsets."){record_delimiter}
("entity"{tuple_delimiter}"DataProcessor"{tuple_delimiter}"class"{tuple_delimiter}"A class designed to load and preprocess data from a specified file path, with options for normalization."){record_delimiter}
("entity"{tuple_delimiter}"__init__"{tuple_delimiter}"method"{tuple_delimiter}"Constructor method for the DataProcessor class that initializes the data_path attribute and sets data to None."){record_delimiter}
("entity"{tuple_delimiter}"load_data"{tuple_delimiter}"method"{tuple_delimiter}"Method of DataProcessor that loads data from the specified file path using numpy's loadtxt function."){record_delimiter}
("entity"{tuple_delimiter}"preprocess"{tuple_delimiter}"method"{tuple_delimiter}"Method of DataProcessor that normalizes data if specified, ensuring data is loaded first if it hasn't been already."){record_delimiter}
("entity"{tuple_delimiter}"split_dataset"{tuple_delimiter}"function"{tuple_delimiter}"A standalone function that splits a dataset into features and target, then into training and testing sets using sklearn's train_test_split."){record_delimiter}
("relationship"{tuple_delimiter}"DataProcessor"{tuple_delimiter}"load_data"{tuple_delimiter}"The DataProcessor class contains the load_data method which is responsible for loading data from a file."{tuple_delimiter}"class method, data loading"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"DataProcessor"{tuple_delimiter}"preprocess"{tuple_delimiter}"The DataProcessor class contains the preprocess method which is responsible for normalizing the data."{tuple_delimiter}"class method, data transformation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"preprocess"{tuple_delimiter}"load_data"{tuple_delimiter}"The preprocess method calls the load_data method if data hasn't been loaded yet."{tuple_delimiter}"method call, dependency"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"load_data"{tuple_delimiter}"numpy"{tuple_delimiter}"The load_data method uses numpy's loadtxt function to read data from a file."{tuple_delimiter}"library usage, data loading"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"preprocess"{tuple_delimiter}"numpy"{tuple_delimiter}"The preprocess method uses numpy functions like mean and std for data normalization."{tuple_delimiter}"library usage, data processing"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"split_dataset"{tuple_delimiter}"train_test_split"{tuple_delimiter}"The split_dataset function uses the train_test_split function from sklearn to divide data into training and testing sets."{tuple_delimiter}"function call, data splitting"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"data processing, machine learning preprocessing, dataset splitting, normalization, numpy operations"){completion_delimiter}

#############################""",
    """
Example 2:

Entity_types: [class, function, module, variable, interface]
Text:
```python
# Authentication service for user management
from typing import Optional, Dict, Any
import requests
import json
import os

class AuthService:
    \\\"\\\"\\\"Service for handling authentication operations\\\"\\\"\\\"\"
    
    def __init__(self, base_url: str):
        \\\"\\\"\\\"Initialize the auth service with base URL\\\"\\\"\\\"\"
        self.api_url = base_url + '/auth'
        self.current_user_key = 'current_user'
        self.session = requests.Session()
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        \\\"\\\"\\\"Log in a user with email and password\\\"\\\"\\\"\"
        response = self.session.post(
            self.api_url + '/login',
            data=dict(email=email, password=password)
        )
        response.raise_for_status()
        data = response.json()
        
        # Store user details and token
        os.environ[self.current_user_key] = json.dumps(data)
        return data.get('user')
    
    def logout(self) -> None:
        \\\"\\\"\\\"Log out the current user\\\"\\\"\\\"\"
        if self.current_user_key in os.environ:
            del os.environ[self.current_user_key]
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        \\\"\\\"\\\"Get the currently logged in user\\\"\\\"\\\"\"
        user_data = os.environ.get(self.current_user_key)
        if user_data:
            data = json.loads(user_data)
            return data.get('user')
        return None
```

Output:
("entity"{tuple_delimiter}"requests"{tuple_delimiter}"module"{tuple_delimiter}"A Python library for making HTTP requests, providing a simple API for interacting with web services."){record_delimiter}
("entity"{tuple_delimiter}"json"{tuple_delimiter}"module"{tuple_delimiter}"A Python module for encoding and decoding JSON data, used for serializing and deserializing data structures."){record_delimiter}
("entity"{tuple_delimiter}"os"{tuple_delimiter}"module"{tuple_delimiter}"A Python module providing a way to interact with the operating system, including environment variables."){record_delimiter}
("entity"{tuple_delimiter}"AuthService"{tuple_delimiter}"class"{tuple_delimiter}"A service class responsible for handling authentication operations like login, logout, and retrieving the current user."){record_delimiter}
("entity"{tuple_delimiter}"__init__"{tuple_delimiter}"method"{tuple_delimiter}"Constructor method for the AuthService class that initializes the API URL, user key, and creates a session."){record_delimiter}
("entity"{tuple_delimiter}"api_url"{tuple_delimiter}"variable"{tuple_delimiter}"An instance variable in AuthService that stores the base URL for authentication API endpoints."){record_delimiter}
("entity"{tuple_delimiter}"current_user_key"{tuple_delimiter}"variable"{tuple_delimiter}"An instance variable in AuthService that defines the key used for storing user data in environment variables."){record_delimiter}
("entity"{tuple_delimiter}"session"{tuple_delimiter}"variable"{tuple_delimiter}"An instance variable in AuthService that holds a requests Session object for making HTTP requests."){record_delimiter}
("entity"{tuple_delimiter}"login"{tuple_delimiter}"method"{tuple_delimiter}"A method in AuthService that sends user credentials to the server and stores the returned user data."){record_delimiter}
("entity"{tuple_delimiter}"logout"{tuple_delimiter}"method"{tuple_delimiter}"A method in AuthService that removes the current user's data from environment variables."){record_delimiter}
("entity"{tuple_delimiter}"get_current_user"{tuple_delimiter}"method"{tuple_delimiter}"A method in AuthService that retrieves and parses the current user's data from environment variables."){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"__init__"{tuple_delimiter}"The AuthService class contains the __init__ method which initializes the service with necessary configuration."{tuple_delimiter}"class method, initialization"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"login"{tuple_delimiter}"The AuthService class contains the login method which authenticates users with the server."{tuple_delimiter}"class method, authentication"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"logout"{tuple_delimiter}"The AuthService class contains the logout method which removes user authentication data."{tuple_delimiter}"class method, authentication"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"get_current_user"{tuple_delimiter}"The AuthService class contains the get_current_user method which retrieves the authenticated user."{tuple_delimiter}"class method, user retrieval"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"login"{tuple_delimiter}"requests"{tuple_delimiter}"The login method uses the requests module to make HTTP POST requests to the authentication endpoint."{tuple_delimiter}"module usage, HTTP communication"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"login"{tuple_delimiter}"json"{tuple_delimiter}"The login method uses the json module to serialize and deserialize data between Python and JSON."{tuple_delimiter}"module usage, data serialization"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"login"{tuple_delimiter}"os"{tuple_delimiter}"The login method uses the os module to store user data in environment variables."{tuple_delimiter}"module usage, data storage"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"logout"{tuple_delimiter}"os"{tuple_delimiter}"The logout method uses the os module to remove user data from environment variables."{tuple_delimiter}"module usage, data removal"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"get_current_user"{tuple_delimiter}"os"{tuple_delimiter}"The get_current_user method uses the os module to retrieve user data from environment variables."{tuple_delimiter}"module usage, data retrieval"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"get_current_user"{tuple_delimiter}"json"{tuple_delimiter}"The get_current_user method uses the json module to parse user data from a JSON string."{tuple_delimiter}"module usage, data deserialization"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"authentication, Python service, HTTP requests, environment variables, user management, JSON serialization"){completion_delimiter}
""",
]

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Remember Steps---

1. Identify all code entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. Preserve the exact casing of the entity name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's purpose, functionality, and implementation details
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other (e.g., inheritance, composition, dependency, function call)
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details (e.g., "imports", "extends", "implements", "calls", "uses")
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire code. These should capture the overarching ideas, patterns, or architecture present in the code.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

----Output---

Add them below using the same format:
"""

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

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

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
