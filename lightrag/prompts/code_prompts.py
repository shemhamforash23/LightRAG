from typing import Any, Dict

# промпты для анализа кода
PROMPTS: Dict[str, Any] = {}

# Типы сущностей кода
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

# Примеры для извлечения сущностей
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
("entity"{tuple_delimiter}"train_test_split"{tuple_delimiter}"function"{tuple_delimiter}"A function from sklearn.model_selection that splits datasets into random train and test subsets based on specified parameters."){record_delimiter}
("entity"{tuple_delimiter}"DataProcessor"{tuple_delimiter}"class"{tuple_delimiter}"A class that handles data loading and preprocessing operations, including normalization of data using NumPy functions."){record_delimiter}
("entity"{tuple_delimiter}"load_data"{tuple_delimiter}"method"{tuple_delimiter}"Method of DataProcessor that loads data from a file using NumPy's loadtxt function and stores it in the class instance."){record_delimiter}
("entity"{tuple_delimiter}"preprocess"{tuple_delimiter}"method"{tuple_delimiter}"Method of DataProcessor that normalizes data if specified, ensuring data is loaded first if it hasn't been already."){record_delimiter}
("entity"{tuple_delimiter}"split_dataset"{tuple_delimiter}"function"{tuple_delimiter}"A standalone function that splits a dataset into features and target, then into training and testing sets using sklearn's train_test_split."){record_delimiter}
("relationship"{tuple_delimiter}"DataProcessor"{tuple_delimiter}"numpy"{tuple_delimiter}"The DataProcessor class uses numpy for data loading and numerical operations."{tuple_delimiter}"library usage, numerical operations"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"load_data"{tuple_delimiter}"numpy"{tuple_delimiter}"The load_data method uses numpy's loadtxt function to read data from a file."{tuple_delimiter}"library usage, data loading"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"preprocess"{tuple_delimiter}"numpy"{tuple_delimiter}"The preprocess method uses numpy's mean and std functions for normalization."{tuple_delimiter}"library usage, statistical operations"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"preprocess"{tuple_delimiter}"load_data"{tuple_delimiter}"The preprocess method calls load_data if data is not already loaded."{tuple_delimiter}"method call, data loading"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"split_dataset"{tuple_delimiter}"train_test_split"{tuple_delimiter}"The split_dataset function uses the train_test_split function from sklearn to divide data into training and testing sets."{tuple_delimiter}"function call, data splitting"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"data processing, machine learning preprocessing, dataset splitting, normalization, numpy operations"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [class, function, module, variable, interface]
Text:
```python
# Authentication service for user management
from typing import Optional, Dict, Any
import requests
import json
import os

class AuthService:
    \"\"\"\"\"\Service for handling authentication operations\"\"\"\"\"
    
    def __init__(self, base_url: str):
        \"\"\"\"\"\Initialize the auth service with base URL\"\"\"\"\"
        self.base_url = base_url
        self.session = requests.Session()
        self.current_user_key = "AUTH_CURRENT_USER"
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        \"\"\"\"\"\Login a user and store their credentials\"\"\"\"\"
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        user_data = response.json()
        
        # Store user in environment variable
        os.environ[self.current_user_key] = json.dumps(user_data)
        
        return user_data
    
    def logout(self) -> bool:
        \"\"\"\"\"\Logout the current user\"\"\"\"\"
        # Clear user from environment
        if self.current_user_key in os.environ:
            del os.environ[self.current_user_key]
            
        response = self.session.post(f"{self.base_url}/auth/logout")
        return response.status_code == 200
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        \"\"\"\"\"\Get the current logged-in user\"\"\"\"\"
        user_data = os.environ.get(self.current_user_key)
        if user_data:
            data = json.loads(user_data)
            return data.get('user')
        return None
```

Output:
("entity"{tuple_delimiter}"requests"{tuple_delimiter}"module"{tuple_delimiter}"A Python library for making HTTP requests, providing a simple API for interacting with web services."){record_delimiter}
("entity"{tuple_delimiter}"json"{tuple_delimiter}"module"{tuple_delimiter}"A Python module that provides functions for working with JSON data, such as parsing and serializing."){record_delimiter}
("entity"{tuple_delimiter}"os"{tuple_delimiter}"module"{tuple_delimiter}"A Python module that provides a way of interacting with the operating system, including environment variables."){record_delimiter}
("entity"{tuple_delimiter}"AuthService"{tuple_delimiter}"class"{tuple_delimiter}"A service class for handling user authentication operations, including login, logout, and user retrieval."){record_delimiter}
("entity"{tuple_delimiter}"current_user_key"{tuple_delimiter}"variable"{tuple_delimiter}"An instance variable in AuthService that defines the key used for storing user data in environment variables."){record_delimiter}
("entity"{tuple_delimiter}"session"{tuple_delimiter}"variable"{tuple_delimiter}"An instance variable in AuthService that holds a requests Session object for making HTTP requests."){record_delimiter}
("entity"{tuple_delimiter}"login"{tuple_delimiter}"function"{tuple_delimiter}"A method in AuthService that authenticates a user by sending credentials to an API endpoint and stores the response."){record_delimiter}
("entity"{tuple_delimiter}"logout"{tuple_delimiter}"function"{tuple_delimiter}"A method in AuthService that logs out the current user by clearing environment variables and sending a request to an API endpoint."){record_delimiter}
("entity"{tuple_delimiter}"get_current_user"{tuple_delimiter}"function"{tuple_delimiter}"A method in AuthService that retrieves the currently logged-in user from environment variables, deserializing JSON data."){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"requests"{tuple_delimiter}"The AuthService class uses the requests module to make HTTP requests to authentication endpoints."{tuple_delimiter}"module usage, HTTP communication"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"login"{tuple_delimiter}"json"{tuple_delimiter}"The login method uses the json module to serialize user data for storage in environment variables."{tuple_delimiter}"module usage, data serialization"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"login"{tuple_delimiter}"os"{tuple_delimiter}"The login method uses the os module to store user data in environment variables."{tuple_delimiter}"module usage, data storage"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"logout"{tuple_delimiter}"os"{tuple_delimiter}"The logout method uses the os module to remove user data from environment variables."{tuple_delimiter}"module usage, data removal"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"get_current_user"{tuple_delimiter}"json"{tuple_delimiter}"The get_current_user method uses the json module to parse user data from a JSON string."{tuple_delimiter}"module usage, data deserialization"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"authentication, Python service, HTTP requests, environment variables, user management, JSON serialization"){completion_delimiter}
#############################""",
]

# Промпт для продолжения извлечения сущностей
PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Task---
Continue extraction of entities and relationships from the provided text.

Remember that for each entity, extract:
- entity_name: Name of the entity, use same language as input text. Preserve the exact casing of the entity name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's purpose, functionality, and implementation details
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

For each relationship between entities, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation as to why source_entity and target_entity are related
- relationship_keywords: key words that summarize the relationship
- relationship_strength: a numeric score indicating strength of the relationship
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire code. 
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

Return output in {language} as a single list of all the entities and relationships. Use **{record_delimiter}** as the list delimiter.

When finished, output {completion_delimiter}

---Input Text---
{input_text}

---Output---
"""
