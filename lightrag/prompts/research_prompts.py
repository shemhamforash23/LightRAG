from typing import Any, Dict

# prompts for research papers analysis
PROMPTS: Dict[str, Any] = {}

# Entity types for research papers
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "concept",
    "method",
    "dataset",
    "model",
    "metric",
    "result",
    "author",
    "institution",
    "paper",
    "technology",
    "limitation",
    "future_work",
]

PROMPTS["entity_extraction"] = """---Goal---
Given a research paper or scientific text that is potentially relevant to this activity and a list of entity types, identify all research entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all research entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. Preserve the exact casing of the entity name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's purpose, methodology, significance, and context within the research
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other (e.g., builds upon, compares with, outperforms, uses, references)
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details (e.g., "comparison", "improvement", "application", "evaluation")
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity (1-10)
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire research text. These should capture the overarching ideas, methodologies, or findings present in the research.
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

# Examples for entity extraction
PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [concept, method, dataset, model, metric, result]
Text:
```
# Attention Is All You Need

## Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

## 1. Introduction
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have been made to improve recurrent neural networks by factorization of the parameter matrices and conditional computation. However, recurrent models still require sequential computation which prevents parallelization within training examples.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.

In this work, we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
```

Output:
("entity"{tuple_delimiter}"Transformer"{tuple_delimiter}"model"{tuple_delimiter}"A neural network architecture for sequence transduction that relies entirely on attention mechanisms without using recurrence or convolutions. It allows for greater parallelization and achieves state-of-the-art results in machine translation tasks."){record_delimiter}
("entity"{tuple_delimiter}"attention mechanism"{tuple_delimiter}"concept"{tuple_delimiter}"A component in neural networks that allows models to focus on different parts of the input sequence when generating output, enabling the modeling of dependencies regardless of their distance in sequences."){record_delimiter}
("entity"{tuple_delimiter}"recurrent neural networks"{tuple_delimiter}"model"{tuple_delimiter}"A class of neural networks designed for sequential data processing that maintain hidden states across time steps, including variants like LSTM and gated RNNs. They require sequential computation which limits parallelization."){record_delimiter}
("entity"{tuple_delimiter}"WMT 2014 English-to-German translation task"{tuple_delimiter}"dataset"{tuple_delimiter}"A benchmark dataset for machine translation that evaluates models on translating English text to German. The Transformer model achieved 28.4 BLEU score on this task."){record_delimiter}
("entity"{tuple_delimiter}"WMT 2014 English-to-French translation task"{tuple_delimiter}"dataset"{tuple_delimiter}"A benchmark dataset for machine translation that evaluates models on translating English text to French. The Transformer model achieved a state-of-the-art score of 41.8 BLEU on this task."){record_delimiter}
("entity"{tuple_delimiter}"BLEU"{tuple_delimiter}"metric"{tuple_delimiter}"A standard evaluation metric for machine translation that measures the similarity between model-generated translations and reference translations. Higher scores indicate better translation quality."){record_delimiter}
("entity"{tuple_delimiter}"parallelization"{tuple_delimiter}"concept"{tuple_delimiter}"The ability to perform multiple computations simultaneously, which is a key advantage of the Transformer architecture over recurrent models, resulting in faster training times."){record_delimiter}
("relationship"{tuple_delimiter}"Transformer"{tuple_delimiter}"attention mechanism"{tuple_delimiter}"The Transformer architecture is built entirely on attention mechanisms, eschewing recurrence and convolutions to achieve better parallelization and performance."{tuple_delimiter}"core component, architectural foundation"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Transformer"{tuple_delimiter}"recurrent neural networks"{tuple_delimiter}"The Transformer was designed as an alternative to recurrent neural networks, addressing their limitation of sequential computation by enabling greater parallelization."{tuple_delimiter}"improvement, alternative approach"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Transformer"{tuple_delimiter}"WMT 2014 English-to-German translation task"{tuple_delimiter}"The Transformer model achieved 28.4 BLEU on this task, improving over previous best results by over 2 BLEU points."{tuple_delimiter}"performance evaluation, benchmark achievement"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Transformer"{tuple_delimiter}"WMT 2014 English-to-French translation task"{tuple_delimiter}"The Transformer established a new single-model state-of-the-art score of 41.8 BLEU on this task after training for 3.5 days on eight GPUs."{tuple_delimiter}"performance evaluation, state-of-the-art result"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Transformer"{tuple_delimiter}"parallelization"{tuple_delimiter}"The Transformer architecture allows for significantly more parallelization than recurrent models, reducing training time substantially."{tuple_delimiter}"key advantage, efficiency improvement"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"attention mechanism"{tuple_delimiter}"recurrent neural networks"{tuple_delimiter}"Attention mechanisms have typically been used in conjunction with recurrent networks in most sequence modeling approaches prior to the Transformer."{tuple_delimiter}"historical combination, traditional usage"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"Transformer architecture, attention mechanisms, machine translation, parallelization, sequence transduction, neural networks, state-of-the-art results, BLEU score"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [concept, method, dataset, model, metric, result, limitation, future_work]
Text:
```
# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Abstract
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

## 1. Introduction
Language model pre-training has been shown to be effective for improving many natural language processing tasks. These include sentence-level tasks such as natural language inference and paraphrasing, which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained output at the token level.

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo, uses task-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT), introduces minimal task-specific parameters and is trained on the downstream tasks by simply fine-tuning all pretrained parameters. The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer. Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT addresses the previously mentioned unidirectional constraints by using a "masked language model" (MLM) pre-training objective, inspired by the Cloze task. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer. In addition to the masked language model, we also use a "next sentence prediction" task that jointly pretrains text-pair representations.
```

Output:
("entity"{tuple_delimiter}"BERT"{tuple_delimiter}"model"{tuple_delimiter}"Bidirectional Encoder Representations from Transformers, a language representation model designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. It can be fine-tuned for various NLP tasks with minimal task-specific modifications."){record_delimiter}
("entity"{tuple_delimiter}"bidirectional representations"{tuple_delimiter}"concept"{tuple_delimiter}"A language modeling approach that incorporates context from both left and right sides of a token, allowing for richer understanding of text compared to unidirectional models. BERT uses this approach in its pre-training phase."){record_delimiter}
("entity"{tuple_delimiter}"masked language model"{tuple_delimiter}"method"{tuple_delimiter}"A pre-training objective used by BERT that randomly masks some tokens from the input and trains the model to predict the original vocabulary ID of the masked word based only on its context. This enables bidirectional representation learning."){record_delimiter}
("entity"{tuple_delimiter}"next sentence prediction"{tuple_delimiter}"method"{tuple_delimiter}"A pre-training task used by BERT alongside the masked language model that trains the model to predict whether two sentences follow each other in the original text, helping to capture relationship between sentences."){record_delimiter}
("entity"{tuple_delimiter}"fine-tuning"{tuple_delimiter}"concept"{tuple_delimiter}"An approach to transfer learning where a pre-trained model is further trained on a downstream task with minimal additional parameters. BERT uses this approach to adapt to specific NLP tasks."){record_delimiter}
("entity"{tuple_delimiter}"GLUE"{tuple_delimiter}"dataset"{tuple_delimiter}"General Language Understanding Evaluation, a benchmark collection of NLP tasks used to evaluate language understanding models. BERT achieved a score of 80.5%, representing a 7.7% absolute improvement over previous methods."){record_delimiter}
("entity"{tuple_delimiter}"SQuAD v1.1"{tuple_delimiter}"dataset"{tuple_delimiter}"Stanford Question Answering Dataset version 1.1, a reading comprehension dataset where BERT achieved a Test F1 score of 93.2, representing a 1.5 point absolute improvement over previous methods."){record_delimiter}
("entity"{tuple_delimiter}"SQuAD v2.0"{tuple_delimiter}"dataset"{tuple_delimiter}"An enhanced version of the Stanford Question Answering Dataset that includes unanswerable questions. BERT achieved a Test F1 score of 83.1, representing a 5.1 point absolute improvement."){record_delimiter}
("entity"{tuple_delimiter}"MultiNLI"{tuple_delimiter}"dataset"{tuple_delimiter}"Multi-Genre Natural Language Inference, a dataset for natural language inference where BERT achieved an accuracy of 86.7%, representing a 4.6% absolute improvement over previous methods."){record_delimiter}
("entity"{tuple_delimiter}"unidirectional language models"{tuple_delimiter}"limitation"{tuple_delimiter}"Models that can only process text in one direction (typically left-to-right), which restricts their ability to understand context. BERT addresses this limitation through its bidirectional approach."){record_delimiter}
("relationship"{tuple_delimiter}"BERT"{tuple_delimiter}"bidirectional representations"{tuple_delimiter}"BERT is specifically designed to pre-train deep bidirectional representations, which is its key innovation compared to previous language models."{tuple_delimiter}"core innovation, defining characteristic"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"BERT"{tuple_delimiter}"masked language model"{tuple_delimiter}"BERT uses the masked language model pre-training objective to enable bidirectional representation learning, which is crucial to its architecture."{tuple_delimiter}"training methodology, enabling technique"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"BERT"{tuple_delimiter}"next sentence prediction"{tuple_delimiter}"BERT incorporates next sentence prediction as a secondary pre-training task to better understand relationships between sentences."{tuple_delimiter}"training methodology, supplementary technique"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"BERT"{tuple_delimiter}"fine-tuning"{tuple_delimiter}"BERT is designed to be fine-tuned on downstream tasks with minimal additional parameters, making it versatile for various NLP applications."{tuple_delimiter}"application approach, versatility"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"BERT"{tuple_delimiter}"GLUE"{tuple_delimiter}"BERT achieved a state-of-the-art score of 80.5% on the GLUE benchmark, demonstrating its superior language understanding capabilities."{tuple_delimiter}"performance evaluation, benchmark achievement"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"BERT"{tuple_delimiter}"SQuAD v1.1"{tuple_delimiter}"BERT achieved a Test F1 score of 93.2 on SQuAD v1.1, showing its effectiveness for question answering tasks."{tuple_delimiter}"performance evaluation, task-specific achievement"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"BERT"{tuple_delimiter}"unidirectional language models"{tuple_delimiter}"BERT was developed to address the limitations of unidirectional language models by incorporating context from both directions."{tuple_delimiter}"problem solution, improvement"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"masked language model"{tuple_delimiter}"bidirectional representations"{tuple_delimiter}"The masked language model objective enables BERT to learn bidirectional representations by predicting masked tokens using both left and right contexts."{tuple_delimiter}"enabling mechanism, methodological connection"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"BERT, bidirectional transformers, language representation, pre-training, fine-tuning, masked language model, next sentence prediction, natural language processing, state-of-the-art results"){completion_delimiter}
#############################""",
]

# Prompt for continuing entity extraction
PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Task---
Continue extraction of entities and relationships from the provided text.

Remember that for each entity, extract:
- entity_name: Name of the entity, use same language as input text. Preserve the exact casing of the entity name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's purpose, methodology, significance, and context within the research
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

For each relationship between entities, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation as to why source_entity and target_entity are related
- relationship_keywords: key words that summarize the relationship
- relationship_strength: a numeric score indicating strength of the relationship (1-10)
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire research. 
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

Return output in {language} as a single list of all the entities and relationships. Use **{record_delimiter}** as the list delimiter.

When finished, output {completion_delimiter}

---Input Text---
{input_text}

---Output---
"""
