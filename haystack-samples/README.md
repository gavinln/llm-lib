# Haystack AI

[Haystack][100] is an open-source Python framework that helps developers build large language model (LLM) powered custom applications.

[100]: https://github.com/deepset-ai/haystack

## Setup Poetry Python environment

Add poetry source as shown.

```
poetry source add torch-cpu https://download.pytorch.org/whl/cpu
poetry source add torch-gpu https://download.pytorch.org/whl/cu121
```

Add the cpu torch library or the gpu torch library (choose only one)

```
poetry add torch --source=torch-cpu
```

OR

```
poetry add torch --source=torch-gpu
```

## Tutorials

### L1 - level 1

#### Retrieval augmented generation question-answering pipeline

./rag-qa-pipeline.py

#### metadata filtering

https://haystack.deepset.ai/tutorials/31_metadata_filtering

#### preprocess files

https://haystack.deepset.ai/tutorials/30_file_type_preprocessing_index_pipeline

#### embed metadata

https://haystack.deepset.ai/tutorials/39_embedding_metadata_for_improved_retrieval

#### serialize pipelines

https://haystack.deepset.ai/tutorials/29_serializing_pipelines

#### extractive pipeline

https://haystack.deepset.ai/tutorials/34_extractive_qa_pipeline

### L2 - level 2

#### hybrid retrieval pipeline

https://haystack.deepset.ai/tutorials/33_hybrid_retrieval

#### structured output

https://haystack.deepset.ai/tutorials/28_structured_output_with_loop

#### classify documents

https://haystack.deepset.ai/tutorials/32_classifying_documents_and_queries_by_language

#### evaluate pipelines

https://haystack.deepset.ai/tutorials/35_evaluating_rag_pipelines

#### web search fallback

https://haystack.deepset.ai/tutorials/36_building_fallbacks_with_conditional_routing

Needs third-party key: SERPERDEV_API_KEY

#### pipeline input multiplexer

https://haystack.deepset.ai/tutorials/37_simplifying_pipeline_inputs_with_multiplexer

Needs third-party key: HF_API_TOKEN

### L3 - level 3

#### chat app with function calls

https://haystack.deepset.ai/tutorials/40_building_chat_application_with_function_calling
