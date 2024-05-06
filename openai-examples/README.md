# openai examples

List of example openai python programs

## Software used by examples

### Weights and Biases

A a tool for experiment tracking, visualization, and collaboration for machine
learning projects.

### Pinecone

Pinecone is a vector database designed specifically for building and scaling
machine learning applications that require efficient and accurate vector search
capabilities.

### Weaviate

Weaviate is an open-source vector search engine that enables scalable and efficient searching of high-dimensional data.

### Qdrant

Qdrant is an open-source vector search engine optimized for storing and searching high-dimensional vector data efficiently.

## Examples list as a csv file

The examples-list.csv is a file using the pipe '|' character as a separator with no header.

Create the file from the [examples][200] page.

[200]: https://cookbook.openai.com/

Run the ./process_openai_examples.py file to create a list of tags from the ./openai-examples-list.csv file

### Unit test writing

1. Use openai to write unit tests

```
make unit-test-writing
```

### Embeddings for long text

1. Use chunking to get embeddings for long texts

```
make embedding-long-texts
```

## Other libraries

### Spacy

[spaCy][900] is a high-performance, open-source Python library for Natural Language Processing (NLP). It offers pre-trained models in multiple languages and supports tasks like tokenization, tagging, entity recognition, and parsing.

[900]: https://github.com/explosion/spaCy

### TextBlob

[TextBlob][910] is a Python library for processing textual data. It supports common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

[910]: https://github.com/sloria/TextBlob
