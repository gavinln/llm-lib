# LangChain example

https://github.com/hwchase17/langchain

Building applications with Large Language Models (LLMs)

## ChatGPT uses

Login at the following address

https://chat.openai.com/auth/login

### Introduction

https://platform.openai.com/docs/introduction/overview

* Content generation
* Summarization
* Classification, categorization, and sentiment analysis
* Data extraction
* Translation

### Quickstart

Example prompt: Write a tagline for an ice cream shop
Completion: We serve up smiles with every scoop

Clone the project

```
git clone https://github.com/openai/openai-quickstart-python.git
```

## Run openapi-quickstart-python

1. Create a virtual environment

```
python3 -m venv venv
```

2. Activate the virtual environment

```
source venv/bin/activate
```

3. Install Python libraries

```
pip install -r requirements.txt
```

4. Run the Flask application

```
flask run
```

## Run openai-cookbook

1. Clone the project

```
git clone https://github.com/openai/openai-cookbook
```

2. Change to the project root

## Run openai-examples

1. Setup API key

```
source ./do_not_checkin/openai_key.sh
```

2. Run openai-example in a poetry virtual environment

```
make open-ai-example
```

### Datasets

#### Fine food reviews

There are two files

```
./openai-cookbook/examples/data/fine_food_reviews_1k.csv
./openai-cookbook/examples/data/fine_food_reviews_with_embeddings_1k.csv
```

Obtain_dataset.ipynb - read the first file and create the second file with embeddings
Visualizing_embeddings_in_2D.ipynb - visualize embeddings using tSNE
Visualizing_embeddings_with_Atlas.ipynb - requires Nomic login https://atlas.nomic.ai/

Classification_using_embeddings.ipynb
Clustering.ipynb
Regression_using_embeddings.ipynb
Semantic_text_search_using_embeddings.ipynb
Visualizing_embeddings_in_W&B.ipynb
Zero-shot_classification_with_embeddings.ipynb


## Links

[Prompt engineering guide][1000]

[1000]: https://github.com/dair-ai/Prompt-Engineering-Guide

Chat [GPT login][1010]

[1010]: https://chat.openai.com/auth/login

### Getting started with LLMs using LangChain

https://www.pinecone.io/learn/langchain-intro/

### Videos

https://www.youtube.com/@DataIndependent/videos

https://www.youtube.com/@echohive/videos

### Other

[How to build a semantic search][1100]

[1100]: https://haystack.deepset.ai/blog/how-to-build-a-semantic-search-engine-in-python

[Question answering chatbot][1110]

[1110]: https://github.com/jerpint/buster

[Byte pair encoding][1120]

[1120]: https://huggingface.co/course/chapter6/5

#### Github

https://github.com/hwchase17/langchain - 43k stars

https://github.com/nomic-ai/gpt4all - 44k stars

https://github.com/microsoft/guidance - 8k stars
