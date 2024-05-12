# redis examples

Redis examples using Python including those for machine learning.

## Redis in title examples from openai

[Cookbook][100]

[100]: https://cookbook.openai.com/

## Examples

### Using Redis for embeddings search

https://cookbook.openai.com/examples/vector_databases/redis/using_redis_for_embeddings_search

1. Get documents and embeddings
2. Create a vector search index
3. Load text and embeddings
4. Run a vector search
5. Run a hybrid search

vector-hybrid-search1.py

### Redis as a context store with Chat Completions

https://cookbook.openai.com/examples/vector_databases/redis/redisqna/redisqna

1. Ask chat gpt a question outside its training window
2. Create vector search index
3. Create openai embeddings
4. Load text and embeddings
5. Run a vector search to get context for the question
6. Ask chat gpt a question with addional context

vector-search-for-context.py

### Running hybrid VSS queries with Redis and OpenAI

https://cookbook.openai.com/examples/vector_databases/redis/redis-hybrid-query-examples

1. Create a vector search index
2. Create openai embeddings
3. Load embeddings
4. Run a vector search
5. Run a hybrid search

vector-hybrid-search2.py

### Redis vectors as JSON with OpenAI

https://cookbook.openai.com/examples/vector_databases/redis/redisjson/redisjson

1. Create embeddings
2. Create a vector search index
3. Load text and embeddings into Redis
4. Run a vector similarity search
5. Run a hybrid search

vector-hybrid-search3.py

### Using Redis as a vector database with OpenAI

https://cookbook.openai.com/examples/vector_databases/redis/getting-started-with-redis-and-openai

1. Get text documents
2. Create embeddings
3. Create a vector index in Redis
4. Load text and embeddings into Redis
5. Run a vector similarity search
6. Run a hybrid search
6. Run a HNSW search

vector-hybrid-hnsw-search.py
