# openai-capabilities

## Text generation

https://platform.openai.com/docs/guides/text-generation

Examples:

```
# complete text with text output
make chat-completion-text           

# complete text with json output
make chat-completion-json           

# count tokens using the tiktoken library
make count-tokens-local             

# count tokens using an openai call
make count-tokens-remote            
```

## Function calling

https://platform.openai.com/docs/guides/function-calling

## Embeddings

https://platform.openai.com/docs/guides/embeddings

Examples:

```
# get and save food reviews embeddings
make save-food-reviews-embeddings   

# reduce embeddings by normalizing
make reduce-embeddings-dim          

# provide context article to answer question
make answer-question-using-context  

# search reviews using embeddings
make search-reviews                 

# get and save ag news embeddings
make save-ag-news-embeddings        

# recommend similar news articles
make recommendation-news            

# visualize distribution of reviews
make visualization-reviews          

# regression of reviews vs score
make regression-reviews             

# classification of reviews
make classification-reviews         

# classification of reviews; positive/negative
make zero-shot-classification       

# similarity between user/product embeddings
make user-product-embeddings        

# cluster using embeddings
make clustering-reviews             
```

## Fine-tuning

https://platform.openai.com/docs/guides/fine-tuning


