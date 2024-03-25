# openai-quickstart-python

## Setup the project

1. Change to the project environment directory

```
cd openai-quickstart-python
```

2. Clone the project

```
git clone https://github.com/openai/openai-quickstart-python.git
```

3. Activate the environment

```
poetry shell
```

4. Change to the project directory

```
cd openai-quickstart-python
```

5. Install Python libraries

```
poetry add $(cat openai-quickstart-python/requirements.txt)
```

6. Setup the OPENAI_API_KEY in the `.env` file

7. Run the Flask application

```
make chat-basic
```

8. Run the command line application assistant-basic

```
make assistant-basic
```

9. Run the command line application assistant-functions that uses tools

```
make assistant-functions
```

10. Run the flask app assistant-flask that uses uploaded files

```
make assistant-flask
```
