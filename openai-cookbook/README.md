# OpenAI cookbook

This project include examples from the [OpenAI cookbook][100]

[100]: https://cookbook.openai.com/

## Setup the nix flakes environment

1. Create the lock file

```
nix flake lock
```

2. Start a shell with the environment

```
nix develop
```

## Setup the project using uv

1. Create the project

```
uv init --no-package --name openai-cookbook -p 3.13 openai-cookbook
```

2. Change to the project directory

```
cd openai-cookbook
```

3. Setup the virtual environment directory

```
export UV_PROJECT_ENVIRONMENT=~/.cache/venv/$(basename $(pwd))
```

4. Create a virtual env

```
uv venv
```

5. Run the project

```
uv run python hello.py
```

## Examples

### How to implement LLM [guardrails][110]

[110]: https://cookbook.openai.com/examples/how_to_use_guardrails

## Links

[Guardrails][1000] is a library to add guardrails to large language models.

[1000]: https://github.com/guardrails-ai/guardrails

[NeMo-Guardrails][1010] is an open-source toolkit for easily adding
programmable guardrails to LLM-based conversational systems.

[1010]: https://github.com/NVIDIA/NeMo-Guardrails/
