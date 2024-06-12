# Guidance samples

[Guidance][100] is a Python library to control large language models. It helps reduce latency and cost vs conventional prompting or fine-tuning.

[100]: https://github.com/guidance-ai/guidance

## Updated notebook guidance samples

```
AzureOpenAI.py
OpenAI.py
TogetherAI.py
anachronism.py
chat.py
code_generation.py
gen.py
guaranteeing_valid_syntax.py
guidance_acceleration.py
intro_to_guidance.py
json_output_bench.py
prompt_boundaries_and_token_healing.py
proverb.py
rag.py
react.py
regex_constraints.py
server_anachronism.py
token_healing.py
use_clear_syntax.py
```

## Setup Poetry Python environment

Add poetry source as shown below to choose the CPU version of PyTorch

```
[[tool.poetry.source]]
name = "torch"
url = "https://download.pytorch.org/whl/cpu"
priority = "explicit"
```

Add the torch library

```
poetry add torch --source=torch
```
