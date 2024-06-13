# Guidance samples

[Guidance][100] is a Python library to control large language models. It helps reduce latency and cost vs conventional prompting or fine-tuning.

[100]: https://github.com/guidance-ai/guidance

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
