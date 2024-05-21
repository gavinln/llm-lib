# faiss samples

[FAISS][100] or Facebook AI Similarity Search is a library for efficient similarity search written in a combination of C++ and Python that works on the CPU and the GPU.

[100]: https://github.com/facebookresearch/faiss

## Wiki

https://github.com/facebookresearch/faiss/wiki/

## Installing faiss-cpu using pip

If you have the error "No module named 'packaging' follow instructions below

```
  File "/mnt/d/ws/llm-lib/faiss-samples/getting-started.py", line 10, in <module>
    import faiss
  File "/home/gavin/.cache/pypoetry/virtualenvs/faiss-samples-RHHLVhWO-py3.11/lib/python3.11/site-packages/faiss/__init__.py", line 16, in <module>
    from .loader import *
  File "/home/gavin/.cache/pypoetry/virtualenvs/faiss-samples-RHHLVhWO-py3.11/lib/python3.11/site-packages/faiss/loader.py", line 6, in <module>
    from packaging.version import Version
ModuleNotFoundError: No module named 'packaging'
```

Install the following package

```
poetry add packaging
```
