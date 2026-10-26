# Merlin AI
## The AI engineering framework

This library is heavily inspired by [Marvin](https://pypi.org/project/marvin/) and gives all credits for the idea to it's authors.

It's a complete rewrite and does not depend on Pydantic, but uses Python's dataclasses and Enums instead.


## Python compatibility

Merlin supports standard CPython 3.9 through 3.14. Version 1.0.1 removes the
old tiktoken 0.8 constraint and requires tiktoken 0.12 or newer, which provides
Python 3.14 wheels. Python 3.14 also selects LiteLLM 1.106.4 or newer so its
Pydantic and tokenizer dependencies support that interpreter. Older Python
versions retain the existing LiteLLM version range.

## Development

Use Poetry 2.2.1 or newer with the checked-in lockfile:

```sh
poetry env use python3.14
poetry install
poetry run black .
poetry run pylint merlin_ai tests
poetry run python -m unittest discover -s tests -v
poetry build
```

The compatibility tests cover dataclass schemas, enums, datetime conversion,
tiktoken's native extension, and mocked completion dispatch. They require no
API keys and make no LLM calls. The tokenizer downloads its public vocabulary
on first use and caches it locally.
