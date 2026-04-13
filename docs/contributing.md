# Contributing

Thank you for your interest in contributing to CLAIMED!

## Code of Conduct

Please read and follow our [Code of Conduct](https://github.com/claimed-framework/claimed/blob/main/CODE_OF_CONDUCT.md).

## Development Setup

```bash
git clone https://github.com/claimed-framework/claimed.git
cd claimed
pip install -e ".[dev]"
pip install -r test_requirements.txt
```

## Running Tests

```bash
pytest tests/
```

## Adding a New Component

1. Create a directory under `src/claimed/components/<category>/`
2. Add an empty `__init__.py`
3. Create `<name>.py` with:
   - Module-level `os.environ.get(...)` parameter declarations
   - A `run(...)` function with type annotations and a docstring
   - A `main()` entry-point that calls `run()`
   - `if __name__ == "__main__": main()`
4. Add a documentation page under `docs/components/<category>/<name>.md`
5. Register the page in `mkdocs.yml` under `nav`

## Improving Documentation

The docs live in `docs/` and are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

Local preview:

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

Then open <http://127.0.0.1:8000>.

## Submitting a Pull Request

1. Fork the repository
2. Create a branch: `git checkout -b feat/my-feature`
3. Commit your changes
4. Push: `git push origin feat/my-feature`
5. Open a Pull Request against `main`

Please follow the [contribution process](https://github.com/claimed-framework/claimed/blob/main/contribution_process.md)
and the [release process](https://github.com/claimed-framework/claimed/blob/main/release_process.md) docs.
