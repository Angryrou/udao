# Contributing to `udao``

Welcome to our project! We appreciate your interest in contributing to `udao`.

## Types of Contributions

You can contribute to `udao` in many ways. Here are some examples:

- Signaling issues.
- Fixing typos and grammatical errors.
- Improving the documentation.
- Adding new features.
- Fixing bugs.

## Installing the Project for Development

You can install the project for development by running the following command:

```
pip install -e .[dev]
```

## Pre-commit Hooks

You can install pre-commit hooks by running the following command:

```
pre-commit install
```

Pre-commit hooks will then be run at each commit.
You can also run the pre-commit hooks manually by running the following command:

```
pre-commit run --all-files
```

## Code Style

- We use [black](https://pypi.org/project/black/) for formatting our code.
- We use [mypy](https://mypy.readthedocs.io/en/stable/) for type checking.

## Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) for documentation.
The documentation is hosted on github pages.

To build the documentation locally, run the following command:

```
cd docs
make html
```

## Running Tests

We use [pytest](https://docs.pytest.org/en/stable/) for testing.
To run the tests, run the following command:

```
pytest udao
```

## Submitting a Pull Request

- Ensure your code passes all CI checks (pre-commit hooks, tests and documentation build)
- Submit your PR with a detailed description.

## Questions or Need Help?

- Contact us at chenghao@cs.umass.edu

Thank you for contributing to udao!
