# UDAO

the Unified Data Analytics Optimizer (UDAO) package enables the optimization of data analytics pipelines.

## Getting Started

### Install from PyPI

You can install the package from PyPI with:

```
pip install udao
```

### Install from source

This repository uses [Poetry](https://python-poetry.org/) for dependency management. If poetry is not installed, you can install it with:

```
pip install poetry
```

To install the dependencies, you can then run:

```
poetry install
```

Optional dependencies for the docs can be installed with:

```
poetry install --with docs
```

### Building and publishing the package

You can build the package using Poetry with:

```
poetry build
```

You can then publish the package to PyPI with:

```
poetry publish
```

### Install on GPU

The current GPU version relies on CUDA 11.8 and PyTorch 2.0.1. The following instructions are for installing the GPU version of UDAO.

#### Requirements

Before installing, please make sure you have the following dependencies installed (using pip):

```
pip install cuda-python==11.8
pip install torch==2.0.1 -f https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.2 -f https://download.pytorch.org/whl/cu118
pip install torchaudio==2.0.2 -f https://download.pytorch.org/whl/cu118
pip install dglgo==0.0.2
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

Alternatively, you can install the additional dependencies for the GPU version with Poetry (`)

```
poetry install --only gpu
```

#### Install

You can then follow the instructions above to install the package (either from PyPI or from source).
```

## Contributing
You can install the development dependencies using

```
poetry install --with dev
```

### Pre-commit hooks
First install the pre-commit hooks with:

```
pre-commit install
```

You can then run the pre-commit hooks with:

```
pre-commit run --all-files
```

Pre-commits will run automatically before every commit.
