# UDAO
This repository is the home of the UDAO library - a next-generation unified data analytics optimizer.

References:
- [Spark-based Cloud Data Analytics using Multi-Objective Optimization](https://ieeexplore.ieee.org/document/9458826/)
- [UDAO: a next-generation unified data analytics optimizer](https://dl.acm.org/doi/10.14778/3352063.3352103)

## Getting Started

### Install

Using pip:

```
pip install udao
```

### Install on GPU

The current GPU version relies on CUDA 11.8 and PyTorch 2.0.1. The following instructions are for installing the GPU version of UDAO.

#### Requirements

Before installing, please make sure you have the following dependencies installed (using pip):

```
  pip install cuda-python==11.8
  pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
  pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
  pip install torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
  pip install dglgo==0.0.2
  pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html
```

### Documentation
You can find the documentation on our (GitHub Pages)[https://angryrou.github.io/udao/]

## Limitations

Some known limitations:
1. Pandas DataFrame may have limitations when working with very large datasets.
2. Optimization algorithms require independent functions for each objective or constraint, impacting optimization speed, which may not match the speed achieved in our referenced papers (a fix is planned soon)
3. Categorical variables are always enumerated in MOGD.
4. Preprocessed data is not cached for reuse in hyper-parameter tuning

## Contributing

We welcome contributions!
You can go to [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
