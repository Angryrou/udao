# UDAO
the Unified Data Analytics Optimizer (UDAO) package enables the optimization of data analytics pipelines.

## Getting Started

### Install

Using pip:

```
pip install udao
```

## Install on GPU

The current GPU version relies on CUDA 11.8 and PyTorch 2.0.1. The following instructions are for installing the GPU version of UDAO.

### Requirements

Before installing, please make sure you have the following dependencies installed (using pip):

```
pip install cuda-python==11.8
pip install torch==2.0.1 -f https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.2 -f https://download.pytorch.org/whl/cu118
pip install torchaudio==2.0.2 -f https://download.pytorch.org/whl/cu118
pip install dglgo==0.0.2
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```
