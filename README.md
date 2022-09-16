# Generative Latent Replay

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx) ![PyTorch](https://img.shields.io/badge/​-PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) ![conda](https://img.shields.io/badge/%E2%80%8B-conda-%2344A833.svg?style=flat&logo=anaconda&logoColor=44A833) ![Avalanche](https://img.shields.io/badge/%E2%80%8B-Avalanche-29B6F6.svg?style=flat&logo=data:image/png;base64,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) ![Python](https://badges.aleen42.com/src/python.svg)

![Generative Latent Replay diagram](img/../data/img/model.png)

## Method overview

Repo for Generative Latent Replay (GLR) - a continual learning method which aleviates catastophic forgetting through strict regularisation of low level data representation and synthetic latent replay. Explicitly GLR:

1. Freezes the backbone of a network after initial training
2. Builds generative models of the backbone-output latent representations of each dataset encountered by the model
3. Samples latent pseudo-examples from these generators for replay during subsequent training (to mitigate catastrophic forgetting)

## Features

Generative latent replay overcomes two issues encountered in traditional replay strategies:

1. High memory footprint:
   - replays can be sampled ad hoc
   - caches [compressed] latent representations
2. Privacy concerns
   - data is synthetic

Continual Learning Method | Replay based | Low memory | Privacy
--------------------------|--------------|------------|----------------
Naive                     | ❌           | ❌        | ✅
Replay                    | ✅           | ❌        | ❌
Latent Replay             | ✅           | ✅        | ❌
Generative Latent Replay  | ✅           | ✅        | ✅

## Experiments

### Description

We compare generative latent replay against the above methods on the following datasets:

- Permuted MNIST
- Rotated MNIST
- CoRE50

We also explore the effect of different:

- generative models (GMM, etc)
- network freeze depths
- replay buffer sizes
- replay sampling strategies

### Reproducing experiments

To run experiments, first create and activate a virtual environment:

```python
conda env create -f environment.yml
conda activate env-glr
```

Then [run](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) the appropriate [notebooks](experiments.ipynb) detailing the experiments.

**Alternatively** you can run the notebook directly in Google Colab:

<center>[![Benchmark baseline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iacobo/generative-latent-replay/blob/main/experiments.ipynb)</center>

## Porting method

Our implementation is fully compatible with the [Avalanche](https://github.com/ContinualAI/avalanche) continual learning library, and can be imported as a plugin in the same way as other Avalanche strategies:

```python
from avalanche.training.plugins import StrategyPlugin
from glr import GenerativeLatentReplay
```
