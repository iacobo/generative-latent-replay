# generative-latent-replay

![Python](https://badges.aleen42.com/src/python.svg) ![PyTorch](https://img.shields.io/badge/​-PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch) ![conda](https://img.shields.io/badge/%E2%80%8B-conda-%2344A833.svg?style=flat&logo=anaconda&logoColor=44A833) ![Avalanche](https://img.shields.io/badge/%E2%80%8B-avalanche-%2344A833.svg?style=flat&logo=avalanche&logoColor=skyblue)[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Repo for generative latent replay - a replay based continual learning method which:

1. freezes the backbone of a network after initial training
2. builds generative models of the latent representations of subsequent datasets encountered by the model (as output by the backbone)
3. samples latent pseudo-examples from these generators for replay during subsequent training (to mitigate catastrophic forgetting)

Generative latent replay overcomes two issues encountered in alternative replay strategies:

1. aleviates storage issues as replays can be sampled on the fly
2. aleviates privacy concerns of storing raw data indefinitely (data is synthetic)

We compare generative latent replay against raw replay, raw latent replay, and naive transfer learning.

We also explore:

- different modelling methods (GMM, etc)
- freezing the network at differing depths
- different replay buffer sizes
- different replay sampling strategies

## Reproducing experiments

To run experiments, first create and activate a virtual environment:

```python
conda env create -f environment.yml
conda activate env-glr
```

Then [run](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) the appropriate notebooks detailing the experiments.

**Alternatively** you can run the notebook directly in Google Colab:
[![Benchmark baseline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iacobo/generative-latent-replay/blob/main/experiments.ipynb)

NOTE: Search "generative latent replay" (no quotes) to get relevant papers for discussion in paper, namely generative feature replay, latent autoencoder replay etc.
