# generative-latent-replay

![Python](https://badges.aleen42.com/src/python.svg) ![PyTorch](https://img.shields.io/badge/​-PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch) ![conda](https://img.shields.io/badge/%E2%80%8B-conda-%2344A833.svg?style=flat&logo=anaconda&logoColor=44A833) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Repo for generative latent replay - a replay based continual learning method which:

1. freezes the backbone of a network after initial training
2. builds a generative model of latent representations of datasets as transformed by this backbone
3. replays sampled pseudo--latent-examples for subsequent training to mitigate catastrophic forgetting

We compare generartive latent replay against raw latent replay and baseline transfer learning.

We also explore:

- different modelling methods (GMM, etc)
- freezing the network at differing depths
- different replay buffer sizes
- different replay samploing strategies

## Reproducing experiments

To run experiments, first create and activate a virtual environment:

```python
conda env create -f environment.yml
conda activate env-glr
```

Then [run](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) the appropriate notebooks detailing the experiments.

**Alternatively** you can run the notebook directly in Google Colab:
[![Benchmark baseline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iacobo/generative-latent-replay/blob/main/Latent_Replay.ipynb)

NOTE: Search "generative latent replay" (no quotes) to get relevant papers for discussion in paper, namely generative feature replay, latent autoencoder replay etc.
