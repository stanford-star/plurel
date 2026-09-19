<div align="center">
  <h1>PluRel</h1>
  <p>
Synthetic Data unlocks Scaling Laws for Relational Foundation Models
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=github)](https://star-project.stanford.edu/plurel)
[![arXiv](https://img.shields.io/badge/arXiv-2602.04029-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2602.04029)
[![PyPI](https://img.shields.io/pypi/v/plurel.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/plurel/)

<img src="docs/static/images/scaling_law.png" alt="Scaling Law Plot"/>
</div>
<br>

## Latest Updates

- [07/2026] Released **v1.1.0** on [PyPI](https://pypi.org/project/plurel/) with the latest features and performance improvements.
- [04/2026] PluRel is accepted to **ICML 2026!**

## Overview

PluRel is an open-source library for synthesizing diverse relational and tabular data using Structural Causal Models (SCMs). It is the reference implementation for the [PluRel paper](https://arxiv.org/abs/2602.04029).

> [!NOTE]
> Pretraining models on PluRel data (preprocessing, checkpoints, inference with [Relational Transformer](https://github.com/rishabh-ranjan/relational-transformer)) is covered in [`examples/relational_transformer/`](examples/relational_transformer/).

## Framework Design

<img src="docs/static/images/plurel_animated.gif" alt="PluRel Logo"/>


## Installation

To use PluRel as a library (requires Python 3.12+):

```bash
pip install plurel
```

## Setup

For development and testing, set up the full environment with [pixi](https://pixi.sh/latest/installation/).

```bash
# setup pixi environment
$ pixi install

# Run tests
$ pixi run pytest

# Lint and format code
$ pixi run ruff check .
$ pixi run ruff format .

# Install pre-commit hooks
$ pixi run pre-commit install
```


## Status: v2 in progress

This branch rebuilds PluRel's core around a realized structural causal model API: one `SCM` per
table with explicit, inspectable mechanisms; a `Schema` that links tables through foreign keys
and lets child tables influence parent rows; a `Prior` that samples realized schemas; and
`sample()` for observational or interventional databases in the `relbench` format. The v1
generator and the paper-exact code remain at the [`v1.1.0`](https://github.com/stanford-star/plurel/tree/v1.1.0)
and [`v1.0.0`](https://github.com/stanford-star/plurel/tree/v1.0.0) tags.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{kothapalli2026plurel,
    title={{PluRel:} Synthetic Data unlocks Scaling Laws for Relational Foundation Models},
    author={Vignesh Kothapalli and Rishabh Ranjan and Valter Hudovernik and Vijay Prakash Dwivedi and Johannes Hoffart and Carlos Guestrin and Jure Leskovec},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026}
}
```

If you use the architecture, training loop or sampler code, please also cite the Relational Transformer paper:
```bibtex
@inproceedings{ranjan2026relationaltransformer,
    title={{Relational Transformer:} Toward Zero-Shot Foundation Models for Relational Data},
    author={Rishabh Ranjan and Valter Hudovernik and Mark Znidar and Charilaos Kanatsoulis and Roshan Upendra and Mahmoud Mohammadi and Joe Meyer and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```
