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

PluRel is a framework for synthesizing diverse multi-tabular relational databases using Structural Causal Models (SCMs). It is the reference implementation for the [PluRel paper](https://arxiv.org/abs/2602.04029), with architecture and training code building on [Relational Transformer](https://github.com/snap-stanford/relational-transformer) ([ICLR 2026](https://arxiv.org/abs/2510.06377)).

This repository provides:

- Scalable generation of synthetic relational data (from scratch or SQL schemas), written directly in the `relbench` format: a self-describing dataset directory with a `manifest.yaml` (relational metadata) next to plain `db/<table>.parquet` files, loadable with [relbench](https://github.com/snap-stanford/relbench)'s `load_dataset`.

Preprocessing (the Rust-based rustler sampler), pretraining, evaluation, and inference live in the [relational-transformer](https://github.com/rishabh-ranjan/relational-transformer) repo, which consumes the `relbench` format, which PluRel outputs.

> [!NOTE]
> The paper-exact code — including the vendored rustler sampler and `rt/` training code used for all paper experiments (and matching the [stanford-star/rt-plurel](https://huggingface.co/stanford-star/rt-plurel) checkpoints) — is preserved at the [`v1.0.0`](https://github.com/stanford-star/plurel/tree/v1.0.0) tag.

## Framework Design

<img src="docs/static/images/plurel_animated.gif" alt="PluRel Logo"/>


## Installation

To use PluRel as a library:

```bash
pip install plurel
```

Requires Python 3.12+.

> [!NOTE]
> Development moves on `main` ahead of tagged releases. If you need features or fixes that have not yet been published to PyPI, install from source using the setup below.

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


## Synthesize Relational Data from Scratch

- The `SyntheticDataset` class can be used to create [relbench](https://github.com/snap-stanford/relbench) compatible dataset objects. With a `cache_dir` set, `get_db()` writes the dataset in `relbench` format (`manifest.yaml` + `db/*.parquet`), ready for `relbench.load.load_dataset` and [relational-transformer](https://github.com/rishabh-ranjan/relational-transformer) preprocessing/training.
- It only requires a `seed` and a `Config` object that contains `database`, `scm` and `dag` level params for sampling. See example below.

```py
from plurel import SyntheticDataset, Config

# create relbench compatible dataset
dataset = SyntheticDataset(seed=0, config=Config())

# create database which can be cached via relbench APIs
db = dataset.make_db()
```

### Configuration

The `Config` class controls all aspects of synthetic database generation through three parameter groups:

| Parameters | Description |
|-----------------|-------------|
| `DatabaseParams` | Table layout (`BarabasiAlbert`, `ReverseRandomTree`, `WattsStrogatz`, `Layered`), number of tables, row counts, column counts, timestamp ranges, and column post-processing (transforms, zero inflation, NaN rate). |
| `SCMParams` | SCM graph layouts, column types, MLP initialization, activation functions, noise distributions, and time-series trend/cycle parameters. |
| `DAGParams` | DAG-specific parameters like edge dropout, in-degree limits, and rewiring probabilities for different graph types. |

```py
from plurel import Config, DatabaseParams, SCMParams

config = Config(
    database_params=DatabaseParams(num_tables_choices=Choices(kind="range", value=[5, 10])),
    schema_file="path/to/schema.sql",  # optional: generate from SQL schema
    cache_dir="~/.cache/relbench",       # optional: cache generated databases
)
```

### Scalable Generation

We also provide a multiprocessing-based script to generate databases in parallel.

```bash
$ pixi run python scripts/synthetic_gen.py \
    --seed_offset 0 \
    --num_dbs 1000 \
    --num_proc 16
```

| Argument | Description |
|----------|-------------|
| `--seed_offset` | Seed offset for database generation. DBs will be named `plurel-<seed>` (override with `--db_prefix`). |
| `--num_dbs` | Number of databases to generate. |
| `--num_proc` | Number of parallel processes (default: number of CPU cores). |

> [!NOTE]
> See [`examples/generation/`](examples/generation/) for a notebook that synthesizes from a SQL schema.


## Preprocessing, Pretraining, and Inference

All model-side code — the Rust-based rustler context sampler, preprocessing, pretraining, evaluation, and inference (including on your own database) — lives in the [relational-transformer](https://github.com/rishabh-ranjan/relational-transformer) repo. PluRel's `relbench`-format output plugs in directly:

```bash
# in the relational-transformer repo: preprocess a generated database
pixi run preprocess --dataset ~/.cache/relbench/plurel-0 --out-dir ~/scratch/pre
```

Preprocessed data is hosted on the Hugging Face Hub and downloaded automatically on demand — every `pre_dir` argument there accepts a local path or a Hub repo spec:

- [stanford-star/plurel-preprocessed](https://huggingface.co/datasets/stanford-star/plurel-preprocessed) — all 2000 PluRel synthetic databases, preprocessed.
- [stanford-star/relbench-preprocessed](https://huggingface.co/datasets/stanford-star/relbench-preprocessed) — preprocessed relbench databases.

Synthetic pretrained checkpoints are on the Hub at [stanford-star/rt-plurel](https://huggingface.co/stanford-star/rt-plurel/tree/main); see the [relational-transformer docs](https://github.com/rishabh-ranjan/relational-transformer/tree/main/docs) for training and inference with them.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{kothapalli2026plurel,
title={PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models},
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
