# Pretraining Relational Transformer on PluRel data

PluRel's synthetic databases are what the [PluRel paper](https://arxiv.org/abs/2602.04029) used to
pretrain [Relational Transformer](https://github.com/snap-stanford/relational-transformer)
([ICLR 2026](https://arxiv.org/abs/2510.06377)). All model-side code — the Rust-based rustler
context sampler, preprocessing, pretraining, evaluation, and inference (including on your own
database) — lives in the [relational-transformer](https://github.com/rishabh-ranjan/relational-transformer)
repo. This page shows how PluRel's output plugs into it.

## Preprocessing a generated database

PluRel writes each database in the `relbench` format (`manifest.yaml` + `db/*.parquet`), which the
relational-transformer preprocessor consumes as-is:

```bash
# generate databases with plurel
$ pixi run python scripts/synthetic_gen.py --seed_offset 0 --num_dbs 1000 --num_proc 16

# in the relational-transformer repo: preprocess one generated database
$ pixi run preprocess --dataset ~/.cache/relbench/plurel-0 --out-dir ~/scratch/pre
```

## Preprocessed data on the Hugging Face Hub

Preprocessed data is hosted on the Hub and downloaded automatically on demand — every `pre_dir`
argument in relational-transformer accepts a local path or a Hub repo spec:

- [stanford-star/plurel-preprocessed](https://huggingface.co/datasets/stanford-star/plurel-preprocessed) — all 2000 PluRel synthetic databases from [stanford-star/plurel](https://huggingface.co/datasets/stanford-star/plurel), preprocessed.
- [stanford-star/relbench-preprocessed](https://huggingface.co/datasets/stanford-star/relbench-preprocessed) — preprocessed relbench databases.

## Pretrained checkpoints

Synthetic pretrained checkpoints are on the Hub at
[stanford-star/rt-plurel](https://huggingface.co/stanford-star/rt-plurel/tree/main); see the
[relational-transformer docs](https://github.com/rishabh-ranjan/relational-transformer/tree/main/docs)
for training and inference with them.

## Paper-exact code

The code used for all paper experiments — including the vendored rustler sampler and `rt/` training
code matching the [stanford-star/rt-plurel](https://huggingface.co/stanford-star/rt-plurel)
checkpoints — is preserved at the [`v1.0.0`](https://github.com/stanford-star/plurel/tree/v1.0.0) tag.

If you use the architecture, training loop or sampler code, please also cite the Relational
Transformer paper:

```bibtex
@inproceedings{ranjan2026relationaltransformer,
    title={{Relational Transformer:} Toward Zero-Shot Foundation Models for Relational Data},
    author={Rishabh Ranjan and Valter Hudovernik and Mark Znidar and Charilaos Kanatsoulis and Roshan Upendra and Mahmoud Mohammadi and Joe Meyer and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```
