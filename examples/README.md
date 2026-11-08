# Examples

- [`generation/`](generation/): standalone v2 recipes that compose the generation API around a
  domain, each checking the relationships it declares. Each runs on its own from the installed
  package with embedded synthetic fixtures, no data or network:

  | Recipe | Shows |
  |---|---|
  | [`retail.py`](generation/retail.py) | retained customers and prices; counted orders; 1 to 8 lines per order (`Counts(least=1, most=8)`); no product twice in an order; prices copied exactly, line totals derived, order totals summed from recorded values; a target; a static save and load |
  | [`visits.py`](generation/visits.py) | retained regions and clinics; patients matched to a clinic of their region by a shared key; visit counts that follow patient age; visits after registration and completions after starts by positive delays; a temporal save and load |
  | [`enrollments.py`](generation/enrollments.py) | retained students and courses; one to six enrollments each; courses matched to a level; unique student, term and course; noisy grades in points from recorded GPA and difficulty |

  `python examples/generation/retail.py --mechanism-seed 1 --data-seed 2 --scale 3` picks the
  mechanism, the dataset of it and the size; `--out` keeps the saved database. The mechanisms are
  authored for illustration: they show how declarations compose, not how real shops, clinics or
  students behave.
- [`relational_transformer/`](relational_transformer/): pretrain
  [Relational Transformer](https://github.com/stanford-star/relational-transformer) on PluRel
  data: preprocessing, hosted preprocessed datasets, and pretrained checkpoints.
