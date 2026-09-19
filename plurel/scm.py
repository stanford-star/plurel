"""
Generate synthetic tabular data using Structured Causal Models.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.distributions import Beta
from torch_frame import stype
from tqdm import tqdm

from plurel.bipartite import sample_bipartite_assignments
from plurel.config import DAGParams, SCMParams
from plurel.dag import DAG_REGISTRY
from plurel.transforms import CategoricalDecoder, CategoricalEncoder, make_mechanism
from plurel.ts import (
    BetaSourceGenerator,
    CategoricalTSDataGenerator,
    ExponentialSourceGenerator,
    GaussianSourceGenerator,
    IIDCategoricalGenerator,
    LogNormalSourceGenerator,
    MixedSourceGenerator,
    ParetoSourceGenerator,
    PoissonSourceGenerator,
    TSDataGenerator,
    UniformSourceGenerator,
    calendar_aware_timestamps,
)
from plurel.utils import TableType, set_random_seed


class SourceGenFactory:
    """Base class for source data generator factories."""

    def make_numerical(self, scm_params: SCMParams, num_rows: int, table_type: TableType):
        raise NotImplementedError

    def make_categorical(
        self,
        scm_params: SCMParams,
        num_categories: int,
        num_rows: int,
        table_type: TableType,
    ):
        raise NotImplementedError


class TSSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        min_value, max_value = sorted(np.random.uniform(size=2) * scale)
        trend_scale = (
            scm_params.activity_table_ts_trend_scale_choices.sample_uniform()
            if table_type == TableType.Activity
            else scm_params.entity_table_ts_trend_scale
        )
        noise_scale = (
            scm_params.activity_table_ts_noise_scale
            if table_type == TableType.Activity
            else scm_params.entity_table_ts_noise_scale
        )
        cycle_scale = (
            scm_params.activity_table_ts_cycle_scale_choices.sample_uniform()
            if table_type == TableType.Activity
            else scm_params.entity_table_ts_cycle_scale
        )
        ar_rho = (
            scm_params.ts_ar_rho_choices.sample_uniform()
            if table_type == TableType.Activity
            else scm_params.entity_table_ts_ar_rho
        )
        return TSDataGenerator(
            num_points=num_rows,
            min_value=min_value,
            max_value=max_value,
            trend_alpha=scm_params.ts_trend_alpha_choices.sample_uniform(),
            trend_scale=trend_scale,
            cycle_frequency=np.ceil(
                num_rows * scm_params.ts_cycle_freq_perc_choices.sample_uniform()
            ),
            cycle_scale=cycle_scale,
            noise_scale=noise_scale,
            ar_rho=ar_rho,
        )

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return CategoricalTSDataGenerator(
            [
                self.make_numerical(scm_params=scm_params, num_rows=num_rows, table_type=table_type)
                for _ in range(num_categories)
            ]
        )


class UniformSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        low, high = sorted(np.random.uniform(size=2) * scale)
        return UniformSourceGenerator(low=low, high=high)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class GaussianSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        low, high = sorted(np.random.uniform(size=2) * scale)
        return GaussianSourceGenerator(
            mean=(low + high) / 2,
            std=max((high - low) / 4, 1e-6),
            low=low,
            high=high,
        )

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class BetaSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        low, high = sorted(np.random.uniform(size=2) * scale)
        alpha = float(scm_params.source_beta_alpha_choices.sample_uniform())
        beta = float(scm_params.source_beta_beta_choices.sample_uniform())
        return BetaSourceGenerator(alpha=alpha, beta=beta, scale=high - low, offset=low)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class MixedSourceGenFactory(SourceGenFactory):
    _sub_types = ["uniform", "gaussian", "beta"]

    def make_numerical(self, scm_params, num_rows, table_type):
        sub_types = np.random.choice(self._sub_types, size=np.random.randint(2, 4))
        return MixedSourceGenerator(
            [
                SOURCE_GEN_REGISTRY[t].make_numerical(
                    scm_params=scm_params, num_rows=num_rows, table_type=table_type
                )
                for t in sub_types
            ]
        )

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class LogNormalSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        sigma = float(scm_params.lognormal_sigma_choices.sample_uniform())
        mean = float(np.random.uniform(-1.0, 1.0))
        return LogNormalSourceGenerator(mean=mean, sigma=sigma)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class ExponentialSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        return ExponentialSourceGenerator(scale=scale)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class ParetoSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        alpha = float(scm_params.pareto_alpha_choices.sample_uniform())
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        return ParetoSourceGenerator(alpha=alpha, scale=scale)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class PoissonSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        lam = float(scm_params.poisson_lambda_choices.sample_uniform())
        return PoissonSourceGenerator(lam=lam)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


SOURCE_GEN_REGISTRY: dict[str, SourceGenFactory] = {
    "ts": TSSourceGenFactory(),
    "uniform": UniformSourceGenFactory(),
    "gaussian": GaussianSourceGenFactory(),
    "beta": BetaSourceGenFactory(),
    "mixed": MixedSourceGenFactory(),
    "lognormal": LogNormalSourceGenFactory(),
    "exponential": ExponentialSourceGenFactory(),
    "pareto": ParetoSourceGenFactory(),
    "poisson": PoissonSourceGenFactory(),
}


class PropagationStrategy:
    """Base class for SCM propagation strategies.

    To add a new mode: subclass, implement all abstract methods, and register
    in PROPAGATION_STRATEGY_REGISTRY.
    """

    def __init__(self, scm_params: SCMParams):
        self.scm_params = scm_params
        self.mlp_emb_dim = scm_params.mlp_emb_dim

    def _get_numerical_encoder(self):
        return make_mechanism(
            scm_params=self.scm_params,
            in_dim=self.scm_params.mlp_in_dim,
            hid_dim=self.mlp_emb_dim,
            out_dim=self.mlp_emb_dim,
        )

    def _get_numerical_decoder(self):
        return make_mechanism(
            scm_params=self.scm_params,
            in_dim=self.mlp_emb_dim,
            hid_dim=self.mlp_emb_dim,
            out_dim=self.scm_params.mlp_out_dim,
        )

    def _get_categorical_encoder(self, num_categories: int):
        return CategoricalEncoder(
            scm_params=self.scm_params,
            num_embeddings=num_categories,
            embedding_dim=self.mlp_emb_dim,
        )

    def _get_categorical_decoder(self, num_categories: int):
        return CategoricalDecoder(
            scm_params=self.scm_params,
            num_embeddings=num_categories,
            embedding_dim=self.mlp_emb_dim,
        )

    def get_encoder(self, _stype: stype, num_categories: int | None = None):
        return {
            stype.numerical: self._get_numerical_encoder,
            stype.categorical: lambda: self._get_categorical_encoder(num_categories=num_categories),
        }[_stype]()

    def get_decoder(self, _stype: stype, num_categories: int | None = None):
        return {
            stype.numerical: self._get_numerical_decoder,
            stype.categorical: lambda: self._get_categorical_decoder(num_categories=num_categories),
        }[_stype]()

    def _get_source_data_gen(self, node_stype, num_categories, num_rows, table_type):
        source_gen_type = self.scm_params.source_gen_type_choices.sample_uniform()
        source_gen_factory = SOURCE_GEN_REGISTRY[source_gen_type]
        return {
            stype.numerical: lambda: source_gen_factory.make_numerical(
                scm_params=self.scm_params, num_rows=num_rows, table_type=table_type
            ),
            stype.categorical: lambda: source_gen_factory.make_categorical(
                scm_params=self.scm_params,
                num_categories=num_categories,
                num_rows=num_rows,
                table_type=table_type,
            ),
        }[node_stype]()

    def make_ts_data_gen(self, node_stype, num_categories, num_rows, table_type):
        raise NotImplementedError

    def make_node_encoder(self, _stype, num_categories):
        raise NotImplementedError

    def make_node_decoder(self, _stype, num_categories):
        raise NotImplementedError

    def make_edge_encoder(self, _stype, num_categories):
        raise NotImplementedError

    def tensorize_source(self, values: np.ndarray, _stype) -> torch.Tensor:
        """Tensorize a 1-D batch of source values for one node.

        Returns shape (N,) for categorical (long) and (N, 1) for numerical (float).
        """
        raise NotImplementedError

    def tensorize_col(self, values: np.ndarray, _stype) -> torch.Tensor:
        """Tensorize a 1-D batch of column values for cross-SCM collation.

        Returns shape (N,) for categorical (long) and (N, 1) for numerical (float).
        """
        raise NotImplementedError

    def post_generate(self, scm) -> None:
        """Called after all rows are generated. Override for post-processing."""
        pass


class EagerPropagationStrategy(PropagationStrategy):
    """Type-aware strategy: uses type-specific encoders/decoders throughout."""

    def make_ts_data_gen(self, node_stype, num_categories, num_rows, table_type):
        return self._get_source_data_gen(node_stype, num_categories, num_rows, table_type)

    def make_node_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=_stype, num_categories=num_categories)

    def make_node_decoder(self, _stype, num_categories):
        return self.get_decoder(_stype=_stype, num_categories=num_categories)

    def make_edge_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=_stype, num_categories=num_categories)

    def tensorize_source(self, values, _stype):
        if _stype == stype.categorical:
            return torch.as_tensor(values, dtype=torch.long)
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(-1)

    def tensorize_col(self, values, _stype):
        if _stype == stype.categorical:
            return torch.as_tensor(values, dtype=torch.long)
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(-1)


class LazyPropagationStrategy(PropagationStrategy):
    """Lazy strategy: treats all values as numerical; quantizes categoricals post-hoc."""

    def make_ts_data_gen(self, node_stype, num_categories, num_rows, table_type):
        source_gen_type = self.scm_params.source_gen_type_choices.sample_uniform()
        return SOURCE_GEN_REGISTRY[source_gen_type].make_numerical(
            scm_params=self.scm_params, num_rows=num_rows, table_type=table_type
        )

    def make_node_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=stype.numerical)

    def make_node_decoder(self, _stype, num_categories):
        return self.get_decoder(_stype=stype.numerical)

    def make_edge_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=stype.numerical)

    def tensorize_source(self, values, _stype):
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(-1)

    def tensorize_col(self, values, _stype):
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(-1)

    def post_generate(self, scm):
        scm._apply_categorical_quantization()


PROPAGATION_STRATEGY_REGISTRY: dict[str, type[PropagationStrategy]] = {
    "type_eager": EagerPropagationStrategy,
    "type_lazy": LazyPropagationStrategy,
}

AGGREGATION_REGISTRY: dict[str, callable] = {
    "sum": lambda s: s.sum(dim=0),
    "max": lambda s: s.max(dim=0).values,
    "product": lambda s: s.prod(dim=0),
    "logexp": lambda s: torch.logsumexp(s, dim=0),
}

CATEGORICAL_BOUNDARY_REGISTRY: dict[str, callable] = {
    "rank": lambda values, k: values[torch.randint(0, len(values), (k,))],
    "value": lambda values, k: torch.randn(k),
}


class NoiseGenerator:
    """Additive per-node noise, sampled at shape (num_rows, emb_dim)."""

    def sample(self, num_rows: int, emb_dim: int) -> torch.Tensor:
        raise NotImplementedError


class GaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, scm_params: SCMParams, std: float | torch.Tensor):
        self.std = std

    def sample(self, num_rows, emb_dim):
        return torch.randn(num_rows, emb_dim) * self.std


class BetaNoiseGenerator(NoiseGenerator):
    def __init__(self, scm_params: SCMParams, std: float | torch.Tensor):
        alpha = float(scm_params.node_noise_alpha_choices.sample_uniform())
        beta = float(scm_params.node_noise_beta_choices.sample_uniform())
        self.dist = Beta(torch.tensor([alpha]), torch.tensor([beta]))
        self.std = std

    def sample(self, num_rows, emb_dim):
        return self.dist.sample(sample_shape=(num_rows, emb_dim)).squeeze(-1) * self.std


NOISE_GENERATOR_REGISTRY: dict[str, type[NoiseGenerator]] = {
    "beta": BetaNoiseGenerator,
    "gaussian": GaussianNoiseGenerator,
}


def make_noise_generator(scm_params: SCMParams, emb_dim: int) -> NoiseGenerator:
    noise_type = scm_params.node_noise_type_choices.sample_uniform()
    base_std = float(scm_params.node_noise_std_choices.sample_log_uniform())
    per_dim = (
        torch.randn(1, emb_dim).abs()
        if scm_params.pre_sample_noise_std_choices.sample_uniform()
        else 1.0
    )
    return NOISE_GENERATOR_REGISTRY[noise_type](scm_params=scm_params, std=base_std * per_dim)


class SCM:
    def __init__(
        self,
        table_name: str,
        child_table_names: list[str],
        feature_columns: dict[str, stype],
        pkey_col: str,
        fkey_col_to_pkey_table: dict[str, str],
        foreign_scm_info: dict[str, SCM],
        scm_params: SCMParams,
        dag_params: DAGParams,
        seed: int | None = None,
    ):
        self.table_name = table_name
        self.child_table_names = child_table_names
        self.num_col_nodes = len(feature_columns)
        self.feature_columns = feature_columns
        self.pkey_col = pkey_col
        self.fkey_col_to_pkey_table = fkey_col_to_pkey_table
        self.foreign_scm_info = foreign_scm_info
        self.scm_params = scm_params
        self.dag_params = dag_params
        self.seed = seed
        if self.seed:
            set_random_seed(seed=self.seed)

        self.propagation_mode = self.scm_params.propagation_mode_choices.sample_uniform()
        self.strategy = PROPAGATION_STRATEGY_REGISTRY[self.propagation_mode](
            scm_params=self.scm_params
        )

        self.validate_foreign_scms()
        self.initialize_dag()
        self.initialize_nodes_and_edges()
        self._topological_generations = list(nx.topological_generations(self.dag.graph))
        self._collation_cache: dict[str, list[tuple]] = {}

    def initialize_dag(self):
        dag_class = self.scm_params.scm_layout_choices.sample_uniform()
        dag_class = DAG_REGISTRY[dag_class]
        scm_col_node_perc = self.scm_params.scm_col_node_perc_choices.sample_uniform()
        num_nodes = (
            self.num_col_nodes
            + int(np.ceil(self.num_col_nodes / scm_col_node_perc))
            + 3  # buffer to avoid empty graphs
        )
        self.dag = dag_class(num_nodes=num_nodes, dag_params=self.dag_params, seed=self.seed)
        self.num_edges = len(self.dag.graph.edges)

    def validate_foreign_scms(self):
        for foreign_table_name, scm in self.foreign_scm_info.items():
            assert hasattr(scm, "df"), (
                f"the foreign_scm for table: {foreign_table_name} does not contain the `self.df` attribute"
            )

    def initialize_ts_data_gens(self, num_rows: int, table_type: TableType):
        self.source_node_to_ts_data_gen = {}
        for node in self.source_nodes:
            _stype = self.dag.graph.nodes[node]["_stype"]
            num_categories = self.dag.graph.nodes[node]["num_categories"]
            self.source_node_to_ts_data_gen[node] = self.strategy.make_ts_data_gen(
                node_stype=_stype,
                num_categories=num_categories,
                num_rows=num_rows,
                table_type=table_type,
            )

    def initialize_nodes_and_edges(self):
        self.source_nodes = [
            node for node in self.dag.graph.nodes if self.dag.graph.in_degree(node) == 0
        ]
        self.col_nodes = np.random.choice(
            a=self.dag.graph.nodes, size=self.num_col_nodes, replace=False
        )
        for col_idx, (col_name, col_info) in enumerate(self.feature_columns.items()):
            col_node = self.col_nodes[col_idx]
            self.dag.graph.nodes[col_node]["col_name"] = col_name
            self.dag.graph.nodes[col_node]["_stype"] = col_info["_stype"]
            self.dag.graph.nodes[col_node]["num_categories"] = (
                len(col_info["categories"]) if col_info["categories"] else None
            )

        self._reset_node_attributes()
        self._reset_edge_attributes()

    def _reset_node_attributes(self):
        for node in self.dag.graph.nodes:
            if node in self.col_nodes:
                _stype = self.dag.graph.nodes[node]["_stype"]
                num_categories = self.dag.graph.nodes[node]["num_categories"]
            else:
                _stype = self.scm_params.col_stype_choices.sample_uniform()
                self.dag.graph.nodes[node]["_stype"] = _stype
                num_categories = (
                    self.scm_params.num_categories_choices.sample()
                    if _stype == stype.categorical
                    else None
                )
                self.dag.graph.nodes[node]["num_categories"] = num_categories

            self.dag.graph.nodes[node]["noise_generator"] = make_noise_generator(
                scm_params=self.scm_params, emb_dim=self.strategy.mlp_emb_dim
            )
            self.dag.graph.nodes[node]["propagation_agg"] = (
                self.scm_params.propagation_agg_choices.sample_uniform()
            )
            self.dag.graph.nodes[node]["decoder"] = self.strategy.make_node_decoder(
                _stype=_stype, num_categories=num_categories
            )
            if node in self.col_nodes:
                self.dag.graph.nodes[node]["collation_encoders"] = {
                    (
                        self.table_name,
                        child_table_name,
                    ): self.strategy.make_node_encoder(_stype=_stype, num_categories=num_categories)
                    for child_table_name in self.child_table_names
                }

    def _reset_edge_attributes(self):
        for parent_node, child_node in self.dag.graph.edges:
            parent_stype = self.dag.graph.nodes[parent_node]["_stype"]
            parent_num_categories = self.dag.graph.nodes[parent_node]["num_categories"]
            self.dag.graph.edges[parent_node, child_node]["encoder"] = (
                self.strategy.make_edge_encoder(
                    _stype=parent_stype, num_categories=parent_num_categories
                )
            )

    def _aggregate_embeddings(
        self, embs: list[torch.Tensor], weights: list[float], mode: str
    ) -> torch.Tensor:
        weighted = [w * e for w, e in zip(weights, embs)]
        stack = torch.clamp(torch.stack(weighted, dim=0), -1e4, 1e4)  # (n, emb_dim)
        if mode not in AGGREGATION_REGISTRY:
            raise ValueError(f"Unknown aggregation mode: {mode}")
        return AGGREGATION_REGISTRY[mode](stack)

    def propagate(self):
        """Run the SCM forward over all rows of this table in chunks.

        Pre-generates source values and per-node Beta noise once over all
        rows so the per-row outputs are independent of `propagate_batch_size`
        — chunking is a pure implementation detail. Then walks the DAG
        topologically per chunk so every per-node op is a single batched
        tensor call. Column-node outputs are accumulated per chunk and
        concatenated at the end.

        Memory note: pre-allocated noise is O(num_non_source_nodes × num_rows
        × emb_dim). At default emb_dim=32 that's ~3 KB per row.
        """
        # Pre-generate source values for all rows so any sequential state
        # (e.g. AR noise in TSDataGenerator) is built in row-index order.
        source_values_full: dict[int, np.ndarray] = {}
        for node, gen in self.source_node_to_ts_data_gen.items():
            source_values_full[node] = np.asarray(
                [gen.get_value(row_idx=i) for i in range(self.num_rows)]
            )

        # Pre-sample noise per (non-source node, row) so the noise vector at
        # row r for node n is fixed regardless of chunk boundaries.
        emb_dim = self.strategy.mlp_emb_dim
        node_noise: dict[int, torch.Tensor] = {}
        for node in self.dag.graph.nodes:
            if node in self.source_nodes:
                continue
            node_noise[node] = self.dag.graph.nodes[node]["noise_generator"].sample(
                self.num_rows, emb_dim
            )

        foreign_table_names = list(self.fkey_col_to_pkey_table.values())
        foreign_scms = [self.foreign_scm_info[fname] for fname in foreign_table_names]

        col_node_chunks: dict[int, list[torch.Tensor]] = {node: [] for node in self.col_nodes}

        batch_size = self.scm_params.propagate_batch_size
        for chunk_start in range(0, self.num_rows, batch_size):
            chunk_end = min(chunk_start + batch_size, self.num_rows)

            foreign_scms_embds: list[list[torch.Tensor]] = []
            for fname, foreign_scm in zip(foreign_table_names, foreign_scms):
                chunk_foreign_idxs = self.foreign_row_idxs_map[fname][chunk_start:chunk_end]
                foreign_scms_embds.append(
                    foreign_scm.collate_feature_embeddings(
                        row_idxs=chunk_foreign_idxs, child_table_name=self.table_name
                    )
                )

            for gen_layer in self._topological_generations:
                for node in gen_layer:
                    node_stype = self.dag.graph.nodes[node]["_stype"]
                    if node in self.source_nodes:
                        chunk_values = source_values_full[node][chunk_start:chunk_end]
                        value = self.strategy.tensorize_source(
                            values=chunk_values, _stype=node_stype
                        )
                    else:
                        parent_nodes = list(self.dag.graph.predecessors(node))
                        propagation_agg = self.dag.graph.nodes[node]["propagation_agg"]

                        # noise was pre-sampled per (node, row); slice per chunk
                        node_emb = node_noise[node][chunk_start:chunk_end]

                        all_embs, all_weights = [], []
                        for parent_node in parent_nodes:
                            parent_value = self.dag.graph.nodes[parent_node]["value"]
                            encoder = self.dag.graph.edges[parent_node, node]["encoder"]
                            all_embs.append(encoder(parent_value))
                            all_weights.append(
                                self.dag.graph.get_edge_data(parent_node, node)["weight"]
                            )
                        for foreign_row_embds in foreign_scms_embds:
                            w = 1 / len(foreign_row_embds) if propagation_agg == "sum" else 1.0
                            for foreign_row_embd in foreign_row_embds:
                                all_embs.append(foreign_row_embd)
                                all_weights.append(w)

                        if all_embs:
                            node_emb = node_emb + self._aggregate_embeddings(
                                embs=all_embs, weights=all_weights, mode=propagation_agg
                            )

                        decoder = self.dag.graph.nodes[node]["decoder"]
                        value = decoder(node_emb)

                    self.dag.graph.nodes[node]["value"] = value
                    if node in self.col_nodes:
                        col_node_chunks[node].append(value)

        for node in self.col_nodes:
            chunks = col_node_chunks[node]
            self.dag.graph.nodes[node]["value"] = (
                chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
            )

    def initialize_bi_fk_pk_graph_map(self):
        """Sample, for each foreign table, a parent-row index per child row
        from a hierarchical SBM joint distribution. Stored as a flat
        ``(num_rows,)`` int64 array per foreign table.
        """
        self.foreign_row_idxs_map: dict[str, np.ndarray] = {}
        for foreign_table_name, foreign_scm in tqdm(
            self.foreign_scm_info.items(),
            desc="Sampling FK→PK assignments",
            leave=False,
        ):
            num_levels = self.scm_params.bi_hsbm_levels_choices.sample_uniform()
            hierarchy_a = [
                self.scm_params.bi_hsbm_clusters_per_level_choices.sample_uniform()
                for _ in range(num_levels)
            ]
            hierarchy_b = [
                self.scm_params.bi_hsbm_clusters_per_level_choices.sample_uniform()
                for _ in range(num_levels)
            ]
            self.foreign_row_idxs_map[foreign_table_name] = sample_bipartite_assignments(
                size_a=len(foreign_scm.df),
                size_b=self.num_rows,
                hierarchy_a=hierarchy_a,
                hierarchy_b=hierarchy_b,
                parent_attractiveness_alpha=self.scm_params.parent_attractiveness_alpha,
                inactive_parent_frac=self.scm_params.inactive_parent_frac,
            )

    def _apply_categorical_quantization(self):
        for node in self.col_nodes:
            if self.dag.graph.nodes[node]["_stype"] != stype.categorical:
                continue
            col_name = self.dag.graph.nodes[node]["col_name"]
            num_categories = self.dag.graph.nodes[node]["num_categories"]
            col_values = torch.tensor(self.df[col_name].values, dtype=torch.float32)
            col_values = (col_values - col_values.mean()) / col_values.std().clamp_min(1e-8)
            mode = self.scm_params.cat_boundary_mode_choices.sample_uniform()
            boundaries = CATEGORICAL_BOUNDARY_REGISTRY[mode](col_values, num_categories - 1)
            classes = (col_values.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)
            # randomly permute class labels to break ordinality
            permute_prob = self.scm_params.cat_label_permute_prob_choices.sample_uniform()
            if np.random.random() < permute_prob:
                perm = torch.randperm(num_categories)
                classes = perm[classes]
            # randomly reverse class labels
            reverse_prob = self.scm_params.cat_label_reverse_prob_choices.sample_uniform()
            if np.random.random() < reverse_prob:
                classes = num_categories - 1 - classes
            self.df[col_name] = classes.numpy()

    def generate_df(
        self,
        num_rows: int,
        table_type: TableType,
        min_timestamp: pd.Timestamp | None = None,
        max_timestamp: pd.Timestamp | None = None,
    ):
        self.num_rows = num_rows
        self.initialize_ts_data_gens(num_rows=num_rows, table_type=table_type)
        self.initialize_bi_fk_pk_graph_map()
        self.propagate()

        df_dict: dict[str, np.ndarray] = {}
        if self.pkey_col is not None:
            df_dict[self.pkey_col] = np.arange(num_rows, dtype=np.int64)
        fk_null_prob = self.scm_params.fk_null_prob
        for fkey_col, foreign_table_name in self.fkey_col_to_pkey_table.items():
            fk_values = self.foreign_row_idxs_map[foreign_table_name]
            if fk_null_prob > 0.0:
                null_mask = np.random.rand(len(fk_values)) < fk_null_prob
                arr = pd.array(fk_values, dtype="Int64")
                arr[null_mask] = pd.NA
                df_dict[fkey_col] = arr
            else:
                df_dict[fkey_col] = fk_values
        for node in sorted(self.col_nodes):
            col_name = self.dag.graph.nodes[node]["col_name"]
            value = self.dag.graph.nodes[node]["value"]
            if value.dtype == torch.int64:
                df_dict[col_name] = value.numpy()
            else:
                # numerical decoder produces (N, 1); flatten + cast to float64
                # so downstream dtype checks (`_type in [float]`) match the
                # previous per-row .item() behavior.
                df_dict[col_name] = value.squeeze(-1).numpy().astype(np.float64)
        self.df = pd.DataFrame(df_dict)

        self.strategy.post_generate(scm=self)
        if min_timestamp and max_timestamp:
            if self.scm_params.calendar_aware_timestamps:
                self.df["date"] = calendar_aware_timestamps(
                    min_ts=min_timestamp, max_ts=max_timestamp, num_rows=num_rows
                )
            else:
                self.df["date"] = pd.date_range(
                    start=min_timestamp, end=max_timestamp, periods=num_rows
                )
        return self.df

    def source_node_flags(self) -> dict[str, bool]:
        """Per column, True iff its SCM node is a source node."""
        return {
            self.dag.graph.nodes[node]["col_name"]: bool(node in self.source_nodes)
            for node in self.col_nodes
        }

    def cross_table_input_flags(self) -> dict[str, bool]:
        """Per column, True iff its SCM node receives foreign-table embeddings."""
        has_fk_parents = bool(self.foreign_scm_info)
        return {
            self.dag.graph.nodes[node]["col_name"]: has_fk_parents
            and (node not in self.source_nodes)
            for node in self.col_nodes
        }

    def collate_feature_embeddings(self, row_idxs: np.ndarray, child_table_name: str):
        """Encode parent column values at the given row indices for cross-SCM
        propagation. Returns one (N, emb_dim) tensor per col-node, where N is
        the length of row_idxs.
        """
        if child_table_name not in self._collation_cache:
            # (col_name, _stype, encoder, col_values_array) — stable across calls
            # for this child_table_name; col_values_array caches the numpy view
            # of self.df[col_name] so per-chunk gathers don't re-pay pandas cost.
            self._collation_cache[child_table_name] = [
                (
                    self.dag.graph.nodes[node]["col_name"],
                    self.dag.graph.nodes[node]["_stype"],
                    self.dag.graph.nodes[node]["collation_encoders"][
                        (self.table_name, child_table_name)
                    ],
                    self.df[self.dag.graph.nodes[node]["col_name"]].to_numpy(),
                )
                for node in sorted(self.col_nodes)
            ]
        col_entries = self._collation_cache[child_table_name]
        embds: list[torch.Tensor] = []
        for _col_name, _stype, encoder, col_values_arr in col_entries:
            col_values = col_values_arr[row_idxs]
            value_tensor = self.strategy.tensorize_col(values=col_values, _stype=_stype)
            embds.append(encoder(value_tensor))
        return embds
