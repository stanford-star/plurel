import abc
import math

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from plurel.config import RandomFunctionActivation, SCMParams


def _standard_weight_init(W: torch.Tensor, scm_params: SCMParams) -> None:
    init_fn = scm_params.initialization_choices.sample_uniform()
    init_fn(W)
    # Bernoulli sparsity mask with variance compensation; density=1.0 => identity
    density = float(scm_params.mlp_weight_density_choices.sample_uniform())
    mask = torch.bernoulli(torch.full_like(W, density))
    W.data *= mask / max(density, 1e-6)


def _block_dropout_weight_init(W: torch.Tensor, scm_params: SCMParams) -> None:
    torch.nn.init.zeros_(W)
    init_std = float(scm_params.mlp_init_std_choices.sample_log_uniform())
    n_blocks = int(np.random.randint(1, math.ceil(math.sqrt(min(W.shape))) + 1))
    block_size = [dim // n_blocks for dim in W.shape]
    keep_prob = (n_blocks * block_size[0] * block_size[1]) / W.numel()
    scale = scm_params.scale_init_std_by_dropout_choices.sample_uniform()
    std = init_std / (keep_prob**0.5 if scale else 1)
    for b in range(n_blocks):
        block_slice = tuple(slice(bs * b, bs * (b + 1)) for bs in block_size)
        torch.nn.init.normal_(W[block_slice], std=std)


WEIGHT_INIT_REGISTRY: dict[str, callable] = {
    "standard": _standard_weight_init,
    "block_dropout": _block_dropout_weight_init,
}


def _build_tree_regressor(tree_model: str, max_depth: int, n_estimators: int):
    if tree_model == "decision_tree":
        return DecisionTreeRegressor(max_depth=max_depth, splitter="random")
    if tree_model == "extra_trees":
        return ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=1)
    if tree_model == "random_forest":
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=1)
    if tree_model == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            tree_method="hist",
            multi_strategy="multi_output_tree",
            n_jobs=1,
        )
    raise ValueError(f"Invalid tree model: {tree_model}")


class Mechanism(abc.ABC):
    """A frozen random function mapping node/edge inputs to outputs.

    Every mechanism is constructed with the same signature
    ``(scm_params, in_dim, hid_dim, out_dim)`` and called as ``m(x)`` where
    ``x`` has shape ``(..., in_dim)`` and the result has shape
    ``(..., out_dim)``. A mechanism's parameters are fixed for its lifetime, so
    it behaves as a deterministic function reused across propagation chunks.
    ``MLPMechanism`` is additionally invariant to how rows are batched;
    ``TreeMechanism`` fits on the first ``tree_fit_rows`` rows of its first call.
    Register concrete subclasses in ``MECHANISM_REGISTRY``.
    """

    @abc.abstractmethod
    def __init__(
        self,
        scm_params: SCMParams,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
    ): ...

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class MLPMechanism(Mechanism):
    def __init__(
        self,
        scm_params: SCMParams,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
    ):
        num_layers = int(scm_params.mlp_num_layers_choices.sample_uniform())
        assert num_layers >= 1
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.weights = [torch.empty(dims[i], dims[i + 1]) for i in range(num_layers)]
        for W in self.weights:
            init_strategy = scm_params.weight_init_choices.sample_uniform()
            WEIGHT_INIT_REGISTRY[init_strategy](W, scm_params)
        self.act_scales = []
        for _ in range(num_layers - 1):
            act_fn = scm_params.activation_choices.sample_uniform()
            if isinstance(act_fn, RandomFunctionActivation):
                # The instance in activation_choices is a marker; instantiate
                # a fresh RFA so this layer's activation has its own frozen
                # params, independent of any other layer that picked RFA.
                act_fn = RandomFunctionActivation()
            self.act_scales.append((act_fn, math.exp(np.random.uniform(-1.0, 1.0))))

    def __call__(self, x):
        for W, (act_fn, scale) in zip(self.weights[:-1], self.act_scales):
            x = torch.clamp(scale * act_fn(x @ W), -1e6, 1e6)
        return x @ self.weights[-1]


class TreeMechanism(Mechanism):
    """Tree-based mechanism ported from TabICL's TreeLayer.

    A tree regressor is fit against random Gaussian targets and its predictions
    become the output — a piecewise-constant map whose partition is calibrated
    to the input distribution. The fit happens once, on the first
    ``tree_fit_rows`` rows of the first call, and is cached.
    """

    def __init__(
        self,
        scm_params: SCMParams,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
    ):
        self.out_dim = out_dim
        tree_model = scm_params.tree_model_choices.sample_uniform()
        max_depth = 2 + int(np.random.exponential(1.0 / scm_params.tree_depth_lambda))
        n_estimators = 1 + int(np.random.exponential(1.0 / scm_params.tree_n_estimators_lambda))
        self.model = _build_tree_regressor(tree_model, max_depth, n_estimators)
        self.fit_rows = int(scm_params.tree_fit_rows)
        self._fitted = False

    def __call__(self, x):
        orig_shape = x.shape[:-1]
        X = x.reshape(-1, x.shape[-1]).nan_to_num(0.0).detach().cpu().numpy()
        if not self._fitted:
            n_fit = min(X.shape[0], self.fit_rows)
            y_fake = np.random.randn(n_fit, self.out_dim)
            self.model.fit(X[:n_fit], y_fake.ravel() if self.out_dim == 1 else y_fake)
            self._fitted = True
        y = np.asarray(self.model.predict(X), dtype=np.float32).reshape(-1, self.out_dim)
        return torch.from_numpy(y).reshape(*orig_shape, self.out_dim)


MECHANISM_REGISTRY: dict[str, type[Mechanism]] = {
    "mlp": MLPMechanism,
    "tree": TreeMechanism,
}


def make_mechanism(scm_params: SCMParams, in_dim: int, hid_dim: int, out_dim: int):
    name = scm_params.mechanism_type_choices.sample_uniform()
    return MECHANISM_REGISTRY[name](
        scm_params=scm_params, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim
    )


class CategoricalEncoder:
    def __init__(
        self,
        scm_params: SCMParams,
        num_embeddings: int,
        embedding_dim: int,
    ):
        self.E = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, _freeze=True
        )
        init_fn = scm_params.initialization_choices.sample_uniform()
        init_fn(self.E.weight)
        self.mechanism = make_mechanism(
            scm_params=scm_params,
            in_dim=embedding_dim,
            hid_dim=embedding_dim,
            out_dim=embedding_dim,
        )

    def __call__(self, x: torch.LongTensor):
        return self.mechanism(self.E(x))


class CategoricalDecoder:
    def __init__(
        self,
        scm_params: SCMParams,
        num_embeddings: int,
        embedding_dim: int,
    ):
        self.E = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, _freeze=True
        )
        init_fn = scm_params.initialization_choices.sample_uniform()
        init_fn(self.E.weight)
        self.mechanism = make_mechanism(
            scm_params=scm_params,
            in_dim=embedding_dim,
            hid_dim=embedding_dim,
            out_dim=embedding_dim,
        )

    def __call__(self, x: torch.Tensor):
        x = self.mechanism(x)
        if x.dim() == 1:
            sims = self.E.weight @ x
        else:
            sims = self.E.weight @ x.mT
        return torch.argmax(sims, dim=0)
