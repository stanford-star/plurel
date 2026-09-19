import networkx as nx
import numpy as np


def assign_cluster_at_levels(num_nodes: int, hierarchy: list):
    num_base_clusters = np.prod(hierarchy)
    nodes_per_cluster = int(np.ceil(num_nodes / num_base_clusters))
    base_cluster_offsets = [
        nodes_per_cluster * (c_idx + 1) for c_idx in range(num_base_clusters - 1)
    ]
    base_cluster_offsets.append(num_nodes)
    cluster_at_levels = np.zeros((num_nodes, len(hierarchy)), dtype=int)
    cluster_node_idx_start = 0
    for c_idx in range(num_base_clusters):
        for l_idx in range(len(hierarchy)):
            fac = np.prod(hierarchy[l_idx + 1 :]) if l_idx + 1 < len(hierarchy) else 1
            cluster_node_idx_end = base_cluster_offsets[c_idx]
            cluster_at_levels[cluster_node_idx_start:cluster_node_idx_end, l_idx] = (
                c_idx // fac
            ) % hierarchy[l_idx]
        cluster_node_idx_start = cluster_node_idx_end
    return cluster_at_levels


def get_probs_at_levels(hierarchy_a: list, hierarchy_b: list):
    assert len(hierarchy_a) == len(hierarchy_b), "only similar hierarchy levels are supported"

    num_levels = len(hierarchy_a)
    probs_at_levels = []
    for l_idx in range(num_levels):
        shape = (hierarchy_a[l_idx], hierarchy_b[l_idx])
        probs = np.random.uniform(0.001, 0.002, size=shape)
        for i in range(max(shape)):
            probs[i % shape[0], i % shape[1]] = 0.9
        probs_at_levels.append(probs)
    return probs_at_levels


def get_nodes_connect_prob(
    node_idx_a: int,
    node_idx_b: int,
    probs_at_levels: list,
    cluster_at_levels_a: list,
    cluster_at_levels_b: list,
):
    num_levels = len(probs_at_levels)
    probs = [
        probs_at_levels[l_idx][
            cluster_at_levels_a[node_idx_a, l_idx],
            cluster_at_levels_b[node_idx_b, l_idx],
        ]
        for l_idx in range(num_levels)
    ]
    return np.prod(probs)


def sample_bipartite_assignments(
    size_a: int,
    size_b: int,
    hierarchy_a: list,
    hierarchy_b: list,
    chunk_memory_bytes: int = 100_000_000,
    parent_attractiveness_alpha: float | None = None,
    inactive_parent_frac: float = 0.0,
) -> np.ndarray:
    """For each of ``size_b`` child nodes, sample one parent index in
    ``[0, size_a)`` according to the hierarchical SBM joint probability:
    ``P(a, b) ∝ Π_l P_l[cluster_a[a, l], cluster_b[b, l]] · w_a``.

    ``w_a ~ Pareto(parent_attractiveness_alpha)`` per parent (1 when None), and
    a random ``inactive_parent_frac`` fraction of parents gets ``w_a = 0``.

    Vectorized across the child axis in chunks sized to fit the
    ``(size_a, chunk)`` log-prob matrix in ``chunk_memory_bytes``. Probabilities
    are accumulated in log space to avoid underflow when many small per-level
    probabilities are multiplied. Sampling is done via inverse-CDF over the
    per-column distribution — equivalent in distribution to the original
    ``np.random.choice`` per child, but vectorized.

    Returns
    -------
    parent_idx : (size_b,) int64 array
        ``parent_idx[b]`` is the sampled parent for child ``b``.
    """
    assert len(hierarchy_a) == len(hierarchy_b), "only similar hierarchy levels are supported"
    assert 0.0 <= inactive_parent_frac < 1.0, (
        f"inactive_parent_frac must be in [0, 1), got {inactive_parent_frac}"
    )

    cluster_at_levels_a = assign_cluster_at_levels(num_nodes=size_a, hierarchy=hierarchy_a)
    cluster_at_levels_b = assign_cluster_at_levels(num_nodes=size_b, hierarchy=hierarchy_b)
    probs_at_levels = get_probs_at_levels(hierarchy_a=hierarchy_a, hierarchy_b=hierarchy_b)
    log_p_at_levels = [np.log(p) for p in probs_at_levels]

    parent_log_w = None
    if parent_attractiveness_alpha is not None:
        w = np.random.pareto(parent_attractiveness_alpha, size=size_a) + 1.0
        parent_log_w = np.log(w)

    inactive_mask = None
    n_inactive = int(round(size_a * inactive_parent_frac))
    if n_inactive > 0:
        perm = np.random.permutation(size_a)
        inactive_mask = np.zeros(size_a, dtype=bool)
        inactive_mask[perm[:n_inactive]] = True

    bytes_per_cell = 8  # float64
    chunk = max(1, min(size_b, chunk_memory_bytes // max(1, size_a * bytes_per_cell)))

    parent_idx = np.empty(size_b, dtype=np.int64)
    for b_start in range(0, size_b, chunk):
        b_end = min(b_start + chunk, size_b)
        cw = b_end - b_start
        log_p = np.zeros((size_a, cw), dtype=np.float64)
        for l_idx, log_p_l in enumerate(log_p_at_levels):
            log_p += log_p_l[
                cluster_at_levels_a[:, l_idx][:, None],
                cluster_at_levels_b[b_start:b_end, l_idx][None, :],
            ]
        if parent_log_w is not None:
            log_p += parent_log_w[:, None]
        if inactive_mask is not None:
            log_p[inactive_mask, :] = -np.inf
        # log-softmax per column for numerical stability before exp
        log_p -= log_p.max(axis=0, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=0, keepdims=True)
        cdf = np.cumsum(p, axis=0)
        u = np.random.uniform(0.0, 1.0, size=(1, cw))
        parent_idx[b_start:b_end] = (cdf >= u).argmax(axis=0)

    return parent_idx


def get_bipartite_hsbm(size_a: int, size_b: int, hierarchy_a: list, hierarchy_b: list):
    """Build a NetworkX bipartite DiGraph by sampling parent assignments and
    materializing nodes/edges. Retained for backward compat (tests / external
    callers); inside SCM we use ``sample_bipartite_assignments`` directly.
    """
    parent_idx = sample_bipartite_assignments(
        size_a=size_a, size_b=size_b, hierarchy_a=hierarchy_a, hierarchy_b=hierarchy_b
    )
    cluster_at_levels_a = assign_cluster_at_levels(num_nodes=size_a, hierarchy=hierarchy_a)
    cluster_at_levels_b = assign_cluster_at_levels(num_nodes=size_b, hierarchy=hierarchy_b)

    bi_hsbm = nx.DiGraph()
    nodes_a = [f"a{i}" for i in range(size_a)]
    nodes_b = [f"b{j}" for j in range(size_b)]
    for a_idx, a_node in enumerate(nodes_a):
        bi_hsbm.add_node(
            a_node,
            node_idx=a_idx,
            hierarchy=list(cluster_at_levels_a[a_idx]),
        )
    for b_idx, b_node in enumerate(nodes_b):
        bi_hsbm.add_node(
            b_node,
            node_idx=b_idx,
            hierarchy=list(cluster_at_levels_b[b_idx]),
        )
    for b_idx, a_idx in enumerate(parent_idx):
        bi_hsbm.add_edge(nodes_a[int(a_idx)], nodes_b[b_idx])

    assert nx.is_bipartite(bi_hsbm)
    return bi_hsbm
