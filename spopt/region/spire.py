"""SPiRe: Spatial Partitioning for Predictive Regimes."""

# ruff: noqa: N803, N806

import warnings

import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy import sparse
from scipy.optimize import OptimizeWarning
from scipy.sparse import csgraph as cg
from sklearn.base import clone, is_classifier
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from ..BaseClass import BaseSpOptHeuristicSolver


def _reference_sample(X, n_reference, trim, rng):
    """Sample the shared reference predictors X* from the observed rows of ``X``.

    Rows with any predictor outside its ``[trim, 1 - trim]`` quantiles are
    dropped first. If trimming removes every row, all rows are used.
    """
    candidates = X
    if trim > 0:
        lower, upper = np.quantile(X, [trim, 1 - trim], axis=0)
        keep = np.all((lower <= X) & (upper >= X), axis=1)
        if keep.any():
            candidates = X[keep]
    size = min(n_reference, candidates.shape[0])
    idx = rng.choice(candidates.shape[0], size=size, replace=False)
    return candidates[idx]


def _local_predictions(A, X, y, estimator, order, reference):
    """Fit one model per area on its ``order``-hop neighborhood.

    Returns an ``(N, M)`` array with each local model's predictions on
    ``reference``; the local models themselves are discarded.
    """
    n = A.shape[0]
    step = sparse.csr_matrix((A + sparse.identity(n)) != 0, dtype=np.int32)
    hood = step.copy()
    for _ in range(order - 1):
        hood = sparse.csr_matrix(hood @ step)
        hood.data[:] = 1
    predictions = np.empty((n, reference.shape[0]))
    for i in range(n):
        idx = hood.indices[hood.indptr[i] : hood.indptr[i + 1]]
        model = clone(estimator).fit(X[idx], y[idx])
        predictions[i] = model.predict(reference)
    return predictions


def _edge_dissimilarities(A, P):
    """Mean squared difference between local predictions across each edge.

    Returns ``rows``, ``cols`` and ``d`` for every edge with ``row < col``.
    """
    upper = sparse.triu(A, k=1).tocoo()
    rows, cols = upper.row, upper.col
    d = np.mean((P[rows] - P[cols]) ** 2, axis=1)
    return rows, cols, d


def _fit_sse(estimator, X, y, idx):
    """Fit a clone of ``estimator`` on rows ``idx``; return in-sample SSE and model."""
    model = clone(estimator).fit(X[idx], y[idx])
    residual = y[idx] - model.predict(X[idx])
    return float(residual @ residual), model


def _tree_preorder(msf, root):
    """Depth-first preorder of the tree containing ``root``.

    ``msf`` is a symmetric CSR spanning forest without explicit zeros. In the
    returned order every subtree is a contiguous block starting at its root.
    ``parent`` has length N, with -1 for ``root`` and for nodes outside the tree.
    """
    parent = np.full(msf.shape[0], -1)
    order = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in msf.indices[msf.indptr[node] : msf.indptr[node + 1]]:
            if neighbor != parent[node]:
                parent[neighbor] = node
                stack.append(neighbor)
    return np.asarray(order), parent


def _best_split(msf, root, X, y, estimator, floor):
    """Best single-edge split of the tree containing ``root``.

    Only edges that leave both sides with at least ``floor`` areas are fitted.
    Returns ``None`` if there is no such edge, otherwise a dict with the combined
    ``sse``, the cut ``edge`` as ``(parent, child)`` and the two ``parts`` as
    ``(members, sse, model)`` tuples.
    """
    order, parent = _tree_preorder(msf, root)
    n_region = order.shape[0]
    position = np.empty(msf.shape[0], dtype=int)
    position[order] = np.arange(n_region)
    size = np.ones(msf.shape[0], dtype=int)
    for node in order[:0:-1]:
        size[parent[node]] += size[node]

    best = None
    for child in order[1:]:
        n_inside = size[child]
        if n_inside < floor or n_region - n_inside < floor:
            continue
        start = position[child]
        inside = order[start : start + n_inside]
        outside = np.concatenate([order[:start], order[start + n_inside :]])
        sse_in, model_in = _fit_sse(estimator, X, y, inside)
        sse_out, model_out = _fit_sse(estimator, X, y, outside)
        if best is None or sse_in + sse_out < best["sse"]:
            best = {
                "sse": sse_in + sse_out,
                "edge": (int(parent[child]), int(child)),
                "parts": (
                    (outside, sse_out, model_out),
                    (inside, sse_in, model_in),
                ),
            }
    return best


def _sequential_cuts(msf, X, y, estimator, floor, n_target, verbose=False):
    """Cut the spanning forest one edge at a time, minimizing the total SSE.

    Each region caches its best split; after a cut only the two new regions are
    searched again. ``msf`` is modified in place.

    Returns
    -------
    msf : scipy.sparse.csr_matrix
        The cut spanning forest.
    labels_path : numpy.ndarray
        ``(L, N)`` labels, starting with the initial components.
    objective_path : numpy.ndarray
        ``(L,)`` global MSE matching ``labels_path``.
    models : dict
        Fitted estimator per final region label.
    """
    n = y.shape[0]
    n_components, labels = cg.connected_components(msf, directed=False)
    regions = []
    for label in range(n_components):
        members = np.flatnonzero(labels == label)
        sse, model = _fit_sse(estimator, X, y, members)
        regions.append({"members": members, "sse": sse, "model": model})
    labels_path = [labels]
    objective_path = [sum(region["sse"] for region in regions) / n]

    with tqdm(
        total=max(n_target - n_components, 0), desc="cutting", disable=not verbose
    ) as progress:
        while len(regions) < n_target:
            for region in regions:
                if "split" not in region:
                    region["split"] = _best_split(
                        msf, region["members"][0], X, y, estimator, floor
                    )
            feasible = [
                (region["sse"] - region["split"]["sse"], i)
                for i, region in enumerate(regions)
                if region["split"] is not None
            ]
            if not feasible:
                warnings.warn(
                    f"No feasible cut remains after finding {len(regions)} "
                    "regions: every split would create a region smaller than "
                    f"`floor` ({floor}). Decrease `floor` to find the remaining "
                    f"{n_target - len(regions)} regions.",
                    OptimizeWarning,
                    stacklevel=3,
                )
                break
            _, best = max(feasible)
            split = regions.pop(best)["split"]
            u, v = split["edge"]
            msf[u, v] = 0
            msf[v, u] = 0
            msf.eliminate_zeros()
            for members, sse, model in split["parts"]:
                regions.append({"members": members, "sse": sse, "model": model})
            _, labels = cg.connected_components(msf, directed=False)
            labels_path.append(labels)
            objective_path.append(sum(region["sse"] for region in regions) / n)
            progress.update()

    models = {int(labels[region["members"][0]]): region["model"] for region in regions}
    return msf, np.vstack(labels_path), np.asarray(objective_path), models


class Spire(BaseSpOptHeuristicSolver):
    """SPiRe (Spatial Partitioning for Predictive Regimes) partitions a map into
    contiguous regions whose areas share a common predictive mechanism
    :cite:`assuncao2026spire`.

    A local model is fitted on the spatial neighborhood of every area, and
    neighboring areas are compared through the mean squared difference of their
    predictions over a shared reference sample of predictors. These predictive
    dissimilarities weight a minimum spanning tree, which is cut sequentially,
    SKATER-style :cite:`assunccao2006efficient`, so that each cut minimizes the
    global mean squared error of one regional model per region.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame
        A Geodataframe containing the predictors and the response. Rows are
        matched to ``w`` by position.
    w : libpysal.weights.W
        A PySAL weights object expressing the neighbor relationships between
        observations. Only its symmetrized neighbor pattern is used; ``w``
        itself is not modified.
    attrs_name : list
        Strings for predictor names (columns of ``gdf``). A single string is
        treated as one column.
    y_name : str
        Name of the continuous response column of ``gdf``.
    n_clusters : int (default 5)
        The number of regions to form. The full path of partitions from one
        region up to ``n_clusters`` is stored.
    floor : int, float (default 10)
        The minimum number of areas in each region. Must be at least 2,
        because flexible models fit very small regions almost perfectly.
    estimator : sklearn regressor (default None)
        Any scikit-learn compatible regressor, cloned for every local and
        regional model. Defaults to
        ``DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)`` seeded from
        ``random_state``. A user-supplied estimator keeps its own
        ``random_state``. Local models are fitted on small neighborhoods
        (at most 13 areas for a 2-hop rook neighborhood on a regular grid), so
        they should hold several times ``min_samples_leaf`` areas: for sparse
        contiguity, increase ``neighborhood_order`` or pass a less constrained
        estimator.
    neighborhood_order : int (default 2)
        Local models are fitted on the ``neighborhood_order``-hop neighborhood
        of each area, including the area itself.
    n_reference : int (default 250)
        Size of the shared reference sample of predictor vectors.
    reference_trim : float (default 0.01)
        Rows with any predictor outside its ``[reference_trim,
        1 - reference_trim]`` quantiles are excluded from the reference sample.
    islands : str (default 'increase')
        Description of what to do with islands. If ``'ignore'``, the algorithm
        will discover ``n_clusters`` regions, treating islands as their own
        regions. If ``'increase'``, the algorithm will discover ``n_clusters``
        regions, treating islands as separate from ``n_clusters``.
    random_state : int, numpy.random.Generator (default None)
        Seed for the reference sample and the default estimator.
    verbose : bool (default False)
        Show a progress bar while cutting the tree.

    Attributes
    ----------

    labels_ : numpy.array
        Region IDs for observations.
    objective_path_ : numpy.array
        Global in-sample mean squared error after each cut. Entry ``i``
        corresponds to ``n_components + i`` regions, where ``n_components`` is
        the number of connected components of ``w`` (``1`` for a connected
        ``w``). Shorter than requested if no feasible cut remains. It is not
        guaranteed to be non-increasing: every cut is made even when no split
        lowers the error, e.g. when small regions limit a tree's leaves.
    labels_path_ : numpy.array
        ``(L, N)`` nested partitions matching ``objective_path_``.
    regional_models_ : dict
        Fitted estimator for each region label in ``labels_``.
    reference_sample_ : numpy.array
        The ``(M, P)`` reference sample of predictors.
    edge_dissimilarity_ : scipy.sparse.csr_matrix
        Predictive dissimilarity for each pair of neighbors.
    minimum_spanning_forest_ : scipy.sparse.csr_matrix
        The spanning forest after the cuts.

    Examples
    --------

    >>> import geopandas
    >>> import libpysal
    >>> from sklearn.linear_model import LinearRegression
    >>> from spopt.region import Spire

    >>> pth = libpysal.examples.get_path("columbus.shp")
    >>> columbus = geopandas.read_file(pth)
    >>> w = libpysal.weights.Queen.from_dataframe(columbus, use_index=False)

    With few areas per region, a linear model is a better regional model than
    the default regression tree.

    >>> model = Spire(
    ...     columbus, w, ["INC", "HOVAL"], "CRIME",
    ...     n_clusters=4, floor=8, estimator=LinearRegression(), random_state=0,
    ... )
    >>> model.solve()

    Get the region IDs and the error path used to choose the number of regions.

    >>> model.labels_
    >>> model.objective_path_

    """

    def __init__(
        self,
        gdf,
        w,
        attrs_name,
        y_name,
        n_clusters=5,
        floor=10,
        estimator=None,
        neighborhood_order=2,
        n_reference=250,
        reference_trim=0.01,
        islands="increase",
        random_state=None,
        verbose=False,
    ):
        if isinstance(attrs_name, str):
            attrs_name = [attrs_name]
        self.gdf = gdf
        self.w = w
        self.attrs_name = list(attrs_name)
        self.y_name = y_name
        self.n_clusters = n_clusters
        self.floor = floor
        self.estimator = estimator
        self.neighborhood_order = neighborhood_order
        self.n_reference = n_reference
        self.reference_trim = reference_trim
        self.islands = islands
        self.random_state = random_state
        self.verbose = verbose
        self._validate()

    def _validate(self):
        columns = [*self.attrs_name, self.y_name]
        missing = [c for c in columns if c not in self.gdf.columns]
        if missing:
            raise ValueError(f"Columns not found in `gdf`: {missing}.")
        non_numeric = [c for c in columns if not is_numeric_dtype(self.gdf[c])]
        if non_numeric:
            raise ValueError(f"Columns must be numeric: {non_numeric}.")
        if self.gdf[columns].isna().to_numpy().any():
            raise ValueError("`attrs_name` and `y_name` columns must not contain NaN.")
        n = len(self.gdf)
        if self.w.n != n:
            raise ValueError(
                f"`w.n` ({self.w.n}) does not match the number of rows in `gdf` ({n})."
            )
        if self.n_clusters < 1:
            raise ValueError("`n_clusters` must be at least 1.")
        if self.floor < 2:
            raise ValueError("`floor` must be at least 2.")
        if self.floor * self.n_clusters > n:
            raise ValueError(
                f"`floor * n_clusters` ({self.floor * self.n_clusters}) exceeds "
                f"the number of areas ({n})."
            )
        if self.neighborhood_order < 1:
            raise ValueError("`neighborhood_order` must be at least 1.")
        if self.n_reference < 1:
            raise ValueError("`n_reference` must be at least 1.")
        if not 0 <= self.reference_trim < 0.5:
            raise ValueError("`reference_trim` must be in [0, 0.5).")
        if self.islands not in ("increase", "ignore"):
            raise ValueError("`islands` must be 'increase' or 'ignore'.")
        if self.estimator is not None:
            if not (
                hasattr(self.estimator, "fit") and hasattr(self.estimator, "predict")
            ):
                raise TypeError("`estimator` must implement `fit` and `predict`.")
            if is_classifier(self.estimator):
                raise ValueError(
                    "`estimator` must be a regressor; SPiRe currently supports "
                    "regression only."
                )

    def _target_regions(self, msf):
        """Number of regions to reach, accounting for islands as in ``Skater``."""
        n_components, labels = cg.connected_components(msf, directed=False)
        if n_components == 1:
            return self.n_clusters
        if self.islands == "increase":
            n_target = self.n_clusters + n_components
            detail = (
                f"Increasing `n_clusters` from {self.n_clusters} to {n_target} "
                "in order to account for islands."
            )
        else:
            n_target = self.n_clusters
            detail = (
                f"Counting the {n_components} components towards the "
                f"{self.n_clusters} regions."
            )
        warnings.warn(
            f"The graph is disconnected ({n_components} components)! {detail}",
            OptimizeWarning,
            stacklevel=3,
        )
        if (np.bincount(labels) < self.floor).any():
            raise ValueError(
                "Islands must be larger than the floor. Drop the small islands "
                "or decrease `floor`."
            )
        return n_target

    def solve(self):
        """Solve the SPiRe regionalization and set the fitted attributes."""
        rng = np.random.default_rng(self.random_state)
        X = self.gdf[self.attrs_name].to_numpy(dtype=float)
        y = self.gdf[self.y_name].to_numpy(dtype=float)
        n = y.shape[0]
        if self.estimator is None:
            estimator = DecisionTreeRegressor(
                max_depth=3,
                min_samples_leaf=5,
                random_state=int(rng.integers(2**31 - 1)),
            )
        else:
            estimator = self.estimator

        # only the (symmetrized) neighbor pattern of ``w`` is used, so the
        # caller's weights and transform are left untouched
        A = abs(self.w.sparse)
        A = sparse.csr_matrix((A + A.T) != 0, dtype=float)

        self.reference_sample_ = _reference_sample(
            X, self.n_reference, self.reference_trim, rng
        )
        P = _local_predictions(
            A, X, y, estimator, self.neighborhood_order, self.reference_sample_
        )
        rows, cols, d = _edge_dissimilarities(A, P)
        both = (np.r_[rows, cols], np.r_[cols, rows])
        self.edge_dissimilarity_ = sparse.csr_matrix((np.r_[d, d], both), shape=(n, n))
        # csgraph treats explicit zeros as missing edges, so shift every weight
        # by a tiny constant; a uniform shift does not change the minimum tree
        shift = np.finfo(float).eps * max(1.0, d.max(initial=0.0))
        graph = sparse.csr_matrix((np.r_[d, d] + shift, both), shape=(n, n))
        msf = cg.minimum_spanning_tree(graph)
        msf = sparse.csr_matrix(msf + msf.T)

        n_target = self._target_regions(msf)
        msf, labels_path, objective_path, models = _sequential_cuts(
            msf, X, y, estimator, self.floor, n_target, verbose=self.verbose
        )
        self.minimum_spanning_forest_ = msf
        self.labels_path_ = labels_path
        self.objective_path_ = objective_path
        self.labels_ = labels_path[-1]
        self.regional_models_ = models
