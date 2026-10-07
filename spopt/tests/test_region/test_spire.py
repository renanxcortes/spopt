# ruff: noqa: N806

import geopandas
import libpysal
import numpy
import pandas
import pytest
from scipy import sparse
from scipy.optimize import OptimizeWarning
from scipy.sparse import csgraph
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from spopt.region import Skater, Spire
from spopt.region.spire import (
    _best_split,
    _edge_dissimilarities,
    _local_predictions,
    _neighborhoods,
    _reference_sample,
    _sequential_cuts,
    _tree_preorder,
)

RANDOM_STATE = 12345


def chain_msf(n):
    """A path graph is its own spanning tree."""
    return libpysal.weights.lat2W(1, n).sparse.tocsr()


def star_msf(n_leaves):
    """Star centred on node 0: removing any edge isolates a single leaf."""
    leaves = list(range(1, n_leaves + 1))
    rows = [0] * n_leaves + leaves
    cols = leaves + [0] * n_leaves
    n = n_leaves + 1
    return sparse.csr_matrix((numpy.ones(2 * n_leaves), (rows, cols)), shape=(n, n))


def two_slopes(n, boundary):
    X = numpy.linspace(-1, 1, n).reshape(-1, 1)
    y = numpy.where(numpy.arange(n) < boundary, 2.0, -2.0) * X[:, 0]
    return X, y


# -- Mexican states
MEXICO = geopandas.read_file(libpysal.examples.get_path("mexicojoin.shp"))
# -- Columbus
COLUMBUS = geopandas.read_file(libpysal.examples.get_path("columbus.shp"))


def planted_grid(nrows=20, ncols=20, noise=0.1, seed=RANDOM_STATE):
    """Top/bottom halves with opposite slopes; ``x`` is identically distributed."""
    rng = numpy.random.default_rng(seed)
    n = nrows * ncols
    truth = (numpy.arange(n) // ncols >= nrows // 2).astype(int)
    x = rng.uniform(-1, 1, n)
    y = numpy.where(truth == 0, 2.0, -2.0) * x + rng.normal(0, noise, n)
    w = libpysal.weights.lat2W(nrows, ncols, rook=True)
    return pandas.DataFrame({"x": x, "y": y}), w, truth


def assert_valid_partition(model, w, n_regions, floor):
    labels = model.labels_
    assert numpy.unique(labels).size == n_regions
    adjacency = w.sparse.tocsr()
    for label in numpy.unique(labels):
        idx = numpy.flatnonzero(labels == label)
        assert idx.size >= floor
        n_parts, _ = csgraph.connected_components(
            adjacency[idx][:, idx], directed=False
        )
        assert n_parts == 1
    numpy.testing.assert_array_equal(model.labels_path_[-1], labels)
    assert model.objective_path_.shape == (model.labels_path_.shape[0],)
    for coarse, fine in zip(
        model.labels_path_[:-1], model.labels_path_[1:], strict=True
    ):
        for label in numpy.unique(fine):
            assert numpy.unique(coarse[fine == label]).size == 1
    assert set(model.regional_models_) == set(numpy.unique(labels).tolist())


class TestReferenceSample:
    def test_size_and_rows_come_from_x(self):
        X = numpy.random.default_rng(0).normal(size=(500, 3))
        ref = _reference_sample(X, 250, 0.01, numpy.random.default_rng(1))
        assert ref.shape == (250, 3)
        assert all((row == X).all(axis=1).any() for row in ref)

    def test_trim_excludes_extremes(self):
        X = numpy.arange(1000, dtype=float).reshape(-1, 1)
        ref = _reference_sample(X, 1000, 0.01, numpy.random.default_rng(0))
        assert ref.min() >= numpy.quantile(X, 0.01)
        assert ref.max() <= numpy.quantile(X, 0.99)
        # capped at the rows that survive trimming (980 of 1000)
        assert ref.shape == (980, 1)

    def test_no_trim_returns_all_rows_when_small(self):
        X = numpy.arange(10, dtype=float).reshape(-1, 1)
        ref = _reference_sample(X, 250, 0.0, numpy.random.default_rng(0))
        assert sorted(ref.ravel().tolist()) == list(range(10))

    def test_trim_removing_every_row_falls_back(self):
        # no row lies inside the quantile band of both columns
        X = numpy.array([[0.0, 1.0], [1.0, 0.0]])
        ref = _reference_sample(X, 250, 0.4, numpy.random.default_rng(0))
        assert ref.shape == (2, 2)


class FitFailsRegressor(RegressorMixin, BaseEstimator):
    """Raises if fitted, to prove a check ran before any local model was fit."""

    def fit(self, X, y):  # noqa: ARG002, N803
        raise RuntimeError("local model was fitted")

    def predict(self, X):  # noqa: ARG002, N803
        raise RuntimeError("local model was used")


def two_chains(n_first, n_second):
    """Adjacency of two disconnected chains (an island next to a mainland)."""
    first = libpysal.weights.lat2W(1, n_first).sparse
    second = libpysal.weights.lat2W(1, n_second).sparse
    return sparse.block_diag([first, second]).tocsr()


class TestNeighborhoods:
    def test_fixed_order_members(self):
        A = libpysal.weights.lat2W(1, 5).sparse
        members, orders = _neighborhoods(
            A, 2, min_size=1, adaptive=False, n_predictors=1
        )
        assert [sorted(m.tolist()) for m in members] == [
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4],
            [2, 3, 4],
        ]
        assert orders.tolist() == [2, 2, 2, 2, 2]

    def test_small_neighborhood_raises_and_suggests_adaptive(self):
        A = libpysal.weights.lat2W(1, 5).sparse
        with pytest.raises(ValueError, match="adaptive_neighborhoods=True") as error:
            _neighborhoods(A, 1, min_size=3, adaptive=False, n_predictors=1)
        message = str(error.value)
        assert "2 of 5 areas" in message
        assert "`min_neighborhood` (3)" in message
        assert "smallest has 2 areas" in message

    def test_adaptive_grows_only_small_neighborhoods(self):
        A = libpysal.weights.lat2W(1, 5).sparse
        members, orders = _neighborhoods(
            A, 1, min_size=3, adaptive=True, n_predictors=1
        )
        assert orders.tolist() == [2, 1, 1, 1, 2]
        assert sorted(members[0].tolist()) == [0, 1, 2]
        assert sorted(members[2].tolist()) == [1, 2, 3]
        assert sorted(members[4].tolist()) == [2, 3, 4]

    def test_adaptive_cannot_grow_past_a_small_island(self):
        A = two_chains(3, 10)
        with pytest.raises(ValueError, match="connected components with fewer"):
            _neighborhoods(A, 1, min_size=4, adaptive=True, n_predictors=1)


class TestLocalPredictions:
    def test_models_are_fit_on_given_members(self):
        # chain 0-1-2-3-4 with y = index; a mean model reveals the training rows
        A = libpysal.weights.lat2W(1, 5).sparse
        X = numpy.zeros((5, 1))
        y = numpy.arange(5, dtype=float)
        ref = numpy.zeros((3, 1))
        members1, _ = _neighborhoods(A, 1, min_size=1, adaptive=False, n_predictors=1)
        members2, _ = _neighborhoods(A, 2, min_size=1, adaptive=False, n_predictors=1)
        P1 = _local_predictions(members1, X, y, DummyRegressor(), ref)
        P2 = _local_predictions(members2, X, y, DummyRegressor(), ref)
        assert P1.shape == (5, 3)
        numpy.testing.assert_allclose(P1[:, 0], [0.5, 1.0, 2.0, 3.0, 3.5])
        numpy.testing.assert_allclose(P2[:, 0], [1.0, 1.5, 2.0, 2.5, 3.0])


class TestEdgeDissimilarities:
    def test_values_on_chain(self):
        A = libpysal.weights.lat2W(1, 3).sparse
        P = numpy.array([[0.0, 0.0], [1.0, 3.0], [1.0, 3.0]])
        rows, cols, d = _edge_dissimilarities(A, P)
        got = {
            (int(r), int(c)): float(v) for r, c, v in zip(rows, cols, d, strict=True)
        }
        assert got == {(0, 1): 5.0, (1, 2): 0.0}


class TestTreePreorder:
    def test_subtrees_are_contiguous_blocks(self):
        # tree edges: 0-1, 0-2, 1-3, 1-4
        rows = [0, 1, 0, 2, 1, 3, 1, 4]
        cols = [1, 0, 2, 0, 3, 1, 4, 1]
        msf = sparse.csr_matrix((numpy.ones(8), (rows, cols)), shape=(5, 5))
        order, parent = _tree_preorder(msf, 0)
        assert order[0] == 0
        assert sorted(order.tolist()) == [0, 1, 2, 3, 4]
        start = order.tolist().index(1)
        assert set(order[start : start + 3].tolist()) == {1, 3, 4}
        assert (parent[0], parent[1], parent[2], parent[3]) == (-1, 0, 0, 1)


class TestBestSplit:
    def test_finds_regime_boundary_on_chain(self):
        X, y = two_slopes(30, 15)
        split = _best_split(chain_msf(30), 0, X, y, LinearRegression(), floor=5)
        assert split["edge"] == (14, 15)
        assert split["sse"] == pytest.approx(0.0, abs=1e-9)
        sides = sorted(sorted(part[0].tolist()) for part in split["parts"])
        assert sides == [list(range(15)), list(range(15, 30))]

    def test_respects_floor(self):
        # the unconstrained best cut (1|2) would leave a side of size 2
        X, y = two_slopes(12, 2)
        split = _best_split(chain_msf(12), 0, X, y, LinearRegression(), floor=5)
        assert min(len(part[0]) for part in split["parts"]) >= 5

    def test_none_when_no_feasible_cut(self):
        X, y = two_slopes(8, 4)
        assert _best_split(chain_msf(8), 0, X, y, LinearRegression(), floor=5) is None


class TestSequentialCuts:
    def setup_method(self):
        n = 30
        self.truth = numpy.arange(n) // 10
        self.X = numpy.linspace(-1, 1, n).reshape(-1, 1)
        slope = numpy.array([2.0, -2.0, 5.0])[self.truth]
        intercept = numpy.array([0.0, 10.0, -10.0])[self.truth]
        self.y = intercept + slope * self.X[:, 0]

    def test_recovers_three_regimes(self):
        msf, labels_path, objective_path, models = _sequential_cuts(
            chain_msf(30), self.X, self.y, LinearRegression(), floor=5, n_target=3
        )
        assert labels_path.shape == (3, 30)
        assert objective_path.shape == (3,)
        assert adjusted_rand_score(self.truth, labels_path[-1]) == 1.0
        assert objective_path[-1] == pytest.approx(0.0, abs=1e-9)
        assert set(models) == {0, 1, 2}
        # two edges removed from the 29-edge chain (stored in both directions)
        assert msf.nnz == 2 * 27

    def test_first_objective_is_global_mse(self):
        _, _, objective_path, _ = _sequential_cuts(
            chain_msf(30), self.X, self.y, LinearRegression(), floor=5, n_target=2
        )
        fit = LinearRegression().fit(self.X, self.y)
        mse = numpy.mean((self.y - fit.predict(self.X)) ** 2)
        assert objective_path[0] == pytest.approx(mse)

    def test_path_is_nested_and_non_increasing(self):
        _, labels_path, objective_path, _ = _sequential_cuts(
            chain_msf(30), self.X, self.y, LinearRegression(), floor=5, n_target=4
        )
        assert numpy.all(numpy.diff(objective_path) <= 1e-9)
        for coarse, fine in zip(labels_path[:-1], labels_path[1:], strict=True):
            for label in numpy.unique(fine):
                assert numpy.unique(coarse[fine == label]).size == 1

    def test_warns_and_stops_when_no_feasible_cut(self):
        msf = star_msf(20)
        X = numpy.linspace(-1, 1, 21).reshape(-1, 1)
        y = 2 * X[:, 0]
        with pytest.warns(OptimizeWarning, match="No feasible cut remains"):
            _, labels_path, objective_path, models = _sequential_cuts(
                msf, X, y, LinearRegression(), floor=5, n_target=3
            )
        assert labels_path.shape == (1, 21)
        assert objective_path.shape == (1,)
        assert set(models) == {0}


class TestSpireValidation:
    def setup_method(self):
        self.df, self.w, _ = planted_grid(6, 6)
        self.params = {"attrs_name": ["x"], "y_name": "y", "n_clusters": 2, "floor": 5}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"y_name": "missing"}, "Columns not found"),
            ({"attrs_name": ["x", "nope"]}, "Columns not found"),
            ({"n_clusters": 0}, "n_clusters"),
            ({"floor": 1}, "floor"),
            ({"n_clusters": 4, "floor": 10}, r"floor \* n_clusters"),
            ({"neighborhood_order": 0}, "neighborhood_order"),
            ({"n_reference": 0}, "n_reference"),
            ({"reference_trim": 0.5}, "reference_trim"),
            ({"reference_trim": -0.1}, "reference_trim"),
            ({"islands": "drop"}, "islands"),
            ({"estimator": DecisionTreeClassifier()}, "regression"),
        ],
    )
    def test_value_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            Spire(self.df, self.w, **{**self.params, **kwargs})

    def test_non_numeric_column(self):
        df = self.df.assign(label="a")
        with pytest.raises(ValueError, match="numeric"):
            Spire(df, self.w, **{**self.params, "attrs_name": ["x", "label"]})

    def test_nan(self):
        df = self.df.copy()
        df.loc[0, "y"] = numpy.nan
        with pytest.raises(ValueError, match="NaN"):
            Spire(df, self.w, **self.params)

    def test_w_size_mismatch(self):
        with pytest.raises(ValueError, match="w.n"):
            Spire(self.df, libpysal.weights.lat2W(5, 5), **self.params)

    def test_estimator_without_predict(self):
        with pytest.raises(TypeError, match="fit.*predict"):
            Spire(self.df, self.w, **self.params, estimator=object())

    def test_single_string_attrs_name(self):
        model = Spire(self.df, self.w, **{**self.params, "attrs_name": "x"})
        assert model.attrs_name == ["x"]


class TestSpire:
    def test_recovers_planted_regimes_where_skater_fails(self):
        df, w, truth = planted_grid()
        # 2-hop rook neighborhoods hold at most 13 cells, too few for the default
        # tree's ``min_samples_leaf=5`` to resolve a slope; allow smaller leaves
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=0)
        model = Spire(df, w, ["x"], "y", n_clusters=2, estimator=tree)
        model.solve()
        assert adjusted_rand_score(truth, model.labels_) > 0.9
        assert_valid_partition(model, w, n_regions=2, floor=10)

        skater = Skater(df, libpysal.weights.lat2W(20, 20), ["x"], n_clusters=2)
        skater.solve()
        assert adjusted_rand_score(truth, skater.labels_) < 0.2

    def test_linear_estimator_recovers_planted_regimes(self):
        df, w, truth = planted_grid()
        model = Spire(df, w, ["x"], "y", n_clusters=2, estimator=LinearRegression())
        model.solve()
        assert adjusted_rand_score(truth, model.labels_) > 0.95

    def test_first_objective_is_global_mse(self):
        df, w, _ = planted_grid()
        model = Spire(df, w, ["x"], "y", n_clusters=3, estimator=LinearRegression())
        model.solve()
        X, y = df[["x"]].to_numpy(), df["y"].to_numpy()
        mse = numpy.mean((y - LinearRegression().fit(X, y).predict(X)) ** 2)
        assert model.objective_path_[0] == pytest.approx(mse)
        assert model.labels_path_.shape == (3, 400)

    def test_deterministic_with_random_state(self):
        df, w, _ = planted_grid()
        runs = []
        for _ in range(2):
            model = Spire(df, w, ["x"], "y", n_clusters=3, random_state=7)
            model.solve()
            runs.append(model)
        numpy.testing.assert_array_equal(runs[0].labels_, runs[1].labels_)
        numpy.testing.assert_array_equal(
            runs[0].objective_path_, runs[1].objective_path_
        )
        numpy.testing.assert_array_equal(
            runs[0].reference_sample_, runs[1].reference_sample_
        )

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    @pytest.mark.filterwarnings("ignore:The graph is disconnected")
    def test_columbus_random_forest(self):
        w = libpysal.weights.Queen.from_dataframe(COLUMBUS, use_index=False)
        n_components, _ = csgraph.connected_components(w.sparse)
        estimator = RandomForestRegressor(n_estimators=10, random_state=0)
        model = Spire(
            COLUMBUS,
            w,
            ["INC", "HOVAL"],
            "CRIME",
            n_clusters=4,
            floor=5,
            estimator=estimator,
            islands="ignore",
            adaptive_neighborhoods=True,
        )
        model.solve()
        assert_valid_partition(model, w, max(4, n_components), floor=5)

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    @pytest.mark.filterwarnings("ignore:The graph is disconnected")
    def test_mexico_invariants(self):
        w = libpysal.weights.Queen.from_dataframe(MEXICO, use_index=False)
        n_components, _ = csgraph.connected_components(w.sparse)
        attrs = [f"PCGDP{year}" for year in range(1950, 2000, 10)]
        model = Spire(
            MEXICO,
            w,
            attrs,
            "PCGDP2000",
            n_clusters=4,
            floor=5,
            islands="ignore",
            adaptive_neighborhoods=True,
            random_state=RANDOM_STATE,
        )
        model.solve()
        assert_valid_partition(model, w, max(4, n_components), floor=5)

    def test_zero_dissimilarities_keep_tree_spanning(self):
        df, w, _ = planted_grid(6, 6)
        df["y"] = 1.0
        model = Spire(df, w, ["x"], "y", n_clusters=3, floor=5)
        model.solve()
        assert model.edge_dissimilarity_.nnz == w.sparse.nnz
        assert not model.edge_dissimilarity_.data.any()
        assert numpy.unique(model.labels_path_[0]).size == 1
        assert_valid_partition(model, w, n_regions=3, floor=5)

    def test_early_stop_on_star(self):
        n_leaves = 20
        neighbors = {0: list(range(1, n_leaves + 1))}
        neighbors.update({i: [0] for i in range(1, n_leaves + 1)})
        w = libpysal.weights.W(neighbors)
        x = numpy.linspace(-1, 1, n_leaves + 1)
        df = pandas.DataFrame({"x": x, "y": 2 * x})
        model = Spire(df, w, ["x"], "y", n_clusters=3, floor=5)
        with pytest.warns(OptimizeWarning, match="No feasible cut remains"):
            model.solve()
        assert model.labels_path_.shape == (1, n_leaves + 1)
        assert numpy.unique(model.labels_).size == 1

    def test_asymmetric_w_is_symmetrized(self):
        n = 30
        X, y = two_slopes(n, 15)
        df = pandas.DataFrame({"x": X[:, 0], "y": y})
        forward = libpysal.weights.W(
            {i: [i + 1] if i < n - 1 else [] for i in range(n)}, silence_warnings=True
        )
        symmetric = libpysal.weights.lat2W(1, n)
        labels = []
        for w in (forward, symmetric):
            model = Spire(
                df,
                w,
                ["x"],
                "y",
                n_clusters=2,
                floor=5,
                estimator=LinearRegression(),
                min_neighborhood=3,
            )
            model.solve()
            labels.append(model.labels_)
        numpy.testing.assert_array_equal(labels[0], labels[1])
        assert adjusted_rand_score(numpy.arange(n) >= 15, labels[0]) == 1.0

    def test_integer_columns(self):
        df, w, _ = planted_grid(10, 10)
        df["x_int"] = (df["x"] * 100).round().astype(int)
        df["y_int"] = (df["y"] * 100).round().astype(int)
        model = Spire(df, w, ["x_int"], "y_int", n_clusters=2, floor=10)
        model.solve()
        assert_valid_partition(model, w, n_regions=2, floor=10)

    def test_verbose_runs(self):
        df, w, _ = planted_grid(10, 10)
        model = Spire(df, w, ["x"], "y", n_clusters=2, floor=10, verbose=True)
        model.solve()
        assert numpy.unique(model.labels_).size == 2

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    @pytest.mark.filterwarnings("ignore:The graph is disconnected")
    def test_mexico_snapshot(self):
        # a linear model keeps the snapshot independent of tree tie-breaking
        expected = [0, 0, 0, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 3, 3, 2]
        expected += [3, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
        w = libpysal.weights.Queen.from_dataframe(MEXICO, use_index=False)
        attrs = [f"PCGDP{year}" for year in range(1950, 2000, 10)]
        model = Spire(
            MEXICO,
            w,
            attrs,
            "PCGDP2000",
            n_clusters=4,
            floor=5,
            islands="ignore",
            adaptive_neighborhoods=True,
            estimator=LinearRegression(),
            random_state=RANDOM_STATE,
        )
        model.solve()
        numpy.testing.assert_equal(model.labels_, expected)


class TestSpireIslands:
    def setup_method(self):
        # same synthetic-island setup as test_skater.py; the index has gaps
        remove = [13, 14, 17, 18, 20, 23, 24, 29]
        self.columbus = COLUMBUS[~COLUMBUS.index.isin(remove)]
        self.w = libpysal.weights.Queen.from_dataframe(
            self.columbus, use_index=False, silence_warnings=True
        )
        self.n_components, _ = csgraph.connected_components(self.w.sparse)
        self.args = (self.columbus, self.w, ["INC", "HOVAL"], "CRIME")
        # islands are too small to grow neighborhoods to the default minimum
        self.kwargs = {"estimator": LinearRegression(), "min_neighborhood": 5}

    def test_islands_increase(self):
        assert self.n_components > 1
        model = Spire(*self.args, **self.kwargs, n_clusters=2, floor=5)
        with pytest.warns(OptimizeWarning, match="The graph is disconnected"):
            model.solve()
        assert_valid_partition(model, self.w, 2 + self.n_components, floor=5)
        assert numpy.unique(model.labels_path_[0]).size == self.n_components

    def test_islands_ignore(self):
        n_clusters = self.n_components + 1
        model = Spire(
            *self.args,
            **self.kwargs,
            n_clusters=n_clusters,
            floor=5,
            islands="ignore",
        )
        with pytest.warns(OptimizeWarning, match="The graph is disconnected"):
            model.solve()
        assert_valid_partition(model, self.w, n_clusters, floor=5)

    @pytest.mark.filterwarnings("ignore:The graph is disconnected")
    def test_island_smaller_than_floor_raises(self):
        model = Spire(*self.args, n_clusters=2, floor=10, min_neighborhood=5)
        with pytest.raises(ValueError, match="Islands must be larger than the floor"):
            model.solve()


class TestSpireReviewFixes:
    def test_solve_does_not_mutate_w(self):
        df, w, _ = planted_grid(10, 10)
        w.transform = "r"
        Spire(df, w, ["x"], "y", n_clusters=2, floor=10).solve()
        assert w.transform == "R"

    def test_docstring_example_path_does_not_increase(self):
        lines = [
            line.strip()[4:]
            for line in Spire.__doc__.splitlines()
            if line.strip().startswith((">>> ", "... "))
        ]
        namespace = {}
        exec("\n".join(lines), namespace)  # noqa: S102
        path = namespace["model"].objective_path_
        assert path.shape == (4,)
        assert numpy.all(numpy.diff(path) <= 0)


class TestSpireNeighborhoodSize:
    def setup_method(self):
        # a 30-area chain: the two end areas have 3-area 2-hop neighborhoods
        X, y = two_slopes(30, 15)
        self.df = pandas.DataFrame({"x": X[:, 0], "y": y})
        self.w = libpysal.weights.lat2W(1, 30)
        self.args = (self.df, self.w, ["x"], "y")
        self.kwargs = {"n_clusters": 2, "floor": 5, "estimator": LinearRegression()}

    def test_default_minimum_is_two_per_parameter(self):
        # one predictor -> 2 * (1 + 1) = 4 areas, so the 3-area ends fail
        model = Spire(*self.args, **self.kwargs)
        with pytest.raises(ValueError, match="adaptive_neighborhoods=True") as error:
            model.solve()
        assert "`min_neighborhood` (4)" in str(error.value)
        assert "2 of 30 areas" in str(error.value)

    def test_check_runs_before_any_local_model_is_fit(self):
        kwargs = {**self.kwargs, "estimator": FitFailsRegressor()}
        with pytest.raises(ValueError, match="adaptive_neighborhoods"):
            Spire(*self.args, **kwargs).solve()

    def test_adaptive_expands_only_small_neighborhoods(self):
        model = Spire(*self.args, **self.kwargs, adaptive_neighborhoods=True)
        model.solve()
        expected = numpy.full(30, 2)
        expected[[0, -1]] = 3
        numpy.testing.assert_array_equal(model.neighborhood_orders_, expected)
        assert adjusted_rand_score(numpy.arange(30) >= 15, model.labels_) == 1.0

    def test_explicit_min_neighborhood_overrides_default(self):
        model = Spire(*self.args, **self.kwargs, min_neighborhood=3)
        model.solve()
        assert (model.neighborhood_orders_ == 2).all()

    @pytest.mark.parametrize("value", [0, -1])
    def test_invalid_min_neighborhood(self, value):
        with pytest.raises(ValueError, match="min_neighborhood"):
            Spire(*self.args, **self.kwargs, min_neighborhood=value)
