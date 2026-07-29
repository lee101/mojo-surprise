from __future__ import annotations

import numpy as np

from .._core import knn_one, trainset_csr
from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class SymmetricAlgo(AlgoBase):
    def __init__(self, sim_options={}, verbose=True, **kwargs):
        super().__init__(sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        user_based = self.sim_options["user_based"]
        self.n_x = trainset.n_users if user_based else trainset.n_items
        self.n_y = trainset.n_items if user_based else trainset.n_users
        self.xr = trainset.ur if user_based else trainset.ir
        self.yr = trainset.ir if user_based else trainset.ur
        (
            self._yr_offsets,
            self._yr_neighbors,
            self._yr_ratings,
        ) = trainset_csr(trainset, not user_based)
        return self

    def switch(self, user_value, item_value):
        return (user_value, item_value) if self.sim_options["user_based"] else (item_value, user_value)

    def _set_stats(self, first=None, second=None, y_bias=None):
        self._stat1 = (
            np.zeros(self.n_x, dtype=np.float64)
            if first is None
            else np.ascontiguousarray(first, dtype=np.float64)
        )
        self._stat2 = (
            np.ones(self.n_x, dtype=np.float64)
            if second is None
            else np.ascontiguousarray(second, dtype=np.float64)
        )
        self._y_bias = (
            np.zeros(self.n_y, dtype=np.float64)
            if y_bias is None
            else np.ascontiguousarray(y_bias, dtype=np.float64)
        )


class KNNBasic(SymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        super().__init__(sim_options=sim_options, verbose=verbose, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        super().fit(trainset)
        if self.k <= 0:
            raise ValueError("k must be greater than zero")
        self.sim = self.compute_similarities()
        self._set_stats()
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        x, y = self.switch(u, i)
        estimate, actual = knn_one(self, x, y, 0)
        if actual < self.min_k:
            raise PredictionImpossible("Not enough neighbors.")
        return estimate, {"actual_k": actual}


class KNNWithMeans(SymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        super().__init__(sim_options=sim_options, verbose=verbose, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        super().fit(trainset)
        if self.k <= 0:
            raise ValueError("k must be greater than zero")
        self.sim = self.compute_similarities()
        self.means = np.array(
            [np.mean([rating for _, rating in self.xr[x]]) for x in range(self.n_x)]
        )
        self._set_stats(self.means)
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        x, y = self.switch(u, i)
        estimate, actual = knn_one(self, x, y, 1)
        return estimate, {"actual_k": actual}


class KNNWithZScore(SymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        super().__init__(sim_options=sim_options, verbose=verbose, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        super().fit(trainset)
        if self.k <= 0:
            raise ValueError("k must be greater than zero")
        all_values = [rating for _, _, rating in trainset.all_ratings()]
        self.overall_sigma = np.std(all_values)
        self.means = np.empty(self.n_x)
        self.sigmas = np.empty(self.n_x)
        for x in range(self.n_x):
            values = [rating for _, rating in self.xr[x]]
            self.means[x] = np.mean(values)
            sigma = np.std(values)
            self.sigmas[x] = self.overall_sigma if sigma == 0 else sigma
        self.sim = self.compute_similarities()
        self._set_stats(self.means, self.sigmas)
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        x, y = self.switch(u, i)
        estimate, actual = knn_one(self, x, y, 2)
        return estimate, {"actual_k": actual}


class KNNBaseline(SymmetricAlgo):
    def __init__(
        self,
        k=40,
        min_k=1,
        sim_options={},
        bsl_options={},
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            sim_options=sim_options,
            bsl_options=bsl_options,
            verbose=verbose,
            **kwargs,
        )
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        super().fit(trainset)
        if self.k <= 0:
            raise ValueError("k must be greater than zero")
        self.bu, self.bi = self.compute_baselines()
        self.bx, self.by = self.switch(self.bu, self.bi)
        self.sim = self.compute_similarities()
        self._set_stats(self.bx, y_bias=self.by)
        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        estimate = self.trainset.global_mean
        if known_user:
            estimate += self.bu[u]
        if known_item:
            estimate += self.bi[i]
        if not (known_user and known_item):
            return float(estimate)
        x, y = self.switch(u, i)
        estimate, actual = knn_one(self, x, y, 3)
        return estimate, {"actual_k": actual}
