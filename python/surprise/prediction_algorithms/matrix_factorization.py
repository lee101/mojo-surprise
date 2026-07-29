from __future__ import annotations

import numpy as np

from .._core import rating_arrays, trainset_csr
from .._lib import addr, lib
from .algo_base import AlgoBase
from .predictions import PredictionImpossible


def _rng(random_state):
    if isinstance(random_state, np.random.RandomState):
        return random_state
    if random_state is None:
        return np.random.mtrand._rand
    return np.random.RandomState(random_state)


class SVD(AlgoBase):
    def __init__(
        self,
        n_factors=100,
        n_epochs=20,
        biased=True,
        init_mean=0,
        init_std_dev=0.1,
        lr_all=0.005,
        reg_all=0.02,
        lr_bu=None,
        lr_bi=None,
        lr_pu=None,
        lr_qi=None,
        reg_bu=None,
        reg_bi=None,
        reg_pu=None,
        reg_qi=None,
        random_state=None,
        verbose=False,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_all if lr_bu is None else lr_bu
        self.lr_bi = lr_all if lr_bi is None else lr_bi
        self.lr_pu = lr_all if lr_pu is None else lr_pu
        self.lr_qi = lr_all if lr_qi is None else lr_qi
        self.reg_bu = reg_all if reg_bu is None else reg_bu
        self.reg_bi = reg_all if reg_bi is None else reg_bi
        self.reg_pu = reg_all if reg_pu is None else reg_pu
        self.reg_qi = reg_all if reg_qi is None else reg_qi
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        random = _rng(self.random_state)
        self.bu = np.zeros(trainset.n_users, dtype=np.float64)
        self.bi = np.zeros(trainset.n_items, dtype=np.float64)
        self.pu = np.ascontiguousarray(
            random.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        )
        self.qi = np.ascontiguousarray(
            random.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        )
        users, items, ratings = rating_arrays(trainset)
        lib().msu_svd_train(
            addr(users, dtype=np.int64, min_size=trainset.n_ratings),
            addr(items, dtype=np.int64, min_size=trainset.n_ratings),
            addr(ratings, dtype=np.float64, min_size=trainset.n_ratings),
            addr(self.bu, dtype=np.float64, min_size=trainset.n_users, writable=True),
            addr(self.bi, dtype=np.float64, min_size=trainset.n_items, writable=True),
            addr(self.pu, dtype=np.float64, min_size=trainset.n_users * self.n_factors, writable=True),
            addr(self.qi, dtype=np.float64, min_size=trainset.n_items * self.n_factors, writable=True),
            trainset.n_ratings,
            self.n_factors,
            self.n_epochs,
            trainset.global_mean if self.biased else 0.0,
            int(self.biased),
            self.lr_bu,
            self.lr_bi,
            self.lr_pu,
            self.lr_qi,
            self.reg_bu,
            self.reg_bi,
            self.reg_pu,
            self.reg_qi,
        )
        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if self.biased:
            estimate = self.trainset.global_mean
            if known_user:
                estimate += self.bu[u]
            if known_item:
                estimate += self.bi[i]
            if known_user and known_item:
                estimate += np.dot(self.qi[i], self.pu[u])
            return float(estimate)
        if known_user and known_item:
            return float(np.dot(self.qi[i], self.pu[u]))
        raise PredictionImpossible("User and item are unknown.")


class SVDpp(AlgoBase):
    def __init__(
        self,
        n_factors=20,
        n_epochs=20,
        init_mean=0,
        init_std_dev=0.1,
        lr_all=0.007,
        reg_all=0.02,
        lr_bu=None,
        lr_bi=None,
        lr_pu=None,
        lr_qi=None,
        lr_yj=None,
        reg_bu=None,
        reg_bi=None,
        reg_pu=None,
        reg_qi=None,
        reg_yj=None,
        random_state=None,
        verbose=False,
        cache_ratings=False,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_all if lr_bu is None else lr_bu
        self.lr_bi = lr_all if lr_bi is None else lr_bi
        self.lr_pu = lr_all if lr_pu is None else lr_pu
        self.lr_qi = lr_all if lr_qi is None else lr_qi
        self.lr_yj = lr_all if lr_yj is None else lr_yj
        self.reg_bu = reg_all if reg_bu is None else reg_bu
        self.reg_bi = reg_all if reg_bi is None else reg_bi
        self.reg_pu = reg_all if reg_pu is None else reg_pu
        self.reg_qi = reg_all if reg_qi is None else reg_qi
        self.reg_yj = reg_all if reg_yj is None else reg_yj
        self.random_state = random_state
        self.verbose = verbose
        self.cache_ratings = cache_ratings

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        random = _rng(self.random_state)
        self.bu = np.zeros(trainset.n_users, dtype=np.float64)
        self.bi = np.zeros(trainset.n_items, dtype=np.float64)
        self.pu = np.ascontiguousarray(
            random.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        )
        self.qi = np.ascontiguousarray(
            random.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        )
        self.yj = np.ascontiguousarray(
            random.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        )
        users, items, ratings = rating_arrays(trainset)
        offsets, user_items, _ = trainset_csr(trainset, True)
        work = np.empty(self.n_factors, dtype=np.float64)
        lib().msu_svdpp_train(
            addr(users, dtype=np.int64, min_size=trainset.n_ratings),
            addr(items, dtype=np.int64, min_size=trainset.n_ratings),
            addr(ratings, dtype=np.float64, min_size=trainset.n_ratings),
            addr(offsets, dtype=np.int64, min_size=trainset.n_users + 1),
            addr(user_items, dtype=np.int64, min_size=trainset.n_ratings),
            addr(self.bu, dtype=np.float64, min_size=trainset.n_users, writable=True),
            addr(self.bi, dtype=np.float64, min_size=trainset.n_items, writable=True),
            addr(self.pu, dtype=np.float64, min_size=trainset.n_users * self.n_factors, writable=True),
            addr(self.qi, dtype=np.float64, min_size=trainset.n_items * self.n_factors, writable=True),
            addr(self.yj, dtype=np.float64, min_size=trainset.n_items * self.n_factors, writable=True),
            addr(work, dtype=np.float64, min_size=self.n_factors, writable=True),
            trainset.n_ratings,
            self.n_factors,
            self.n_epochs,
            trainset.global_mean,
            self.lr_bu,
            self.lr_bi,
            self.lr_pu,
            self.lr_qi,
            self.lr_yj,
            self.reg_bu,
            self.reg_bi,
            self.reg_pu,
            self.reg_qi,
            self.reg_yj,
        )
        return self

    def estimate(self, u, i):
        estimate = self.trainset.global_mean
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user:
            estimate += self.bu[u]
        if known_item:
            estimate += self.bi[i]
        if known_user and known_item:
            implicit = sum((self.yj[j] for j, _ in self.trainset.ur[u]), start=np.zeros(self.n_factors))
            implicit /= np.sqrt(len(self.trainset.ur[u]))
            estimate += np.dot(self.qi[i], self.pu[u] + implicit)
        return float(estimate)
