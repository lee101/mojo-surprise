from __future__ import annotations

import heapq

from .._core import compute_baselines, compute_similarities
from .predictions import Prediction, PredictionImpossible


class AlgoBase:
    def __init__(self, **kwargs):
        self.bsl_options = dict(kwargs.get("bsl_options", {}))
        self.sim_options = dict(kwargs.get("sim_options", {}))
        self.sim_options.setdefault("user_based", True)

    def fit(self, trainset):
        self.trainset = trainset
        self.bu = self.bi = None
        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = "UKN__" + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = "UKN__" + str(iid)
        details = {}
        try:
            estimate = self.estimate(iuid, iiid)
            if isinstance(estimate, tuple):
                estimate, details = estimate
            details["was_impossible"] = False
        except PredictionImpossible as error:
            estimate = self.default_prediction()
            details["was_impossible"] = True
            details["reason"] = str(error)
        if clip:
            lower, upper = self.trainset.rating_scale
            estimate = min(upper, max(lower, estimate))
        prediction = Prediction(uid, iid, r_ui, estimate, details)
        if verbose:
            print(prediction)
        return prediction

    def default_prediction(self):
        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        return [
            self.predict(uid, iid, rating, verbose=verbose)
            for uid, iid, rating in testset
        ]

    def compute_baselines(self):
        if self.bu is None:
            self.bu, self.bi = compute_baselines(self.trainset, self.bsl_options)
        return self.bu, self.bi

    def compute_similarities(self):
        user_based = self.sim_options["user_based"]
        name = self.sim_options.get("name", "msd").lower()
        min_support = self.sim_options.get("min_support", 1)
        bx = by = None
        if name == "pearson_baseline":
            bu, bi = self.compute_baselines()
            bx, by = (bu, bi) if user_based else (bi, bu)
        return compute_similarities(
            self.trainset,
            user_based,
            name,
            min_support,
            bx,
            by,
            self.sim_options.get("shrinkage", 100),
        )

    def get_neighbors(self, iid, k):
        total = self.trainset.n_users if self.sim_options["user_based"] else self.trainset.n_items
        candidates = [(other, self.sim[iid, other]) for other in range(total) if other != iid]
        return [other for other, _ in heapq.nlargest(k, candidates, key=lambda pair: pair[1])]

