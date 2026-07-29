from __future__ import annotations

import numpy as np


class Trainset:
    def __init__(
        self,
        ur,
        ir,
        n_users,
        n_items,
        n_ratings,
        rating_scale,
        raw2inner_id_users,
        raw2inner_id_items,
    ):
        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._inner2raw_id_users = None
        self._inner2raw_id_items = None
        self._global_mean = None

    def knows_user(self, uid):
        return uid in self.ur

    def knows_item(self, iid):
        return iid in self.ir

    def to_inner_uid(self, raw_uid):
        try:
            return self._raw2inner_id_users[raw_uid]
        except KeyError:
            raise ValueError(f"User {raw_uid} is not part of the trainset.") from None

    def to_inner_iid(self, raw_iid):
        try:
            return self._raw2inner_id_items[raw_iid]
        except KeyError:
            raise ValueError(f"Item {raw_iid} is not part of the trainset.") from None

    def to_raw_uid(self, inner_uid):
        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {
                inner: raw for raw, inner in self._raw2inner_id_users.items()
            }
        try:
            return self._inner2raw_id_users[inner_uid]
        except KeyError:
            raise ValueError(f"{inner_uid} is not a valid inner id.") from None

    def to_raw_iid(self, inner_iid):
        if self._inner2raw_id_items is None:
            self._inner2raw_id_items = {
                inner: raw for raw, inner in self._raw2inner_id_items.items()
            }
        try:
            return self._inner2raw_id_items[inner_iid]
        except KeyError:
            raise ValueError(f"{inner_iid} is not a valid inner id.") from None

    def all_ratings(self):
        for user, entries in self.ur.items():
            for item, rating in entries:
                yield user, item, rating

    def all_users(self):
        return range(self.n_users)

    def all_items(self):
        return range(self.n_items)

    def build_testset(self):
        return [
            (self.to_raw_uid(user), self.to_raw_iid(item), rating)
            for user, item, rating in self.all_ratings()
        ]

    def build_anti_testset(self, fill=None):
        fill = self.global_mean if fill is None else float(fill)
        result = []
        for user in self.all_users():
            rated = {item for item, _ in self.ur[user]}
            result.extend(
                (self.to_raw_uid(user), self.to_raw_iid(item), fill)
                for item in self.all_items()
                if item not in rated
            )
        return result

    @property
    def global_mean(self):
        if self._global_mean is None:
            self._global_mean = np.mean([rating for _, _, rating in self.all_ratings()])
        return self._global_mean

