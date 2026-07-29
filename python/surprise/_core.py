from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib, parallel_available


def rating_arrays(trainset):
    cached = getattr(trainset, "_mojo_rating_arrays", None)
    if cached is None:
        users = np.empty(trainset.n_ratings, dtype=np.int64)
        items = np.empty(trainset.n_ratings, dtype=np.int64)
        ratings = np.empty(trainset.n_ratings, dtype=np.float64)
        for row, (user, item, rating) in enumerate(trainset.all_ratings()):
            users[row] = user
            items[row] = item
            ratings[row] = rating
        cached = users, items, ratings
        trainset._mojo_rating_arrays = cached
    return cached


def csr(rating_dict, size: int, *, sort: bool = False):
    offsets = np.zeros(size + 1, dtype=np.int64)
    ids: list[int] = []
    values: list[float] = []
    for x in range(size):
        entries = rating_dict[x]
        if sort:
            entries = sorted(entries, key=lambda pair: pair[0])
        ids.extend(pair[0] for pair in entries)
        values.extend(pair[1] for pair in entries)
        offsets[x + 1] = len(ids)
    return offsets, i64(ids), f64(values)


def trainset_csr(trainset, user_based: bool, *, sort: bool = False):
    cache = getattr(trainset, "_mojo_csr", None)
    if cache is None:
        cache = {}
        trainset._mojo_csr = cache
    key = user_based, sort
    if key not in cache:
        rating_dict = trainset.ur if user_based else trainset.ir
        size = trainset.n_users if user_based else trainset.n_items
        cache[key] = csr(rating_dict, size, sort=sort)
    return cache[key]


def compute_similarities(trainset, user_based, name, min_support, bx=None, by=None, shrinkage=100):
    n_x = trainset.n_users if user_based else trainset.n_items
    n_y = trainset.n_items if user_based else trainset.n_users
    offsets, other_ids, ratings = trainset_csr(
        trainset, user_based, sort=True
    )
    matrix = np.empty((n_x, n_x), dtype=np.float64)
    bx = np.zeros(n_x, dtype=np.float64) if bx is None else f64(bx)
    by = np.zeros(n_y, dtype=np.float64) if by is None else f64(by)
    kinds = {"msd": 0, "cosine": 1, "pearson": 2, "pearson_baseline": 3}
    if name not in kinds:
        raise NameError(
            f"Wrong sim name {name}. Allowed values are "
            + ", ".join(kinds)
            + "."
        )
    if name == "pearson_baseline":
        min_support = max(2, min_support)
    lib().msu_similarities(
        addr(offsets, dtype=np.int64, min_size=n_x + 1),
        addr(other_ids, dtype=np.int64, min_size=trainset.n_ratings),
        addr(ratings, dtype=np.float64, min_size=trainset.n_ratings),
        addr(matrix, dtype=np.float64, min_size=n_x * n_x, writable=True),
        addr(bx, dtype=np.float64, min_size=n_x),
        addr(by, dtype=np.float64, min_size=n_y),
        n_x,
        min_support,
        kinds[name],
        trainset.global_mean,
        shrinkage,
        int(n_x >= 256 and parallel_available()),
    )
    return matrix


def compute_baselines(trainset, options):
    bu = np.zeros(trainset.n_users, dtype=np.float64)
    bi = np.zeros(trainset.n_items, dtype=np.float64)
    method = options.get("method", "als")
    if method == "als":
        uo, ui, ur = trainset_csr(trainset, True)
        io, iu, ir = trainset_csr(trainset, False)
        lib().msu_baseline_als(
            addr(uo, dtype=np.int64, min_size=trainset.n_users + 1),
            addr(ui, dtype=np.int64, min_size=trainset.n_ratings),
            addr(ur, dtype=np.float64, min_size=trainset.n_ratings),
            addr(io, dtype=np.int64, min_size=trainset.n_items + 1),
            addr(iu, dtype=np.int64, min_size=trainset.n_ratings),
            addr(ir, dtype=np.float64, min_size=trainset.n_ratings),
            addr(bu, dtype=np.float64, min_size=trainset.n_users, writable=True),
            addr(bi, dtype=np.float64, min_size=trainset.n_items, writable=True),
            trainset.n_users,
            trainset.n_items,
            options.get("n_epochs", 10),
            trainset.global_mean,
            options.get("reg_u", 15),
            options.get("reg_i", 10),
        )
    elif method == "sgd":
        users, items, ratings = rating_arrays(trainset)
        lib().msu_baseline_sgd(
            addr(users, dtype=np.int64, min_size=trainset.n_ratings),
            addr(items, dtype=np.int64, min_size=trainset.n_ratings),
            addr(ratings, dtype=np.float64, min_size=trainset.n_ratings),
            addr(bu, dtype=np.float64, min_size=trainset.n_users, writable=True),
            addr(bi, dtype=np.float64, min_size=trainset.n_items, writable=True),
            trainset.n_ratings,
            options.get("n_epochs", 20),
            trainset.global_mean,
            options.get("learning_rate", 0.005),
            options.get("reg", 0.02),
        )
    else:
        raise ValueError(
            f"Invalid method {method} for baseline computation. "
            "Available methods are als and sgd."
        )
    return bu, bi


def knn_one(algo, x: int, y: int, mode: int):
    xs = i64([x])
    ys = i64([y])
    estimates = np.empty(1, dtype=np.float64)
    actual = np.empty(1, dtype=np.int64)
    score_work = np.empty(algo.k, dtype=np.float64)
    neighbor_work = np.empty(algo.k, dtype=np.int64)
    stat1 = algo._stat1
    stat2 = algo._stat2
    y_bias = algo._y_bias
    lib().msu_knn_predict(
        addr(xs, dtype=np.int64, min_size=1),
        addr(ys, dtype=np.int64, min_size=1),
        addr(algo._yr_offsets, dtype=np.int64, min_size=algo.n_y + 1),
        addr(algo._yr_neighbors, dtype=np.int64, min_size=algo.trainset.n_ratings),
        addr(algo._yr_ratings, dtype=np.float64, min_size=algo.trainset.n_ratings),
        addr(algo.sim, dtype=np.float64, min_size=algo.n_x * algo.n_x),
        addr(stat1, dtype=np.float64, min_size=algo.n_x),
        addr(stat2, dtype=np.float64, min_size=algo.n_x),
        addr(y_bias, dtype=np.float64, min_size=algo.n_y),
        addr(estimates, dtype=np.float64, min_size=1, writable=True),
        addr(actual, dtype=np.int64, min_size=1, writable=True),
        addr(score_work, dtype=np.float64, min_size=algo.k, writable=True),
        addr(neighbor_work, dtype=np.int64, min_size=algo.k, writable=True),
        1,
        algo.n_x,
        algo.k,
        algo.min_k,
        mode,
        algo.trainset.global_mean,
    )
    return float(estimates[0]), int(actual[0])
