from __future__ import annotations

import numpy as np


def train_test_split(
    data,
    test_size=0.2,
    train_size=None,
    random_state=None,
    shuffle=True,
):
    ratings = list(data.raw_ratings)
    n = len(ratings)
    if train_size is not None:
        n_train = int(train_size * n) if isinstance(train_size, float) else int(train_size)
        n_test = n - n_train
    else:
        n_test = int(np.ceil(test_size * n)) if isinstance(test_size, float) else int(test_size)
        n_train = n - n_test
    indices = np.arange(n)
    if shuffle:
        np.random.RandomState(random_state).shuffle(indices)
    train_raw = [ratings[index] for index in indices[:n_train]]
    test_raw = [ratings[index] for index in indices[n_train : n_train + n_test]]
    return data.construct_trainset(train_raw), data.construct_testset(test_raw)

