from __future__ import annotations

import importlib.metadata

import numpy as np
import pandas as pd
import pytest

from surprise import (
    Dataset,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    Reader,
    SVD,
    SVDpp,
)
from surprise import accuracy
from surprise.model_selection import train_test_split


def test_upstream_distribution_is_installed():
    version = importlib.metadata.version("scikit-surprise")
    assert tuple(map(int, version.split("."))) >= (1, 1, 5)


def test_trainset_contract(trainset):
    assert (trainset.n_users, trainset.n_items, trainset.n_ratings) == (5, 4, 15)
    assert trainset.to_raw_uid(trainset.to_inner_uid("u3")) == "u3"
    assert trainset.to_raw_iid(trainset.to_inner_iid("i4")) == "i4"
    assert len(trainset.build_testset()) == 15
    assert len(trainset.build_anti_testset()) == 5
    with pytest.raises(ValueError):
        trainset.to_inner_uid("missing")


def test_trainset_buffers_are_contiguous_and_reused(trainset):
    from surprise._core import rating_arrays, trainset_csr

    first_ratings = rating_arrays(trainset)
    second_ratings = rating_arrays(trainset)
    first_csr = trainset_csr(trainset, True, sort=True)
    second_csr = trainset_csr(trainset, True, sort=True)
    assert all(first is second for first, second in zip(first_ratings, second_ratings))
    assert all(first is second for first, second in zip(first_csr, second_csr))
    assert all(array.flags.c_contiguous for array in first_ratings + first_csr)


def test_ffi_rejects_wrong_dtype_stride_size_and_mutability():
    from surprise._lib import addr

    with pytest.raises(TypeError, match="dtype"):
        addr(np.ones(2, dtype=np.float32), dtype=np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((2, 2), dtype=np.float64)[:, 0])
    with pytest.raises(ValueError, match="expected at least"):
        addr(np.ones(1, dtype=np.float64), min_size=2)
    readonly = np.ones(1, dtype=np.float64)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        addr(readonly, writable=True)


def _fit_svd(trainset):
    return SVD(
        n_factors=5,
        n_epochs=4,
        random_state=17,
        lr_bu=0.006,
        lr_bi=0.004,
        lr_pu=0.008,
        lr_qi=0.003,
        reg_bu=0.01,
        reg_bi=0.03,
        reg_pu=0.04,
        reg_qi=0.02,
    ).fit(trainset)


def test_svd_learned_parameters_match_upstream(trainset, upstream):
    model = _fit_svd(trainset)
    for name in ("bu", "bi", "pu", "qi"):
        assert np.allclose(
            getattr(model, name), upstream["svd_" + name], rtol=0, atol=2e-15
        )


def test_svd_predictions_match_upstream(trainset, upstream):
    model = _fit_svd(trainset)
    queries = [("u1", "i3"), ("u4", "i1"), ("new", "i2"), ("u2", "new")]
    estimates = [model.predict(user, item, clip=False).est for user, item in queries]
    assert np.array_equal(estimates, upstream["svd_est"])


def test_svd_unbiased_unknown_and_clipping(trainset):
    model = SVD(
        n_factors=3, n_epochs=2, biased=False, random_state=0
    ).fit(trainset)
    unknown = model.predict("new", "i1")
    assert unknown.details["was_impossible"]
    assert unknown.est == pytest.approx(trainset.global_mean)
    assert 1 <= model.predict("u1", "i3").est <= 5


@pytest.mark.parametrize("n_factors", [1, 7, 8, 9, 15, 16, 17])
def test_svd_simd_boundaries_match_upstream(trainset, upstream, n_factors):
    model = SVD(n_factors=n_factors, n_epochs=2, random_state=31).fit(trainset)
    assert np.allclose(
        model.pu, upstream[f"svd_edge_{n_factors}_pu"], rtol=0, atol=3e-15
    )
    assert np.allclose(
        model.qi, upstream[f"svd_edge_{n_factors}_qi"], rtol=0, atol=3e-15
    )


def _fit_svdpp(trainset):
    return SVDpp(
        n_factors=5,
        n_epochs=3,
        random_state=23,
        lr_bu=0.006,
        lr_bi=0.004,
        lr_pu=0.008,
        lr_qi=0.003,
        lr_yj=0.009,
        reg_bu=0.01,
        reg_bi=0.03,
        reg_pu=0.04,
        reg_qi=0.02,
        reg_yj=0.05,
        cache_ratings=True,
    ).fit(trainset)


def test_svdpp_learned_parameters_match_upstream(trainset, upstream):
    model = _fit_svdpp(trainset)
    for name in ("bu", "bi", "pu", "qi", "yj"):
        assert np.allclose(
            getattr(model, name), upstream["svdpp_" + name], rtol=0, atol=2e-15
        )


def test_svdpp_predictions_match_upstream(trainset, upstream):
    model = _fit_svdpp(trainset)
    queries = [("u1", "i3"), ("u4", "i1"), ("new", "i2"), ("u2", "new")]
    estimates = [model.predict(user, item, clip=False).est for user, item in queries]
    assert np.allclose(estimates, upstream["svdpp_est"], rtol=0, atol=1e-15)


def test_svdpp_cache_ratings_compatibility(trainset):
    options = dict(n_factors=5, n_epochs=2, random_state=23)
    cached = SVDpp(cache_ratings=True, **options).fit(trainset)
    uncached = SVDpp(cache_ratings=False, **options).fit(trainset)
    for name in ("bu", "bi", "pu", "qi", "yj"):
        assert np.array_equal(getattr(cached, name), getattr(uncached, name))


@pytest.mark.parametrize("name", ["msd", "cosine", "pearson", "pearson_baseline"])
@pytest.mark.parametrize("user_based", [True, False], ids=["user", "item"])
def test_similarity_matrix_matches_upstream(trainset, upstream, name, user_based):
    model = KNNBasic(
        k=3,
        sim_options={"name": name, "user_based": user_based, "min_support": 2},
        verbose=False,
    ).fit(trainset)
    suffix = "u" if user_based else "i"
    assert np.allclose(model.sim, upstream[f"sim_{name}_{suffix}"], rtol=0, atol=2e-15)


def test_similarity_below_parallel_threshold_stays_serial(trainset, monkeypatch):
    import surprise._core as core

    def unexpected_parallel_initialization():
        raise AssertionError("small similarity matrix initialized parallel runtime")

    monkeypatch.setattr(core, "parallel_available", unexpected_parallel_initialization)
    KNNBasic(k=3, sim_options={"name": "msd"}, verbose=False).fit(trainset)


def test_similarity_parallel_threshold_matches_reference():
    n_users = 256
    profiles = np.array(
        [
            [1.0 + ((user + 3 * item) % 5) for item in range(4)]
            for user in range(n_users)
        ]
    )
    rows = [
        (f"u{user}", f"i{item}", profiles[user, item])
        for user in range(n_users)
        for item in range(4)
    ]
    trainset = Dataset.load_from_df(
        pd.DataFrame(rows, columns=["user", "item", "rating"]),
        Reader(rating_scale=(1, 5)),
    ).build_full_trainset()
    model = KNNBasic(
        k=3,
        sim_options={"name": "msd", "user_based": True},
        verbose=False,
    ).fit(trainset)
    differences = profiles[:, None, :] - profiles[None, :, :]
    expected = 1.0 / (np.mean(differences * differences, axis=2) + 1.0)
    assert np.allclose(model.sim, expected, rtol=0, atol=2e-15)


@pytest.mark.parametrize(
    "algorithm", [KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline]
)
@pytest.mark.parametrize("user_based", [True, False], ids=["user", "item"])
def test_knn_predictions_match_upstream(
    trainset, upstream, algorithm, user_based
):
    model = algorithm(
        k=3,
        sim_options={"name": "pearson_baseline", "user_based": user_based},
        verbose=False,
    ).fit(trainset)
    predictions = [
        model.predict(user, item, clip=False)
        for user, item in [("u1", "i3"), ("u4", "i1"), ("u5", "i2")]
    ]
    suffix = "u" if user_based else "i"
    key = algorithm.__name__ + "_" + suffix
    assert np.allclose(
        [prediction.est for prediction in predictions],
        upstream[key + "_est"],
        rtol=0,
        atol=2e-15,
    )
    assert np.array_equal(
        [prediction.details.get("actual_k", -1) for prediction in predictions],
        upstream[key + "_actual"],
    )
    if algorithm is KNNBaseline:
        assert np.array_equal(model.bu, upstream[key + "_bu"])
        assert np.array_equal(model.bi, upstream[key + "_bi"])


def test_baseline_sgd_matches_upstream(trainset, upstream):
    model = KNNBaseline(
        k=3,
        bsl_options={
            "method": "sgd",
            "n_epochs": 7,
            "learning_rate": 0.009,
            "reg": 0.04,
        },
        sim_options={"name": "pearson_baseline"},
        verbose=False,
    ).fit(trainset)
    assert np.allclose(model.bu, upstream["baseline_sgd_bu"], rtol=0, atol=2e-15)
    assert np.allclose(model.bi, upstream["baseline_sgd_bi"], rtol=0, atol=2e-15)


def test_knn_fallback_semantics(trainset):
    basic = KNNBasic(
        k=2, min_k=99, sim_options={"name": "pearson"}, verbose=False
    ).fit(trainset)
    prediction = basic.predict("u1", "i3")
    assert prediction.details["was_impossible"]
    assert prediction.est == pytest.approx(trainset.global_mean)

    baseline = KNNBaseline(k=2, verbose=False).fit(trainset)
    prediction = baseline.predict("missing", "i1", clip=False)
    assert not prediction.details["was_impossible"]
    assert prediction.est == pytest.approx(trainset.global_mean + baseline.bi[0])


def test_prediction_batch_and_accuracy(trainset):
    model = SVD(n_factors=4, n_epochs=2, random_state=1).fit(trainset)
    predictions = model.test(trainset.build_testset())
    errors = np.array([prediction.r_ui - prediction.est for prediction in predictions])
    assert accuracy.rmse(predictions, verbose=False) == pytest.approx(
        np.sqrt(np.mean(errors**2))
    )
    assert accuracy.mse(predictions, verbose=False) == pytest.approx(np.mean(errors**2))
    assert accuracy.mae(predictions, verbose=False) == pytest.approx(np.mean(abs(errors)))


def test_train_test_split_and_reader(tmp_path, frame):
    data = Dataset.load_from_df(frame, Reader(rating_scale=(1, 5)))
    train, test = train_test_split(data, test_size=3, random_state=7)
    assert train.n_ratings == 12
    assert len(test) == 3

    path = tmp_path / "ratings.txt"
    path.write_text("u1|i1|5|100\nu2|i2|3|101\n", encoding="utf-8")
    loaded = Dataset.load_from_file(
        path,
        Reader(
            line_format="user item rating timestamp",
            sep="|",
            rating_scale=(1, 5),
        ),
    ).build_full_trainset()
    assert list(loaded.all_ratings()) == [(0, 0, 5.0), (1, 1, 3.0)]
