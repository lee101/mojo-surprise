from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

ROWS = [
    ("u1", "i1", 5.0),
    ("u1", "i2", 3.0),
    ("u1", "i4", 1.0),
    ("u2", "i1", 4.0),
    ("u2", "i2", 2.0),
    ("u2", "i3", 2.0),
    ("u3", "i1", 1.0),
    ("u3", "i2", 5.0),
    ("u3", "i3", 4.0),
    ("u4", "i2", 4.0),
    ("u4", "i3", 5.0),
    ("u4", "i4", 2.0),
    ("u5", "i1", 2.0),
    ("u5", "i3", 3.0),
    ("u5", "i4", 5.0),
]


@pytest.fixture
def frame():
    return pd.DataFrame(ROWS, columns=["user", "item", "rating"])


@pytest.fixture
def trainset(frame):
    from surprise import Dataset, Reader

    return Dataset.load_from_df(frame, Reader(rating_scale=(1, 5))).build_full_trainset()


@pytest.fixture(scope="session")
def upstream(tmp_path_factory):
    target = tmp_path_factory.mktemp("upstream") / "snapshot.npz"
    script = f"""
import sys
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, SVDpp
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline

rows = {ROWS!r}
data = Dataset.load_from_df(
    pd.DataFrame(rows, columns=["user", "item", "rating"]),
    Reader(rating_scale=(1, 5)),
)
train = data.build_full_trainset()
result = {{}}

svd = SVD(
    n_factors=5, n_epochs=4, random_state=17, lr_bu=.006, lr_bi=.004,
    lr_pu=.008, lr_qi=.003, reg_bu=.01, reg_bi=.03, reg_pu=.04, reg_qi=.02,
).fit(train)
for name in ("bu", "bi", "pu", "qi"):
    result["svd_" + name] = getattr(svd, name)
result["svd_est"] = [svd.predict(u, i, clip=False).est for u, i in
                     [("u1", "i3"), ("u4", "i1"), ("new", "i2"), ("u2", "new")]]
for factors in (1, 7, 8, 9, 15, 16, 17):
    edge = SVD(n_factors=factors, n_epochs=2, random_state=31).fit(train)
    result[f"svd_edge_{{factors}}_pu"] = edge.pu
    result[f"svd_edge_{{factors}}_qi"] = edge.qi

svdpp = SVDpp(
    n_factors=5, n_epochs=3, random_state=23, lr_bu=.006, lr_bi=.004,
    lr_pu=.008, lr_qi=.003, lr_yj=.009, reg_bu=.01, reg_bi=.03,
    reg_pu=.04, reg_qi=.02, reg_yj=.05, cache_ratings=True,
).fit(train)
for name in ("bu", "bi", "pu", "qi", "yj"):
    result["svdpp_" + name] = getattr(svdpp, name)
result["svdpp_est"] = [svdpp.predict(u, i, clip=False).est for u, i in
                       [("u1", "i3"), ("u4", "i1"), ("new", "i2"), ("u2", "new")]]

classes = [KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline]
queries = [("u1", "i3"), ("u4", "i1"), ("u5", "i2")]
for user_based in (True, False):
    suffix = "u" if user_based else "i"
    for sim_name in ("msd", "cosine", "pearson", "pearson_baseline"):
        algo = KNNBasic(
            k=3,
            sim_options={{"name": sim_name, "user_based": user_based, "min_support": 2}},
            verbose=False,
        ).fit(train)
        result["sim_" + sim_name + "_" + suffix] = algo.sim
    for cls in classes:
        algo = cls(
            k=3,
            sim_options={{"name": "pearson_baseline", "user_based": user_based}},
            verbose=False,
        ).fit(train)
        predictions = [algo.predict(u, i, clip=False) for u, i in queries]
        key = cls.__name__ + "_" + suffix
        result[key + "_est"] = [prediction.est for prediction in predictions]
        result[key + "_actual"] = [
            prediction.details.get("actual_k", -1) for prediction in predictions
        ]
        if cls is KNNBaseline:
            result[key + "_bu"] = algo.bu
            result[key + "_bi"] = algo.bi

sgd = KNNBaseline(
    k=3,
    bsl_options={{"method": "sgd", "n_epochs": 7, "learning_rate": .009, "reg": .04}},
    sim_options={{"name": "pearson_baseline"}},
    verbose=False,
).fit(train)
result["baseline_sgd_bu"] = sgd.bu
result["baseline_sgd_bi"] = sgd.bi
np.savez(sys.argv[1], **result)
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode:
        pytest.fail(
            "real scikit-surprise parity fixture failed:\n"
            + (completed.stderr or completed.stdout)
        )
    import numpy as np

    return np.load(target)
