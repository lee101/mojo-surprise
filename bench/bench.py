from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_data(n_users=800, n_items=600, per_user=24, seed=7):
    rng = np.random.RandomState(seed)
    rows = []
    for user in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False)
        latent = (user % 13) / 13
        for item in items:
            rating = 3.0 + 0.7 * np.sin(item * 0.17) + 0.5 * latent
            rating += rng.normal(scale=0.35)
            rows.append((f"u{user}", f"i{item}", float(np.clip(rating, 1, 5))))
    return pd.DataFrame(rows, columns=["user", "item", "rating"])


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def worker():
    from surprise import Dataset, KNNBasic, Reader, SVD, SVDpp

    frame = make_data()
    trainset = Dataset.load_from_df(
        frame, Reader(rating_scale=(1, 5))
    ).build_full_trainset()
    results = {}

    def fit_svd():
        SVD(n_factors=40, n_epochs=8, random_state=0).fit(trainset)

    fit_svd()
    results["SVD.fit (19.2k ratings, 40 factors, 8 epochs)"] = best_time(fit_svd)

    smaller = Dataset.load_from_df(
        make_data(n_users=500, n_items=450, per_user=16, seed=11),
        Reader(rating_scale=(1, 5)),
    ).build_full_trainset()

    def fit_svdpp():
        SVDpp(n_factors=20, n_epochs=3, random_state=0, cache_ratings=True).fit(smaller)

    fit_svdpp()
    results["SVD++.fit (8k ratings, 20 factors, 3 epochs)"] = best_time(
        fit_svdpp, repeat=2
    )

    def fit_knn():
        KNNBasic(
            k=40, sim_options={"name": "msd", "user_based": True}, verbose=False
        ).fit(trainset)

    fit_knn()
    results["KNNBasic.fit similarity (800 users, 19.2k ratings)"] = best_time(
        fit_knn
    )

    knn = KNNBasic(
        k=40, sim_options={"name": "msd", "user_based": True}, verbose=False
    ).fit(trainset)
    queries = [
        (f"u{index % 800}", f"i{(index * 37 + 19) % 600}", 3.0)
        for index in range(10_000)
    ]
    knn.test(queries)
    results["KNNBasic.test (10k predictions, k=40)"] = best_time(
        lambda: knn.test(queries)
    )
    print(json.dumps(results))


def run_worker(upstream: bool):
    env = os.environ.copy()
    if upstream:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(ROOT / "python")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    ours = run_worker(False)
    upstream = run_worker(True)
    print("| case | mojo-surprise | scikit-surprise | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time in ours.items():
        python_time = upstream[name]
        ratio = python_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1000:.1f} ms | "
            f"{python_time * 1000:.1f} ms | {result} |"
        )
    processor = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                processor = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    processor = processor or platform.processor()
    processor = processor or platform.machine()
    print(f"\nMachine: {processor}; {platform.platform()}; Python {platform.python_version()}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()
