# mojo-surprise

`mojo-surprise` is a focused port of
[Surprise](https://github.com/NicolasHug/Surprise)'s collaborative-filtering
recommenders to Mojo. It keeps the Python package name, class names,
constructor signatures, prediction objects, raw/inner ID handling, clipping,
and cold-start behavior of the covered upstream API. The compute-heavy
training, similarity, baseline, and neighbor-aggregation loops run in one
compiled Mojo shared library.

The parity suite compares learned arrays, similarity matrices, estimates, and
prediction details directly with real `scikit-surprise` 1.1.5.

## Coverage

| Area | Covered |
| --- | --- |
| Matrix factorization | `SVD`, biased and unbiased; all per-parameter learning-rate and regularization overrides |
| Implicit factorization | `SVDpp`, including `cache_ratings` API compatibility and all `yj` controls |
| Neighborhood models | `KNNBasic`, `KNNWithMeans`, `KNNWithZScore`, `KNNBaseline`; user- and item-based |
| Similarities | MSD, cosine, Pearson, shrunk Pearson-baseline, `min_support` |
| Baselines | ALS and SGD with upstream option names and defaults |
| Data/API support | `Dataset.load_from_df`, `Dataset.load_from_file`, `Reader`, `Trainset`, `Prediction`, `AlgoBase.test`, `train_test_split` |
| Accuracy | `rmse`, `mse`, `mae` |

Not covered are Surprise's bundled dataset downloader, cross-validation/search
helpers, persistence helpers, `fcp`, and algorithms outside this repository's
scope (`NMF`, `SlopeOne`, `CoClustering`, `NormalPredictor`, and
`BaselineOnly`).

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` provides the pinned Mojo nightly, Python, NumPy, pandas, and the
real upstream package used by the parity tests. The explicit build writes
`dist/libmojo-surprise.so`; importing from a source checkout also rebuilds a
missing or stale library automatically.

## Usage

This example runs as written from the repository root:

```bash
pixi run python - <<'PY'
import pandas as pd
from surprise import Dataset, Reader, SVDpp

ratings = pd.DataFrame(
    [
        ("alice", "film-a", 5),
        ("alice", "film-b", 3),
        ("bob", "film-a", 4),
        ("bob", "film-c", 2),
        ("carol", "film-b", 4),
        ("carol", "film-c", 5),
    ],
    columns=["user", "item", "rating"],
)

data = Dataset.load_from_df(ratings, Reader(rating_scale=(1, 5)))
model = SVDpp(n_factors=16, n_epochs=20, random_state=0)
model.fit(data.build_full_trainset())
print(model.predict("alice", "film-c"))
PY
```

Existing code for the covered subset can continue to use imports such as
`from surprise import SVD, SVDpp, KNNBaseline`.

## Performance

Measured with `pixi run bench`; each implementation ran in a separate warm
worker under the repository's benchmark lock. Times are the best
of repeated runs and exclude process startup and compilation.

| case | mojo-surprise | scikit-surprise | result |
| --- | ---: | ---: | ---: |
| SVD.fit (19.2k ratings, 40 factors, 8 epochs) | 8.4 ms | 30.8 ms | 3.66x faster |
| SVD++.fit (8k ratings, 20 factors, 3 epochs) | 16.0 ms | 30.5 ms | 1.91x faster |
| KNNBasic.fit similarity (800 users, 19.2k ratings) | 10.5 ms | 23.5 ms | 2.24x faster |
| KNNBasic.test (10k predictions, k=40) | 23.5 ms | 269.7 ms | 11.45x faster |

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz,
Linux 6.8.0-136-generic x86-64, glibc 2.39, Python 3.13.14.

SVD and SVD++ use explicit float64 SIMD for factor dot products and updates,
with scalar remainder loops. KNN test batches cross the FFI once, reuse one
scratch allocation for neighbor indices and values, and use SIMD for the
weighted reduction with a scalar tail. Independent prediction rows parallelize
at 2,048 queries; smaller batches remain serial. Similarity construction
intersects sorted rating lists and parallelizes sufficiently large sets of
independent entity rows. Immutable contiguous rating and CSR buffers are cached
on the trainset and passed to Mojo without FFI copies.

No GPU path is included. The factor trainers are sequential SGD with less than
two floating-point operations per byte moved, while similarity construction
has low arithmetic intensity and irregular sparse reads. Neither kernel can
justify device transfer and launch overhead.

## How it works

Python owns object semantics, raw-to-inner ID maps, NumPy allocation, and RNG
initialization. Initializing factor arrays with NumPy's `RandomState` makes the
starting values identical to upstream. Mojo then receives contiguous buffers
and extents through `ctypes`, performs the complete iterative operation, and
returns into the caller-owned arrays.

The FFI exposes non-parametric `@export` functions with `abi("C")`. Array
buffers cross the ABI as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside Mojo. Nothing in the Mojo
library allocates: ratings use compact CSR arrays where adjacency is needed,
factor matrices are row-major `float64`, IDs and offsets are `int64`, and all
scratch storage belongs to NumPy.

All kernels live in `src/kernels.mojo`, so the fixed shared-library build cost
is paid once.

## License

MIT
