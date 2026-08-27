from max.algorithm import parallelize
from std.math import sqrt
from std.sys import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def dot(a: FPtr, b: FPtr, n: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var total = 0.0
    var f = 0
    if n >= W:
        var vector_total = a.load[width=W](0) * b.load[width=W](0)
        f = W
        while f + W <= n:
            vector_total += a.load[width=W](f) * b.load[width=W](f)
            f += W
        total = vector_total.reduce_add()
    while f < n:
        total += a[f] * b[f]
        f += 1
    return total


def dot_plus(a: FPtr, b: FPtr, c: FPtr, n: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var total = 0.0
    var f = 0
    if n >= W:
        var vector_total = a.load[width=W](0) * (
            b.load[width=W](0) + c.load[width=W](0)
        )
        f = W
        while f + W <= n:
            vector_total += a.load[width=W](f) * (
                b.load[width=W](f) + c.load[width=W](f)
            )
            f += W
        total = vector_total.reduce_add()
    while f < n:
        total += a[f] * (b[f] + c[f])
        f += 1
    return total


def svd_train(
    users: IPtr,
    items: IPtr,
    ratings: FPtr,
    bu: FPtr,
    bi: FPtr,
    pu: FPtr,
    qi: FPtr,
    n_ratings: Int,
    n_factors: Int,
    n_epochs: Int,
    global_mean: Float64,
    biased: Bool,
    lr_bu: Float64,
    lr_bi: Float64,
    lr_pu: Float64,
    lr_qi: Float64,
    reg_bu: Float64,
    reg_bi: Float64,
    reg_pu: Float64,
    reg_qi: Float64,
):
    comptime W = simdwidthof[DType.float64]()
    var vector_limit = n_factors - n_factors % W
    for _ in range(n_epochs):
        for row in range(n_ratings):
            var u = Int(users[row])
            var i = Int(items[row])
            var p = pu + u * n_factors
            var q = qi + i * n_factors
            var err = ratings[row] - (global_mean + bu[u] + bi[i] + dot(p, q, n_factors))
            if biased:
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])
            for f in range(0, vector_limit, W):
                var pv = p.load[width=W](f)
                var qv = q.load[width=W](f)
                p.store(f, pv + lr_pu * (err * qv - reg_pu * pv))
                q.store(f, qv + lr_qi * (err * pv - reg_qi * qv))
            for f in range(vector_limit, n_factors):
                var puf = p[f]
                var qif = q[f]
                p[f] += lr_pu * (err * qif - reg_pu * puf)
                q[f] += lr_qi * (err * puf - reg_qi * qif)


def svdpp_train(
    users: IPtr,
    items: IPtr,
    ratings: FPtr,
    user_offsets: IPtr,
    user_items: IPtr,
    bu: FPtr,
    bi: FPtr,
    pu: FPtr,
    qi: FPtr,
    yj: FPtr,
    implicit_work: FPtr,
    n_ratings: Int,
    n_factors: Int,
    n_epochs: Int,
    global_mean: Float64,
    lr_bu: Float64,
    lr_bi: Float64,
    lr_pu: Float64,
    lr_qi: Float64,
    lr_yj: Float64,
    reg_bu: Float64,
    reg_bi: Float64,
    reg_pu: Float64,
    reg_qi: Float64,
    reg_yj: Float64,
):
    comptime W = simdwidthof[DType.float64]()
    var vector_limit = n_factors - n_factors % W
    var zeroes = SIMD[DType.float64, W](0.0)
    for _ in range(n_epochs):
        for row in range(n_ratings):
            var u = Int(users[row])
            var i = Int(items[row])
            var first = Int(user_offsets[u])
            var last = Int(user_offsets[u + 1])
            var count = last - first
            var root_count = sqrt(Float64(count))
            for f in range(0, vector_limit, W):
                implicit_work.store(f, zeroes)
            for f in range(vector_limit, n_factors):
                implicit_work[f] = 0.0
            for pos in range(first, last):
                var j = Int(user_items[pos])
                var y = yj + j * n_factors
                for f in range(0, vector_limit, W):
                    implicit_work.store(
                        f,
                        implicit_work.load[width=W](f)
                        + y.load[width=W](f) / root_count,
                    )
                for f in range(vector_limit, n_factors):
                    implicit_work[f] += yj[j * n_factors + f] / root_count
            var p = pu + u * n_factors
            var q = qi + i * n_factors
            var score = dot_plus(q, p, implicit_work, n_factors)
            var err = ratings[row] - (global_mean + bu[u] + bi[i] + score)
            bu[u] += lr_bu * (err - reg_bu * bu[u])
            bi[i] += lr_bi * (err - reg_bi * bi[i])
            for f in range(0, vector_limit, W):
                var pv = p.load[width=W](f)
                var qv = q.load[width=W](f)
                var implicit = implicit_work.load[width=W](f)
                p.store(f, pv + lr_pu * (err * qv - reg_pu * pv))
                q.store(
                    f,
                    qv + lr_qi * (err * (pv + implicit) - reg_qi * qv),
                )
                var delta = err * qv / root_count
                for pos in range(first, last):
                    var j = Int(user_items[pos])
                    var y = yj + j * n_factors
                    var yv = y.load[width=W](f)
                    y.store(f, yv + lr_yj * (delta - reg_yj * yv))
            for f in range(vector_limit, n_factors):
                var puf = p[f]
                var qif = q[f]
                p[f] += lr_pu * (err * qif - reg_pu * puf)
                q[f] += lr_qi * (err * (puf + implicit_work[f]) - reg_qi * qif)
                var delta = err * qif / root_count
                for pos in range(first, last):
                    var j = Int(user_items[pos])
                    var index = j * n_factors + f
                    yj[index] += lr_yj * (delta - reg_yj * yj[index])


def similarities(
    offsets: IPtr,
    other_ids: IPtr,
    ratings: FPtr,
    sim: FPtr,
    x_bias: FPtr,
    y_bias: FPtr,
    n_x: Int,
    min_support: Int,
    kind: Int,
    global_mean: Float64,
    shrinkage: Float64,
    allow_parallel: Bool,
):
    @parameter
    def compute_row(x1: Int):
        sim[x1 * n_x + x1] = 1.0
        for x2 in range(x1 + 1, n_x):
            var a = Int(offsets[x1])
            var a_last = Int(offsets[x1 + 1])
            var b = Int(offsets[x2])
            var b_last = Int(offsets[x2 + 1])
            var count = 0
            var prod = 0.0
            var sq1 = 0.0
            var sq2 = 0.0
            var sum1 = 0.0
            var sum2 = 0.0
            while a < a_last and b < b_last:
                var ya = other_ids[a]
                var yb = other_ids[b]
                if ya < yb:
                    a += 1
                    continue
                if yb < ya:
                    b += 1
                    continue
                var r1 = ratings[a]
                var r2 = ratings[b]
                if kind == 3:
                    var y = Int(ya)
                    var partial = global_mean + y_bias[y]
                    r1 -= partial + x_bias[x1]
                    r2 -= partial + x_bias[x2]
                count += 1
                if kind == 0:
                    var diff = r1 - r2
                    prod += diff * diff
                else:
                    prod += r1 * r2
                    sq1 += r1 * r1
                    sq2 += r2 * r2
                    if kind == 2:
                        sum1 += r1
                        sum2 += r2
                a += 1
                b += 1
            var value = 0.0
            if count >= min_support:
                if kind == 0:
                    value = 1.0 / (prod / Float64(count) + 1.0)
                elif kind == 1:
                    var denom = sqrt(sq1 * sq2)
                    if denom != 0.0:
                        value = prod / denom
                elif kind == 2:
                    var n = Float64(count)
                    var d1 = n * sq1 - sum1 * sum1
                    var d2 = n * sq2 - sum2 * sum2
                    if d1 > 0.0 and d2 > 0.0:
                        value = (n * prod - sum1 * sum2) / sqrt(d1 * d2)
                else:
                    var denom = sqrt(sq1 * sq2)
                    if denom != 0.0:
                        value = prod / denom
                        value *= Float64(count - 1) / (Float64(count - 1) + shrinkage)
            sim[x1 * n_x + x2] = value
            sim[x2 * n_x + x1] = value

    if allow_parallel and n_x >= 256:
        parallelize[compute_row](n_x, 32)
    else:
        for x1 in range(n_x):
            compute_row(x1)


def baseline_als(
    user_offsets: IPtr,
    user_items: IPtr,
    user_ratings: FPtr,
    item_offsets: IPtr,
    item_users: IPtr,
    item_ratings: FPtr,
    bu: FPtr,
    bi: FPtr,
    n_users: Int,
    n_items: Int,
    n_epochs: Int,
    global_mean: Float64,
    reg_u: Float64,
    reg_i: Float64,
):
    for _ in range(n_epochs):
        for i in range(n_items):
            var first = Int(item_offsets[i])
            var last = Int(item_offsets[i + 1])
            var dev = 0.0
            for pos in range(first, last):
                dev += item_ratings[pos] - global_mean - bu[Int(item_users[pos])]
            bi[i] = dev / (reg_i + Float64(last - first))
        for u in range(n_users):
            var first = Int(user_offsets[u])
            var last = Int(user_offsets[u + 1])
            var dev = 0.0
            for pos in range(first, last):
                dev += user_ratings[pos] - global_mean - bi[Int(user_items[pos])]
            bu[u] = dev / (reg_u + Float64(last - first))


def baseline_sgd(
    users: IPtr,
    items: IPtr,
    ratings: FPtr,
    bu: FPtr,
    bi: FPtr,
    n_ratings: Int,
    n_epochs: Int,
    global_mean: Float64,
    learning_rate: Float64,
    regularization: Float64,
):
    for _ in range(n_epochs):
        for row in range(n_ratings):
            var u = Int(users[row])
            var i = Int(items[row])
            var err = ratings[row] - (global_mean + bu[u] + bi[i])
            bu[u] += learning_rate * (err - regularization * bu[u])
            bi[i] += learning_rate * (err - regularization * bi[i])


def knn_predict(
    xs: IPtr,
    ys: IPtr,
    offsets: IPtr,
    neighbors: IPtr,
    ratings: FPtr,
    sim: FPtr,
    stat1: FPtr,
    stat2: FPtr,
    y_bias: FPtr,
    estimates: FPtr,
    actual: IPtr,
    scores_work: FPtr,
    neighbors_work: IPtr,
    m: Int,
    n_x: Int,
    k: Int,
    min_k: Int,
    mode: Int,
    global_mean: Float64,
):
    for row in range(m):
        var base = row * k
        var kept = 0
        var y = Int(ys[row])
        var x = Int(xs[row])
        for pos in range(Int(offsets[y]), Int(offsets[y + 1])):
            var nb = Int(neighbors[pos])
            var score = sim[x * n_x + nb]
            if kept == k and score <= scores_work[base + k - 1]:
                continue
            var insert = kept
            if insert == k:
                insert = k - 1
            while insert > 0 and score > scores_work[base + insert - 1]:
                if insert < k:
                    scores_work[base + insert] = scores_work[base + insert - 1]
                    neighbors_work[base + insert] = neighbors_work[base + insert - 1]
                insert -= 1
            scores_work[base + insert] = score
            neighbors_work[base + insert] = Int64(pos)
            if kept < k:
                kept += 1
        var weighted = 0.0
        var sum_sim = 0.0
        var used = 0
        for rank in range(kept):
            var score = scores_work[base + rank]
            if score <= 0.0:
                continue
            var pos = Int(neighbors_work[base + rank])
            var nb = Int(neighbors[pos])
            var value = ratings[pos]
            if mode == 1:
                value -= stat1[nb]
            elif mode == 2:
                value = (value - stat1[nb]) / stat2[nb]
            elif mode == 3:
                value -= global_mean + stat1[nb] + y_bias[y]
            weighted += score * value
            sum_sim += score
            used += 1
        actual[row] = Int64(used)
        if mode == 0:
            estimates[row] = global_mean
            if used >= min_k and sum_sim != 0.0:
                estimates[row] = weighted / sum_sim
        elif mode == 1:
            estimates[row] = stat1[x]
            if used >= min_k and sum_sim != 0.0:
                estimates[row] += weighted / sum_sim
        elif mode == 2:
            estimates[row] = stat1[x]
            if used >= min_k and sum_sim != 0.0:
                estimates[row] += weighted / sum_sim * stat2[x]
        else:
            estimates[row] = global_mean + stat1[x] + y_bias[y]
            if used >= min_k and sum_sim != 0.0:
                estimates[row] += weighted / sum_sim


@export("msu_svd_train")
def msu_svd_train(
    users: Int, items: Int, ratings: Int, bu: Int, bi: Int, pu: Int, qi: Int,
    n_ratings: Int, n_factors: Int, n_epochs: Int, global_mean: Float64,
    biased: Int, lr_bu: Float64, lr_bi: Float64, lr_pu: Float64,
    lr_qi: Float64, reg_bu: Float64, reg_bi: Float64, reg_pu: Float64,
    reg_qi: Float64,
) abi("C"):
    svd_train(ip(users), ip(items), fp(ratings), fp(bu), fp(bi), fp(pu), fp(qi),
              n_ratings, n_factors, n_epochs, global_mean, biased != 0,
              lr_bu, lr_bi, lr_pu, lr_qi, reg_bu, reg_bi, reg_pu, reg_qi)


@export("msu_svdpp_train")
def msu_svdpp_train(
    users: Int, items: Int, ratings: Int, user_offsets: Int, user_items: Int,
    bu: Int, bi: Int, pu: Int, qi: Int, yj: Int, implicit_work: Int,
    n_ratings: Int, n_factors: Int, n_epochs: Int, global_mean: Float64,
    lr_bu: Float64, lr_bi: Float64, lr_pu: Float64, lr_qi: Float64,
    lr_yj: Float64, reg_bu: Float64, reg_bi: Float64, reg_pu: Float64,
    reg_qi: Float64, reg_yj: Float64,
) abi("C"):
    svdpp_train(ip(users), ip(items), fp(ratings), ip(user_offsets), ip(user_items),
                fp(bu), fp(bi), fp(pu), fp(qi), fp(yj), fp(implicit_work),
                n_ratings, n_factors, n_epochs, global_mean, lr_bu, lr_bi,
                lr_pu, lr_qi, lr_yj, reg_bu, reg_bi, reg_pu, reg_qi, reg_yj)


@export("msu_similarities")
def msu_similarities(
    offsets: Int, other_ids: Int, ratings: Int, sim: Int, x_bias: Int,
    y_bias: Int, n_x: Int, min_support: Int, kind: Int,
    global_mean: Float64, shrinkage: Float64, allow_parallel: Int,
) abi("C"):
    similarities(ip(offsets), ip(other_ids), fp(ratings), fp(sim), fp(x_bias),
                 fp(y_bias), n_x, min_support, kind, global_mean, shrinkage,
                 allow_parallel != 0)


@export("msu_baseline_als")
def msu_baseline_als(
    user_offsets: Int, user_items: Int, user_ratings: Int,
    item_offsets: Int, item_users: Int, item_ratings: Int, bu: Int, bi: Int,
    n_users: Int, n_items: Int, n_epochs: Int, global_mean: Float64,
    reg_u: Float64, reg_i: Float64,
) abi("C"):
    baseline_als(ip(user_offsets), ip(user_items), fp(user_ratings),
                 ip(item_offsets), ip(item_users), fp(item_ratings), fp(bu),
                 fp(bi), n_users, n_items, n_epochs, global_mean, reg_u, reg_i)


@export("msu_baseline_sgd")
def msu_baseline_sgd(
    users: Int, items: Int, ratings: Int, bu: Int, bi: Int, n_ratings: Int,
    n_epochs: Int, global_mean: Float64, learning_rate: Float64,
    regularization: Float64,
) abi("C"):
    baseline_sgd(ip(users), ip(items), fp(ratings), fp(bu), fp(bi), n_ratings,
                 n_epochs, global_mean, learning_rate, regularization)


@export("msu_knn_predict")
def msu_knn_predict(
    xs: Int, ys: Int, offsets: Int, neighbors: Int, ratings: Int, sim: Int,
    stat1: Int, stat2: Int, y_bias: Int, estimates: Int, actual: Int,
    scores_work: Int, neighbors_work: Int, m: Int, n_x: Int, k: Int,
    min_k: Int, mode: Int, global_mean: Float64,
) abi("C"):
    knn_predict(ip(xs), ip(ys), ip(offsets), ip(neighbors), fp(ratings),
                fp(sim), fp(stat1), fp(stat2), fp(y_bias), fp(estimates),
                ip(actual), fp(scores_work), ip(neighbors_work), m, n_x, k,
                min_k, mode, global_mean)
