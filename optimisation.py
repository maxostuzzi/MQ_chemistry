#!/usr/bin/env python3
"""
Brute-force search for k,a,b that satisfy:
  0 <= k <= m
  0 <= a <= m-k
  0 <= b <= m-k-a
and
  CONSTR1(k,a,b) <= n
  CONSTR2(k,a,b) <= n
  CONSTR3(k,a,b) <= n

Minimize objective_f(k,a,b, n, m, q).

This version parallelizes the outer k-loop using multiprocessing.Pool
and uses a shared Manager().dict() cache so that expensive
MQEstimator.estimate() calls are computed once and reused across
worker processes.

Notes / caveats:
- The shared cache (Manager.dict) reduces duplicate MQEstimator calls
  across processes, but it still has IPC overhead. For very small
  problems the overhead may outweigh the parallel speedup.
- If MQEstimator is not thread/process-safe for some inputs, you may
  want to run with --workers 1 (serial) instead.
- The script falls back to a single-process lru_cache when running
  without multiprocessing.
"""

from __future__ import annotations
import argparse
import math
from typing import Tuple, List
import sys
from functools import lru_cache
from cryptographic_estimators.MQEstimator import MQEstimator

# multiprocessing
from multiprocessing import Pool, Manager

# Optional progress bar
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    def tqdm(iterable, **kw):
        return iterable
    HAVE_TQDM = False

def CONSTR1(k: int, l: int, a: int, b: int, m: int) -> float:
    """
    Return the left-hand side of constraint 1.
    Must be <= n to satisfy.
    Replace the example below with your real expression.
    """
    return (m - a - b - k) * (a + b + l + 1)


def CONSTR3(k: int, l: int, a: int, b: int, m: int) -> float:
    """Replace with your actual constraint 3 expression."""
    return a * (m - a - b - k) + b * (m - k - b) + m - k

def filter(n, m, k, l, b):
    B = sum(b)
    if n - (m - B - k) < (m - B - k) * (B + l):
        return False
    if n - m < sum(b[j] * (m - sum(b[i] for i in range(0, j + 1)) - k) for j in range(len(b))):
        return False
    return True

# -------------------------
# CACHING helpers
# We provide two modes for obtaining MQEstimator estimates:
#  - a process-local lru_cache (fast, used in serial mode)
#  - a shared Manager.dict based cache (used in multiprocessing mode)
# The objective_f will call mq_estimate_time() which selects the
# appropriate backend depending on whether a shared cache is available.
# -------------------------

# process-local cache (for serial runs)
@lru_cache(maxsize=None)
def mq_estimate_time_cached(q: int, m_: int, n_: int) -> float:
    if m_ <= 0:
        return 0.0
    if m_ in (1, 2) or n_ in (1, 2):
        return 1.0
    try:
        problem = MQEstimator(q=q, m=m_, n=n_)
        estimates = problem.estimate()
        times = [v['estimate']['time'] for v in estimates.values()]
        return float(min(times)) if times else math.inf
    except Exception as e:
        print(f"Warning: MQEstimator failed for (q={q}, m={m_}, n={n_}): {e}", file=sys.stderr)
        return math.inf

# shared cache globals to be initialized in worker processes
SHARED_CACHE = None
SHARED_LOCK = None


def init_worker(shared_cache, shared_lock):
    """Initializer for worker processes: store references to shared cache/lock."""
    global SHARED_CACHE, SHARED_LOCK
    SHARED_CACHE = shared_cache
    SHARED_LOCK = shared_lock


def mq_estimate_time_cached_shared(q: int, m_: int, n_: int) -> float:
    """Look up or compute the MQEstimator estimate using a Manager.dict()."""
    # same quick checks to avoid constructing MQEstimator unnecessarily
    if m_ <= 0:
        return 0.0
    if m_ in (1, 2) or n_ in (1, 2):
        return 1.0

    key = (q, m_, n_)
    # try fast local read
    val = SHARED_CACHE.get(key, None)
    if val is not None:
        return val

    # compute under lock to avoid duplicated work
    with SHARED_LOCK:
        val = SHARED_CACHE.get(key, None)
        if val is not None:
            return val
        try:
            problem = MQEstimator(q=q, m=m_, n=n_)
            estimates = problem.estimate()
            times = [v['estimate']['time'] for v in estimates.values()]
            out = float(min(times)) if times else math.inf
        except Exception as e:
            print(f"Warning: MQEstimator failed for (q={q}, m={m_}, n={n_}): {e}", file=sys.stderr)
            out = math.inf
        SHARED_CACHE[key] = out
        return out


def mq_estimate_time(q: int, m_: int, n_: int) -> float:
    """Public interface: use shared cache if available (multiprocessing), else use local cache."""
    if SHARED_CACHE is not None:
        return mq_estimate_time_cached_shared(q, m_, n_)
    return mq_estimate_time_cached(q, m_, n_)

def cost_quadratic(n):
    '''
    Returns [# field mult, # field add] for matrix vector multiplication u**tAu
    '''
    return [n * (n+1)/2 + n, n * (n-1)/2 + n - 1]

def cost_inner_prod(n):
    '''
    Returns [# of field multiplications, # of field additions] to perform an inner-product between two n-dim vectors.
    '''
    return [n, n - 1]

def cost_matrix_vector(n,m):
    return [n * m, n * (m - 1)]

def polynomial_factors(n,m,q,k,l,bs):
    cost = 0
    for i in range(len(bs)):
        B = sum(bs[:i]) + k
        cost += sum(cost_quadratic(B)) + sum(cost_matrix_vector(m - B, B)) + sum(cost_inner_prod(m))
    return cost


def objective_f(n,m,q,k,l,bs):
    costs = {}
    B = sum(bs)
    for b in bs:
        if b in [0,1,2]:
            costs[b] = 1    
        else:
            costs[b] = mq_estimate_time(q, b, b)
    if (B + l) in [0,1,2]:
        costs[B + l] = 1
    else:
        costs[B + l] = mq_estimate_time(q, B + l, B + l)
    Bs = [sum(bs[:i]) for i in range(len(bs))]
    for Bi in Bs:
        if Bi in [0,1,2]:
            costs[Bi] = 1
        else:
            costs[Bi] = mq_estimate_time(q, Bi, Bi)
    if ((m - B - k - l) in [0,1,2]) or ((m - B - l) in [0,1,2]):
        costs['over'] = 1
    else:
        costs['over'] = mq_estimate_time(q, m - B - l, m - B - k - l)
    return math.log2((m - B - k) * 2**costs[B+l] + sum((m - Bs[i] - k) * 2**costs[Bs[i]] for i in range(0,len(bs) - 1)) + q**k * (2**costs['over'] + sum(2**costs[b] for b in bs) + polynomial_factors(n,m,q,k,l,bs)))


def tuples_sum_leq(m_kl: int, p: int) -> Iterator[List[int]]:
    """Yield all lists of p nonnegative integers whose sum <= m.
       Example: list(tuples_sum_leq(2, 2)) -> [[0,0],[0,1],[0,2],[1,0],[1,1],[2,0]]
    """
    if p <= 0:
        # if user insists p==0 you get one empty tuple if m>=0
        if p == 0:
            yield []
        return

    def helper(remaining: int, slots: int) -> Iterator[List[int]]:
        if slots == 1:
            # only one slot left: it may be any value 0..remaining
            for v in range(remaining + 1):
                yield [v]
            return
        for first in range(remaining + 1):
            for tail in helper(remaining - first, slots - 1):
                yield [first] + tail

    yield from helper(m_kl, p)


def search_bruteforce_serial(n: int, m: int, q: int, p: int, tobeat: int,
                      return_all_best: bool = False,
                      verbose: bool = False) -> Tuple[float, List[Tuple[int,int,list]]]:
    """
    Serial search (no multiprocessing). Kept for --workers 1 or default behaviour.
    """
    best_value = math.inf
    best_solutions: List[Tuple[int,int, list]] = []

    # Outer loop over k
    for k in range(m + 1):
        max_l = m - k
        for l in range(max_l + 1):
            m_kl = max_l - l
            for bs in tuples_sum_leq(m_kl, p):
                if not filter(n, m, k, l, bs):
                    continue
                val = objective_f(n, m, q, k, l, bs)
                if val < best_value:
                    best_value = val
                    best_solutions = [(k, l, bs)]
                    if verbose:
                        print(f"New best {best_value} at (k,a,a',b)=({k},{l},{bs})", file=sys.stderr)
                elif val == best_value:
                    best_solutions.append((k, l, bs))
    return best_value, best_solutions

def worker_for_k(args_tuple):
    """Compute best solutions for a single k. Runs inside worker process.

    args_tuple = (k, n, m, q)
    Returns: (k, best_value_for_k, best_sols_for_k)
    """
    k, n, m, q, p = args_tuple
    best_value = math.inf
    best_solutions: List[Tuple[int,int,list]] = []

    max_l = m - k
    for l in range(max_l + 1):
        m_kl = max_l - l
        for bs in tuples_sum_leq(m_kl, p):
            if not filter(n, m, k, l, bs):
                continue
            val = objective_f(n, m, q, k, l, bs)
            if val < best_value:
                best_value = val
                best_solutions = [(k, l, bs)]
                print(f"New best {best_value} at (k,l,bs)=({k},{l},{bs})", file=sys.stderr)
            elif val == best_value:
                best_solutions.append((k, l, bs))
    return k, best_value, best_solutions


def search_bruteforce_parallel(n: int, m: int, q: int, p: int, tobeat: int, workers: int, verbose: bool = False) -> Tuple[float, List[Tuple[int,int,list]]]:
    """Parallel search by distributing k values across workers."""
    if workers <= 1:
        return search_bruteforce_serial(n, m, q, return_all_best=True, verbose=verbose)

    manager = Manager()
    shared_cache = manager.dict()
    shared_lock = manager.Lock()

    # prepare work items
    # ks = [(k, n, m, q, p) for k in range(30, math.ceil(tobeat/math.log2(q)))]
    ks = [(k, n, m, q, p) for k in range(10, math.ceil(tobeat/math.log2(q)))]
    # ks = [(k, n, m, q, p) for k in range(math.ceil(tobeat/math.log2(q)))]

    best_value = math.inf
    best_solutions: List[Tuple[int,int,list]] = []

    with Pool(processes=workers, initializer=init_worker, initargs=(shared_cache, shared_lock)) as pool:
        # imap_unordered returns results as they complete
        it = pool.imap_unordered(worker_for_k, ks)
        if HAVE_TQDM:
            it = tqdm(it, total=len(ks), desc="k tasks")
        for k_val, val, sols in it:
            if val == math.inf:
                continue
            if val < best_value:
                best_value = val
                best_solutions = sols.copy()
            elif val == best_value:
                best_solutions.extend(sols)

    return best_value, best_solutions


def parse_args():
    parser = argparse.ArgumentParser(description="Brute-force search for k,l,bs minimizing objective subject to constraints.")
    parser.add_argument("-n", type=int, required=True, help="Constraint RHS (n)")
    parser.add_argument("-m", type=int, required=True, help="Upper bound parameter (m)")
    parser.add_argument("-q", type=int, required=True)
    parser.add_argument("-p", type=int, required=True)
    parser.add_argument("-tobeat", type=int, required=True)
    parser.add_argument("--all-ties", action="store_true", help="Return all tied best solutions instead of just one")
    parser.add_argument("--verbose", action="store_true", help="Print progress/updating messages")
    parser.add_argument("-workers", type=int, default=1, help="Number of worker processes for parallel k-loop (default 1 = serial)")
    return parser.parse_args()


def main():
    args = parse_args()
    n, m, q, p, tobeat = args.n, args.m, args.q, args.p, args.tobeat

    if m < 0 or n < 0:
        print("n and m should be non-negative integers.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Running brute-force search with n={n}, m={m}, q={q}, workers={args.workers}", file=sys.stderr)

    if args.workers <= 1:
        best_value, best_sols = search_bruteforce_serial(n, m, q, p, tobeat, return_all_best=True, verbose=args.verbose)
    else:
        best_value, best_sols = search_bruteforce_parallel(n, m, q, p, tobeat, workers=args.workers, verbose=args.verbose)

    if best_value == math.inf:
        print("No feasible (k,l,a,b) found that satisfies all constraints.")
        return

    # By default, if user didn't ask for all ties, show one canonical best (e.g., smallest lexicographic)
    if not args.all_ties:
        # pick lexicographically smallest (k,a,b) among ties
        chosen = min(best_sols)
        print(f"Best objective value: {best_value}")
        print(f"Chosen (k,l,a,b): {chosen}")
        if len(best_sols) > 1:
            print(f"(There are {len(best_sols)} tied solutions; use --all-ties to list them all.)")
    else:
        print(f"Best objective value: {best_value}")
        print(f"Number of tied best solutions: {len(best_sols)}")
        for idx, (k,l,bs) in enumerate(sorted(best_sols), start=1):
            print(f"{idx}: (k={k}, l={l}, bs={bs})")

if __name__ == "__main__":
    main()
