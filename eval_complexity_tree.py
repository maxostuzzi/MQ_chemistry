#!/usr/bin/env python3


from __future__ import annotations
import argparse
import math
from typing import Tuple, List
from cryptographic_estimators.MQEstimator import MQEstimator


def filter(n, m, k, l, b):
    B = sum(b)
    if n - (m - B - k) < (m - B - k) * (B + l):
        return 0
    if n - m < sum(b[j] * (m - sum(b[i] for i in range(0, j + 1)) - k) for j in range(len(b))):
        return 1
    return None

def mq_estimate(n_, m_, q):
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
        print(n_)
        print(m_)
        # print(f"Warning: MQEstimator failed for (q={q}, m={m_}, n={n_}): {e}", file=sys.stderr)
        return math.inf

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

def eval_complexity(n,m,q,k,l,bs):
    costs = {}
    B = sum(bs)
    for b in bs:
        if b in [0,1,2]:
            costs[b] = 1    
        else:
            costs[b] = mq_estimate(b, b, q)
    if (B + l) in [0,1,2]:
        costs[B + l] = 1
    else:
        costs[B + l] = mq_estimate(B + l, B + l, q)
    Bs = [sum(bs[i:]) for i in range(len(bs))]
    for Bi in Bs:
        if Bi in [0,1,2]:
            costs[Bi] = 1
        else:
            costs[Bi] = mq_estimate(Bi, Bi, q)
    if ((m - B - k - l) in [0,1,2]) or ((m - B - l) in [0,1,2]):
        costs['over'] = 1
    else:
        costs['over'] = mq_estimate(m - B - k - l, m - B - l, q)
    # return math.log2((m - B - k) * 2**costs[B+l] + sum((m - Bs[i] - k) * 2**costs[Bs[i]] for i in range(0,len(bs) - 1)) + q**k * (2**costs['over'] + sum(2**costs[b] for b in bs) + polynomial_factors(n,m,q,k,l,bs)))
    # Return Centrifugation, Guess, Square, Over, Polys 
    return math.log2((m - B - k) * 2**costs[B+l] + sum((bs[i]) * 2**costs[Bs[i+1]] for i in range(0,len(bs) - 2))), math.log2(q**k), math.log2(sum(2**costs[b] for b in bs)), math.log2(2**costs['over']), math.log2(polynomial_factors(n,m,q,k,l,bs)), math.log2((m - B - k) * 2**costs[B+l] + sum((bs[i]) * 2**costs[Bs[i+1]] for i in range(0,len(bs) - 2)) + q**k * (2**costs['over'] + sum(2**costs[b] for b in bs) + polynomial_factors(n,m,q,k,l,bs)))


def parse_args():
    p = argparse.ArgumentParser(description="Brute-force search for k,a,b minimizing objective subject to constraints.")
    p.add_argument("-n", type=int, required=True, help="Constraint RHS (n)")
    p.add_argument("-m", type=int, required=True, help="Upper bound parameter (m)")
    p.add_argument("-q", type=int, required=True, help="")
    p.add_argument("-k", type=int, required=True, help="")
    p.add_argument("-l", type=int, required=True, help="")
    p.add_argument("-bs", nargs='+', type=int, help="Guessed")
    return p.parse_args()


def main():
    args = parse_args()
    n, m, q, k, l, bs = args.n, args.m, args.q, args.k, args.l, args.bs
    
    B = sum(bs)

    if sum(bs)+l+k>m:
        print('Structurally not allowed.')
        exit()
    feasible = filter(n, m, k, l, bs)
    if feasible == 0:
        print('Too aggressive for last vector of O0.')
        exit()
    elif feasible == 1:
        print('Too aggressive for last vector of Otilde.')
        exit()
    else:
        complexity = [round(i,1) for i in eval_complexity(n,m,q,k,l,bs)]

        print('Inequalities Satisfied!\n')
        print(f'Centrifugation: {complexity[0]}\n')
        print(f'Guess: {complexity[1]}\n')
        print(f'Square: {complexity[2]}')
        print(f'Overdetermined: {complexity[3]}')
        print(f'Polynomial Costs: {complexity[4]}\n')
        print(f'Total Bit Complexity: {complexity[5]}')

if __name__ == "__main__":
    main()
