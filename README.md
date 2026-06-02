# Pseudo-Oil Subspaces and the Geometry of Underdetermined MQ Solving

This is the discrete optimisation script for computing the optimal parameters $\ell$ and $b_i$ from the paper.
We make use of the Cryptogaphic Estimator library, which can be obtained from https://github.com/Crypto-TII/CryptographicEstimators .

## Optimisation Script

The optimisation script `optimisation.py` takes as inputs the integers $n,m,q,p$ which are the number of variables, number of equations, field size and number of blocks for an instance of the MQ problem.
Additionally, to reduce the search space, one can indicate by specifying the additional input `--tobeat` the bit complexity "to beat".
One can also specify the number of workers and whether one wants all the parameters with tied complexity to be printed.

For example:
`python3 optimisation.py --n 860 --m 78 --q 16 --p 8 --tobeat 155 --all-ties --workers 10`

Warning: the script checks all possible partitions in $p$ integers of $m - k$, for all possible $k$. Therefore, it is pretty slow. To verify the complexity claims, one shall use the following script.

## Evaluating the complexity

On input the integers $n,m,q,p, k, \ell$ and $b_i$, this script `eval_complexity_tree.py` returns the corresponding complexity. This should be used, for example, to verify that the parameter choices
- $(n, m, q, k, p, \ell, b_i) = (860, 78, 16, 6, 32, 14, [3, 3, 3, 3, 3, 3, 3, 3])$ for MAYO SL1
- $(n, m, q, k, p, \ell, b_i) = (840, 100, 7, 4, 54, 12 , [4, 4, 5, 5, 5])$ for QR-UOV SL1

satisfy the constraints of the optimisation problem and yield the bit complexities claimed in the paper.

For example:
`python3 eval_complexity_tree.py -n 860 -m 78 -q 16 -k 32 -l 14 -bs 3 3 3 3 3 3 3 3`

## Underdetermined Solver

On input the integers $n,m,q,p$ as above and additionally $k, \ell$ and $b_i$, this script `mq_solver.sage` generates a random MQ problem according to the inputs and solves it using the algorithm described in the paper. 

For example, on a small instance:
`sage mq_solver.sage -n 30 -m 13 -q 3 -k 6 -l 1 -bs 2 2`

## Toy Example Generator

The script `ex_gen.sage` works exactly as `mq_solver.sage`, but it is more verbose and it prints the corresponding MQ trees.

For example, on a small instance:
`sage ex_gen.sage -n 30 -m 13 -q 3 -k 6 -l 1 -bs 2 2`
