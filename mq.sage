import os, subprocess, tempfile, re
from itertools import product as prd
import argparse

set_random_seed()

def filter(n, m, k, l, b):
    B = sum(b)
    if n - (m - B - k) < (m - B - k) * (B + l):
        return 0
    if n - m < sum(b[j] * (m - sum(b[i] for i in range(0, j + 1)) - k) for j in range(len(b))):
        return 1
    return None

# ── M4GB configuration ────────────────────────────────────────────────────────
# Point this to the root of a cloned https://github.com/cr-marcstevens/m4gb
M4GB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'm4gb')
_m4gb_bins = {}  # cache: (q, n_vars) -> binary path

def _m4gb_binary(q, n_vars):
    """Return path to a compiled M4GB binary that handles n_vars variables over GF(q).
    Reuses any existing binary with MAXVARS >= n_vars before compiling a new one."""
    if (q, n_vars) in _m4gb_bins:
        return _m4gb_bins[(q, n_vars)]
    bin_dir = os.path.join(M4GB_DIR, 'bin')
    if os.path.isdir(bin_dir):
        for fname in sorted(os.listdir(bin_dir)):
            m = re.match(rf'solver_m4gb_n(\d+)_gf{q}$', fname)
            if m and int(m.group(1)) >= n_vars:
                path = os.path.join(bin_dir, fname)
                _m4gb_bins[(q, n_vars)] = path
                return path
    path = os.path.join(M4GB_DIR, f'bin/solver_m4gb_n{n_vars}_gf{q}')
    subprocess.run(
        ['make', f'MAXVARS={n_vars}', f'FIELDSIZE={q}', 'm4gb'],
        cwd=M4GB_DIR, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _m4gb_bins[(q, n_vars)] = path
    return path

def _poly_to_m4gb(p, var_names):
    """Serialize a SageMath polynomial to an M4GB expression string.
    Powers are expanded as repeated products (x^2 -> x*x) to avoid
    any parser ambiguity in M4GB's default format."""
    if p == 0:
        return None
    terms = []
    for mon, c in zip(p.monomials(), p.coefficients()):
        exps = mon.exponents()[0]
        factors = []
        for i, e in enumerate(exps):
            factors.extend([var_names[i]] * int(e))
        c_int = int(c)
        mon_str = '*'.join(factors)
        if not mon_str:
            terms.append(str(c_int))
        elif c_int == 1:
            terms.append(mon_str)
        else:
            terms.append(f'{c_int}*{mon_str}')
    return ' + '.join(terms) if terms else None

def _parse_m4gb_solution(F, var_names, text):
    """Parse M4GB output into a dict {var_name: field_element}, or None if absent."""
    sol = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('$'):
            continue
        for var in var_names:
            # fullmatch avoids prefix collision: 'x1' won't match 'x10 + 5'
            m = re.fullmatch(rf'{re.escape(var)}(\s*[+-]\s*\d+)?', line)
            if m:
                g = m.group(1)
                if g is None:
                    sol[var] = F.zero()
                else:
                    g = g.strip()
                    if g.startswith('+'):
                        sol[var] = F(-int(g[1:].strip()))
                    else:
                        sol[var] = F(int(g[1:].strip()))
                break
    return sol if len(sol) == len(var_names) else None

def _m4gb_gb_variety(F, var_names, text):
    """Parse M4GB output as a Groebner basis and enumerate all solutions via Sage.

    Used when _parse_m4gb_solution fails because the output contains non-linear
    polynomials (i.e. the ideal has multiple solutions).  Returns a list of
    solution dicts {var_name: field_element}."""
    R = PolynomialRing(F, var_names, order='degrevlex')
    locs = {v: R.gen(i) for i, v in enumerate(var_names)}
    gb_polys = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('$'):
            continue
        try:
            gb_polys.append(R(sage_eval(line, locals=locs)))
        except Exception:
            return []
    if not gb_polys:
        return []
    q_ = int(F.cardinality())
    field_eqs = [xi**q_ - xi for xi in R.gens()]
    I = R.ideal(gb_polys + field_eqs)
    try:
        if R.ngens() == 1:
            gb = I.groebner_basis()
            if not gb or gb[0] == R.one():
                return []
            f = gb[0]
            roots = list(F) if f == R.zero() else [r for r, _ in f.roots()]
            return [{var_names[0]: root} for root in roots]
        return [{vn: sol[R.gen(i)] for i, vn in enumerate(var_names)}
                for sol in I.variety()]
    except Exception:
        return []


def m4gb_solve(F, var_names, polynomials):
    """Solve a polynomial system over F using M4GB.

    Returns a list of solution dicts {var: value}, or None if M4GB is
    unavailable (caller should fall back to a different solver)."""
    q = int(F.cardinality())
    n = len(var_names)
    if n == 0:
        return [{}]
    try:
        binary = _m4gb_binary(q, n)
    except Exception:
        return None  # signal: M4GB not available

    poly_strs = []
    for p in polynomials:
        s = _poly_to_m4gb(p, var_names)
        if s is not None:
            poly_strs.append(s)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as tf:
        tf.write(f'$fieldsize {q}\n')
        tf.write(f'$vars {" ".join(var_names)}\n')
        for s in poly_strs:
            tf.write(s + '\n')
        infile = tf.name
    outfile = infile[:-3] + '.out'

    try:
        r = subprocess.run(
            [binary, '--solve', '--loglevel', '0', '-i', infile, '-o', outfile],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0 or not os.path.isfile(outfile):
            return []
        text = open(outfile).read()
    finally:
        for path in [infile, outfile]:
            if os.path.isfile(path):
                os.unlink(path)

    sol = _parse_m4gb_solution(F, var_names, text)
    if sol is not None:
        return [sol]
    # Non-linear GB output -> enumerate all solutions via Sage variety
    return _m4gb_gb_variety(F, var_names, text)

# ── MQ utilities ──────────────────────────────────────────────────────────────

def uptriag(M):
    for i in range(M.nrows()):
        for j in range(i):
            M[j, i] = M[i, j] + M[j, i]
            M[i, j] = 0
    return M

def generate_mq(FF, n, m):
    return [uptriag(Matrix(FF, [vector(FF.random_element() for _ in range(n)) for _ in range(n)])) for _ in range(m)]

def create_variables(F, n, v=tuple()):
    var_names = [f"x{i}" for i in range(n)]
    R = PolynomialRing(F, var_names, order='degrevlex')
    return R, vector(R, R.gens() + v)

def matrices_to_poly(vars_vec, matrices, linear, target, verbose=False):
    polynomials = [(vars_vec * M * vars_vec) - target[i] for i, M in enumerate(matrices)]
    if linear is not None:
        for row in linear:
            l_i = row * vars_vec
            if verbose:
                print(f'Linear: {l_i}')
            polynomials.append(l_i)
    return polynomials

def complete(FF, vecs, dim):
    basis = list(vecs)
    while len(basis) < dim:
        candidate = vector([FF.random_element() for _ in range(dim)])
        if rank(Matrix(FF, basis + [candidate])) > rank(Matrix(FF, basis)):
            basis.append(candidate)
    return basis

def _sage_gb_solve(F, R, n, q, n_eqs, x, polynomials):
    """Sage Groebner-basis fallback for solve_mq_under."""
    poly_sys = polynomials + [product(1 - x[i]**(q-1) for i in range(n))]
    I = R.ideal(poly_sys)
    gb = I.groebner_basis()
    if gb == [R.one()]:
        return []
    # Zero-dimensional ideal: use variety
    if len(gb) > 1 or gb[0].nvariables() > 1:
        q_ = F.cardinality()
        field_eqs = [xi**q_ - xi for xi in R.gens()]
        return R.ideal(gb + field_eqs).variety()
    # Single univariate polynomial
    if gb[0] == 0:
        return [(e, 1) for e in F] if n_eqs > 1 else []
    return gb[0].roots()

def solve_mq_under(n, m, q, matrices, b=None, linear=None, verbose=False,
                   independent=[], max_random=2000):
    """Solve an MQ system over GF(q), yielding non-zero solutions that are
    linearly independent from `independent`.

    Uses M4GB as the primary solver (with random variable-fixing to make the
    system square) and falls back to Sage's Groebner basis if M4GB is absent.
    For purely linear systems (no quadratic terms) uses right_kernel directly.

    n, m      : ambient dimension and nominal equation count (m used for b)
    matrices  : list of upper-triangular matrices defining the quadratic forms
    linear    : matrix of linear constraints (or None)
    max_random: cap on random trials when exhaustive enumeration is too large
    """
    F.<w> = GF(q)
    zero_vec = zero_vector(F, n)

    if b is None:
        b = zero_vector(F, m)
    else:
        b = vector(F, b)

    def _ok(v):
        return v != zero_vec and (
            not independent
            or rank(Matrix(F, independent)) < rank(Matrix(F, independent + [v]))
        )

    # ── Purely linear sub-problem: skip the MQ machinery entirely ────────────
    if len(matrices) == 0:
        if linear is None:
            for i in range(n):
                e = zero_vector(F, n); e[i] = F.one()
                if _ok(e):
                    yield e
        else:
            for sol in linear.right_kernel():
                sol_vec = vector(F, sol)
                if _ok(sol_vec):
                    yield sol_vec
        return

    # ── Fix exactly enough variables to make the system square ───────────────
    n_lin  = 0 if linear is None else linear.nrows()
    n_eqs  = len(matrices) + n_lin
    n_fix  = max(0, n - n_eqs)
    n_free = n - n_fix
    var_names = [f'x{i}' for i in range(n_free)]

    # Check M4GB availability once before the loop
    use_m4gb = True
    try:
        _m4gb_binary(q, max(n_free, 1))
    except Exception:
        use_m4gb = False

    # Exhaustive enumeration for small spaces, random sampling for large ones
    if Integer(q)**n_fix <= max_random:
        assignments = prd(F, repeat=n_fix)
    else:
        assignments = (
            tuple(F.random_element() for _ in range(n_fix))
            for _ in range(max_random)
        )

    for v in assignments:
        R, x = create_variables(F, n_free, v)
        polynomials = matrices_to_poly(x, matrices, linear, b)

        if n_free == 0:
            if all(p == 0 for p in polynomials):
                sol_vec = vector(F, list(v))
                if _ok(sol_vec):
                    yield sol_vec
            continue

        if use_m4gb:
            solutions = m4gb_solve(F, var_names, polynomials)
            if solutions is None:
                use_m4gb = False     # permanent fallback for this call
            else:
                for sol_dict in solutions:
                    sol_vec = vector(F, [sol_dict[vn] for vn in var_names] + list(v))
                    if _ok(sol_vec):
                        yield sol_vec
                continue             # don't fall through to Sage

        # ── Sage Groebner-basis fallback ──────────────────────────────────────
        sage_sols = _sage_gb_solve(F, R, n, q, n_eqs, x, polynomials)
        for sol in sage_sols:
            if isinstance(sol, dict):
                sol_vec = vector(F, [sol[xi] for xi in R.gens()] + list(v))
            elif isinstance(sol, tuple):
                sol_vec = vector(F, [sol[0]] + list(v))
            else:
                continue
            if _ok(sol_vec):
                yield sol_vec

# ── Verification ──────────────────────────────────────────────────────────────

def verify_solutions(solutions, matrices, b, q, linear=None):
    F.<w> = GF(q)
    n = matrices[0].nrows()
    b = vector(F, b)
    zero_vec = zero_vector(F, n)
    print("\n[*] Verifying solutions...")
    for idx, sol in enumerate(solutions):
        if linear is not None:
            is_linear = (zero_vector(F, linear.nrows()) == linear * sol)
        else:
            is_linear = True
        result = vector(F, [(sol * M * sol) for M in matrices])
        is_valid   = (result == b)
        is_nonzero = (sol != zero_vec)
        status = "✓ Valid" if (is_valid and is_nonzero and is_linear) else "✗ Invalid"
        print(f"    Solution {idx}: x = {list(sol)}")
        print(f"               F(x) = {list(result)}")
        print(f"               b    = {list(b)}")
        print(f"               {status}")

# ── Pseudo-Oil Basis computation ─────────────────────────────────────────────────────────

def orthogonality_O0(FF, n, q, k, l, bs, mqmap, basis, verbose=False):
    linears = Matrix(FF, [0] * n)
    for o in basis:
        linear = Matrix(FF, [o * (P + P.T) for P in mqmap[:sum(bs) + l]])
        linears = linears.stack(linear)
    if verbose:
        print(f'Linear from basis: {linears}')
        print(f'# Equations: {sum(bs) + l + linears.nrows()}')
        print(f'# Variables: {n}')
        print(f'Computing o{len(basis) + 1}...')
    return linears[1:]

def compute_O0(FF, mqmap, k, l, bs, verbose=False):
    n = mqmap[0].ncols()
    m = len(mqmap)
    q = int(FF.cardinality())
    basis = []
    dimO0 = m - k - sum(bs)
    while len(basis) < dimO0:
        print(f'Searching for o{len(basis) + 1}...')
        linears = orthogonality_O0(FF, n, q, k, l, bs, mqmap, basis, verbose=verbose)
        if sum(bs) + l + linears.nrows() <= n:
            sols = solve_mq_under(n, m, q, mqmap[:sum(bs) + l], linear=linears,
                                  verbose=verbose, independent=basis)
        else:
            raise NotImplementedError("Overdetermined case not implemented.")
        while True:
            try:
                sol = next(sols)
            except StopIteration:
                raise Exception('No solution found.')
            if not basis or rank(Matrix(FF, basis)) < rank(Matrix(FF, basis + [sol])):
                basis.append(sol)
                break
        if verbose:
            verify_solutions(basis, mqmap[:sum(bs) + l], zero_vector(FF, sum(bs) + l), q,
                             linear=linears)
            print(f'Basis length: {len(basis)}')
    return basis

def orthogonality_Otilde(FF, n, q, k, l, bs, mqmap, basis, verbose=False):
    m = len(mqmap)
    p = len(bs)
    dim0 = m - k - sum(bs)
    # assert len(basis) == m - k - bs[-1]
    linears = Matrix(FF, [0] * n)
    # O0 must be bilinear-orthogonal to the outermost bs[0] matrices
    for o in basis[:dim0]:
        linear = Matrix(FF, [o * (P + P.T) for P in mqmap[sum(bs[1:]):sum(bs)]])
        linears = linears.stack(linear)
    # Vectors up through block (p-1-i) must be bilinear-orthogonal to the
    # bs[p-i-1] matrices immediately inside those blocks
    for i in range(p - 1):
        for o in basis[:dim0 + sum(bs[:p - i - 1])]:
            linear = Matrix(FF, [o * (P + P.T) for P in mqmap[sum(bs[-i:]) if i > 0 else 0 : sum(bs[-(i+1):])]])
            linears = linears.stack(linear)
    if verbose:
        print(f'Linear from basis: {linears}')
        print(f'# Equations: {sum(bs) + l + linears.nrows()}')
        print(f'# Variables: {n}')
        print(f'Computing o{len(basis) + 1}...')
    return linears[1:]

def compute_Otilde(FF, mqmap, k, l, bs, ois, verbose=False):
    n = mqmap[0].ncols()
    m = len(mqmap)
    q = int(FF.cardinality())
    basis = list(ois)
    length = len(basis)
    Otilde = []
    while len(basis) < m:
        print(f'Computing otilde{len(basis) - length + 1}')
        linears = orthogonality_Otilde(FF, n, q, k, l, bs, mqmap, basis, verbose=verbose)
        if verbose:
            print(f'Computing o{len(basis) + 1}...')
            print(f'# Equations: {linears.nrows()}')
            print(f'# Variables: {n - len(basis)}')
        for sol in linears.right_kernel().basis():
            if rank(Matrix(FF, basis)) < rank(Matrix(FF, basis + [sol])):
                basis.append(sol)
                Otilde.append(sol)
                break
        if verbose:
            verify_solutions(basis, mqmap[:sum(bs)], zero_vector(FF, sum(bs)), q,
                             linear=linears)
            print(f'Basis length: {len(basis)}')
    return Otilde

def orthogonality_Oi(FF, n, q, k, l, bs, mqmap, basis, verbose=False):
    m = len(mqmap)
    i = len(basis)
    assert i >= m - sum(bs) - k
    s = i - (m - sum(bs) - k)
    for j in range(len(bs)):
        if s < sum(bs[:j+1]):
            break
    block = j
    if verbose:
        print(f'i: {i}, s: {s}, block: {block}')
    linears = Matrix(FF, [0] * n)
    for o in basis[:m - sum(bs) - k]:
        linear = Matrix(FF, [o * (P + P.T) for P in mqmap[:sum(bs)]])
        linears = linears.stack(linear)
    for b in range(block):
        for o in basis[m - sum(bs) - k + sum(bs[:b]): m - sum(bs) - k + sum(bs[:b+1])]:
            linear = Matrix(FF, [o * (P + P.T) for P in mqmap[:sum(bs[b+1:])]])
            linears = linears.stack(linear)
    sub = sum(bs[block+1:])
    if sub > 0:
        for o in basis[m - k - sum(bs) + sum(bs[:block]):i]:
            linear = Matrix(FF, [o * (P + P.T) for P in mqmap[:sub]])
            linears = linears.stack(linear)
    return linears[1:]

def compute_Oi(FF, mqmap, k, l, bs, O0, verbose=False):
    n = mqmap[0].ncols()
    m = len(mqmap)
    q = int(FF.cardinality())
    basis = list(O0)
    for i in range(len(bs) - 1):
        for j in range(bs[i]):
            linears = orthogonality_Oi(FF, n, q, k, l, bs, mqmap, basis, verbose=verbose)
            n_eqs = len(mqmap[:sum(bs[i+1:])]) + linears.nrows()
            if verbose:
                print(f'Block {i}, step {j}: {n_eqs} equations, {n} variables')
            if n_eqs <= n:
                sols = solve_mq_under(n, m, q, mqmap[:sum(bs[i+1:])], linear=linears,
                                      verbose=verbose, independent=basis)
            else:
                raise NotImplementedError("Overdetermined case not implemented.")
            while True:
                try:
                    sol = next(sols)
                except StopIteration:
                    raise Exception('No solution found.')
                if rank(Matrix(FF, basis)) < rank(Matrix(FF, basis + [sol])):
                    basis.append(sol)
                    break
    return basis

# ── Structured solver on the m×m Ptilde system ───────────────────────────────

def solve_Ptilde(Ptilde_mm, k, l, bs, q, target=None, verbose=False):
    """Find y ∈ GF(q)^m with y^T Ptilde_mm[i] y = target[i] for all i,
    exploiting the block-triangular structure produced by the change of
    coordinates.  target defaults to the zero vector.

    Variable layout  (m = len(Ptilde_mm), p = len(bs), dim0 = m-k-sum(bs)):
      y[0 .. dim0-1]              free variables
      y[dim0 .. dim0+b0-1]        block-0 variables (b0 = bs[0])
      ...
      y[m-k-bp .. m-k-1]          block-(p-1) variables  (bp = bs[-1], Otilde block)
      y[m-k .. m-1]               k enumerated variables

    Equation layout:
      Ptilde_mm[0 .. bp-1]                 innermost block (only last bp+k vars appear)
      Ptilde_mm[bp .. bp+b_{p-2}-1]        next block
      ...
      Ptilde_mm[sum(bs) .. m-1]            remaining m-sum(bs) equations over free vars
                                            (l of these are linear)

    Algorithm: enumerate k vars -> DFS solving each block from innermost to
    outermost -> solve the remaining free-variable system with M4GB.
    Yields solution vectors in GF(q)^m.
    """
    m = len(Ptilde_mm)
    p = len(bs)
    F.<w> = GF(q)
    dim0 = m - k - sum(bs)
    tgt = vector(F, m) if target is None else vector(F, target)

    def block_var_range(j):
        s = dim0 + sum(bs[:j])
        return list(range(s, s + bs[j]))

    def block_eq_range(j):
        s = sum(bs[j + 1:])
        return list(range(s, s + bs[j]))

    k_range   = list(range(m - k, m))
    free_range = list(range(dim0))

    def build_polys(eq_indices, free_idx, fixed):
        """Build a polynomial system from the selected equation indices.

        free_idx  : variable indices treated as ring generators.
        fixed     : dict {var_idx: field_element} for already-known values.
        All other variable indices evaluate to 0 (structural zeros guaranteed
        by the block-triangular sparsity of Ptilde_mm).

        Returns (polys, var_names, R) where R is None when len(free_idx)==0.
        """
        nf = len(free_idx)
        var_names = [f'z{i}' for i in range(nf)]

        if nf == 0:
            y = {i: F.zero() for i in range(m)}
            y.update(fixed)
            polys = []
            for ei in eq_indices:
                M = Ptilde_mm[ei]
                val = F.zero()
                for i in range(m):
                    for j in range(i, m):
                        c = M[i, j]
                        if c:
                            val += c * y[i] * y[j]
                polys.append(val - tgt[ei])
            return polys, var_names, None

        R = PolynomialRing(F, var_names, order='degrevlex')
        zgens = R.gens()
        y = {i: R.zero() for i in range(m)}
        y.update({idx: R(val) for idx, val in fixed.items()})
        for pos, idx in enumerate(free_idx):
            y[idx] = zgens[pos]

        polys = []
        for ei in eq_indices:
            M = Ptilde_mm[ei]
            f = R.zero()
            for i in range(m):
                for j in range(i, m):
                    c = M[i, j]
                    if c:
                        f += R(int(c)) * y[i] * y[j]
            polys.append(f - R(int(tgt[ei])))
        return polys, var_names, R

    def solve_block(j, fixed):
        """Enumerate all solutions for block j, yielding extended fixed dicts."""
        var_idx = block_var_range(j)
        eq_idx  = block_eq_range(j)
        nv = bs[j]

        # Brute force for small blocks — guarantees all solutions for backtracking
        if Integer(q)**nv <= 2000:
            for vals in prd(F, repeat=nv):
                trial = dict(fixed)
                for idx, val in zip(var_idx, vals):
                    trial[idx] = val
                polys, _, _ = build_polys(eq_idx, [], trial)
                if all(v == F.zero() for v in polys):
                    yield trial
            return

        # M4GB for larger blocks (may return only one solution)
        polys, var_names, _ = build_polys(eq_idx, var_idx, fixed)
        sols = m4gb_solve(F, var_names, polys)
        if sols is None:
            for vals in prd(F, repeat=nv):
                trial = dict(fixed)
                for idx, val in zip(var_idx, vals):
                    trial[idx] = val
                polys_c, _, _ = build_polys(eq_idx, [], trial)
                if all(v == F.zero() for v in polys_c):
                    yield trial
            return

        for sol_dict in sols:
            new_fixed = dict(fixed)
            for pos, idx in enumerate(var_idx):
                new_fixed[idx] = sol_dict[f'z{pos}']
            yield new_fixed

    def solve_free(fixed):
        """Solve the free-variable subsystem (equations sum(bs)..m-1), yielding
        extended fixed dicts."""
        eq_idx = list(range(sum(bs), m))

        if not eq_idx:
            yield fixed
            return

        if not free_range:
            polys, _, _ = build_polys(eq_idx, [], fixed)
            if all(v == F.zero() for v in polys):
                yield fixed
            return

        polys, var_names, R = build_polys(eq_idx, free_range, fixed)

        sols = m4gb_solve(F, var_names, polys)
        if sols is not None:
            for sol_dict in sols:
                new_fixed = dict(fixed)
                for pos, idx in enumerate(free_range):
                    new_fixed[idx] = sol_dict[f'z{pos}']
                yield new_fixed
            return

        # Sage Groebner-basis fallback
        field_eqs = [xi**q - xi for xi in R.gens()]
        I = R.ideal(polys + field_eqs)
        gb = I.groebner_basis()
        if gb == [R.one()]:
            return
        if R.ngens() == 1:
            # Univariate ring: Ideal_1poly_field has no .variety(); use .roots()
            f = gb[0]
            roots = list(F) if f == R.zero() else [r for r, _ in f.roots()]
            for root in roots:
                new_fixed = dict(fixed)
                new_fixed[free_range[0]] = root
                yield new_fixed
        else:
            for sol in I.variety():
                new_fixed = dict(fixed)
                for pos, idx in enumerate(free_range):
                    new_fixed[idx] = sol[R.gen(pos)]
                yield new_fixed

    def dfs(j, fixed):
        if j < 0:
            yield from solve_free(fixed)
            return
        for new_fixed in solve_block(j, fixed):
            yield from dfs(j - 1, new_fixed)

    for k_vals in prd(F, repeat=k):
        fixed0 = {k_range[i]: k_vals[i] for i in range(k)}
        if verbose:
            print(f'[solve_Ptilde] trying k-assignment {list(k_vals)}')
        for complete_fixed in dfs(p - 1, fixed0):
            sol = vector(F, m)
            for idx, val in complete_fixed.items():
                sol[idx] = val
            yield sol

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, required=True, help="variables")
    p.add_argument("-m", type=int, required=True, help="equations")
    p.add_argument("-q", type=int, required=True, help="")
    p.add_argument("-k", type=int, required=True, help="")
    p.add_argument("-l", type=int, required=True, help="")
    p.add_argument("-bs", nargs='+', type=int, help="Guessed")
    return p.parse_args()


def main():
    args = parse_args()
    n, m, q, k, l, bs = args.n, args.m, args.q, args.k, args.l, args.bs
    FF.<w> = GF(q)

    if filter(n, m, k, l, bs) is not None:
        print(f'The parameters do not satisfy the constraints.')
        exit()

    P = generate_mq(FF, n, m)
    t = vector(FF, [FF.random_element() for _ in range(m)])
    print(f'Target t = {list(t)}')
    tries = 0
    while True:
        tries += 1
        while True:
            random_mat = random_matrix(FF, n)
            if random_mat.is_invertible():
                break
        P = [random_mat.T * Pi * random_mat for Pi in P]
        try:
            print(f'Centrifugating O0...')
            O0     = compute_O0(FF, P, k, l, bs, verbose=False)
            print(f'Centrifugating Ois...')
            basis  = compute_Oi(FF, P, k, l, bs, O0)
            print(f'Centrifugating Otilde...')
            Otilde = compute_Otilde(FF, P, k, l, bs, basis, verbose=False)
            S      = Matrix(FF, complete(FF, basis + Otilde, n)).T

            Ptilde_full = [uptriag(S.T * Pi * S) for Pi in P]
            Ptilde_mm   = [M[:m, :m] for M in Ptilde_full]

            # Step 3: solve Ptilde(y) = t in GF(q)^m
            print(f'Solving...')
            y = next(solve_Ptilde(Ptilde_mm, k, l, bs, q, target=t, verbose=False), None)
            if y is None:
                print('  no solution found, rerandomising...')
                continue

            # Step 4: pad y with zeros to length n
            x_padded = vector(FF, list(y) + [FF.zero()] * (n - m))

            # Step 5: lift back to original coordinates  x = S * x_padded
            x = S * x_padded

            # Verify P(x) = t
            result = vector(FF, [x * Pi * x for Pi in P])
            ok = (result == t)
            print(f'  P(x) = t : {"OK" if ok else "FAIL"}')
            if ok:
                break
            if not ok:
                print(f'    P(x) = {list(result)}')
                print(f'    t    = {list(t)}')
        except Exception as e:
            print(f'  ERROR: {e}')
            raise

    print(f'Tries: {tries}')

if __name__ == "__main__":
    main()