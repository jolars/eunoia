# NLM Port - Quick Reference Guide

## C Function Line Numbers Reference

Quick lookup for finding functions in `nlm/nlm.c`:

### Public API
- `optif9`: 2550-2614 (main public interface)
- `optif0`: 2506-2545 (simplified interface)

### Core Driver
- `optdrv`: 2157-2445 (main optimization loop)
- `optdrv_end`: 2132-2155 (cleanup)
- `opt_stop`: 1874-1962 (stopping criteria)

### Setup & Validation
- `optchk`: 1964-2067 (option validation)
- `dfault`: 2447-2504 (defaults)
- `prt_result`: 2069-2130 (output)

### Numerical Differentiation
- `fdhess`: 48-114 (finite difference Hessian)
- `fstofd`: 1558-1636 (forward difference)
- `fstocd`: 1638-1674 (central difference)
- `sndofd`: 1676-1749 (second-order forward diff)
- `grdchk`: 1751-1793 (gradient check)
- `heschk`: 1795-1872 (Hessian check)

### Linear Algebra - Cholesky
- `choldc`: 240-317 (Cholesky with tolerance)
- `chlhsn`: 1355-1529 (perturb and decompose)
- `lltslv`: 212-238 (solve L*L^T system)

### Linear Algebra - Matrix-Vector
- `mvmltl`: 130-156 (lower triangular multiply)
- `mvmltu`: 158-179 (upper triangular multiply)
- `mvmlts`: 181-210 (symmetric multiply)

### Linear Algebra - QR
- `qraux1`: 319-341 (QR auxiliary 1)
- `qraux2`: 343-377 (QR auxiliary 2)
- `qrupdt`: 379-439 (QR update)

### Optimization Methods
- `lnsrch`: 611-835 (line search)
- `dogdrv`: 837-1041 (dogleg method)
- `hookdrv`: 1043-1141 (More-Hebdon)

### Hessian Updates
- `secfac`: 1236-1353 (secant factored)
- `secunf`: 1143-1234 (secant unfactored)
- `tregup`: 441-609 (trust region update)
- `hsnint`: 1531-1556 (Hessian init)

### Dummy Functions (Not Needed)
- `d1fcn_dum`: 117-121 (use Option<GradientFn>)
- `d2fcn_dum`: 123-128 (use Option<HessianFn>)

## Key Data Structures

### C Code Storage Conventions
- **Column-major**: Matrices stored column-by-column
- **nr**: Number of rows (row dimension for indexing)
- **n**: Problem dimension (number of variables)
- **Triangular storage**: Only lower/upper triangle stored

### Rust Equivalents
```rust
// C: double *a (nr x n matrix, column-major)
// Rust: DMatrix<f64> (nalgebra handles storage)

// C: double *x (vector of length n)
// Rust: DVector<f64>

// C: fcn_p fcn (function pointer)
// Rust: &dyn Fn(&DVector<f64>) -> f64

// C: void *state (extra data)
// Rust: Generic parameter or trait object
```

## Important Algorithm Details

### Trust Region Methods
- **Method 1**: Line search (simplest)
- **Method 2**: Double dogleg (moderate)
- **Method 3**: More-Hebdon (most sophisticated)

### Stopping Codes (opt_stop returns)
- **0**: Continue iteration
- **1**: X-convergence (step size small)
- **2**: Gradient small
- **3**: Function change small
- **4**: Iteration limit
- **5**: Maximum step taken 5 times
- **6**: No satisfactory step found

### Workspace Arrays (in optdrv)
- `udiag[n]`: Diagonal of Hessian
- `g[n]`: Gradient at current iterate
- `p[n]`: Step vector
- `sx[n]`: Scaling factors
- `wrk0[n]`, `wrk1[n]`, `wrk2[n]`, `wrk3[n]`: General workspace

## Testing Commands

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_choldc

# Run with output
cargo test -- --nocapture

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Check formatting
cargo fmt --check

# Build documentation
cargo doc --no-deps --open
```

## Debugging Tips

1. **Compare with C**: Add print statements in both C and Rust versions
2. **Check matrix indexing**: nalgebra uses (row, col), C uses column-major
3. **Verify tolerances**: Machine epsilon values should match
4. **Trace iterations**: Print x, f, g at each iteration
5. **Check workspace**: Ensure workspace arrays are correct size

## Common Translation Patterns

### Array Indexing
```c
// C: Column-major, 0-indexed
a[i + j * nr]  // Element (i, j)

// Rust: Use nalgebra
a[(i, j)]  // Element (i, j)
```

### In-place Operations
```c
// C: Modify in place
for (i = 0; i < n; i++)
    x[i] = xpls[i];

// Rust: Use iterators or clone
x.copy_from(&xpls);
// or
x.iter_mut().zip(xpls.iter()).for_each(|(x, xp)| *x = *xp);
```

### Error Handling
```c
// C: Return error code, modify msg
if (error_condition) {
    *msg = -1;
    return;
}

// Rust: Return Result
if error_condition {
    return Err(NlmError::SomeError("description".into()));
}
```

### Optional Functions
```c
// C: Use dummy function pointers
if (!iagflg) {
    // Use numerical gradient
}

// Rust: Use Option
if let Some(grad_fn) = gradient {
    // Use analytic gradient
} else {
    // Use numerical gradient
}
```

## References

- **Book**: Dennis & Schnabel (1983) "Numerical Methods for Unconstrained Optimization and Nonlinear Equations"
- **R source**: `src/library/stats/src/optimize.c` (calls nlm)
- **Original Fortran**: `uncmin.f` (before f2c translation)
- **Documentation**: `?nlm` in R

## Contact / Questions

If you encounter issues or need clarification:
1. Check the C source code at the line numbers referenced
2. Look up the algorithm in Dennis & Schnabel book
3. Compare with R's behavior using `nlm()` function
4. Review the instructions in `.github/instructions/nlm.instructions.md`
