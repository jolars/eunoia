---
applyTo: "crates/eunoia-nlm/**/*.rs"
---

# NLM (Nonlinear Minimization) Algorithm - Complete Port Instructions

## Overview

This is a **complete and faithful port** of the Dennis-Schnabel optimization algorithm from R's `nlm()` function (C implementation in `nlm/nlm.c`) to Rust. **No simplifications or shortcuts are allowed**.

The original C code is 2,614 lines and contains approximately 30+ functions implementing:
- Multiple optimization methods (line search, dogleg, More-Hebdon)
- Trust region methods
- Numerical differentiation (forward and central differences)
- Gradient and Hessian checking
- Secant updates (BFGS-style)
- Cholesky decomposition with tolerance handling
- QR updates

## Reference Implementation

**Source**: `nlm/nlm.c` (2,614 lines)
**Algorithm**: Dennis & Schnabel (1983) "Numerical Methods for Unconstrained Optimization and Nonlinear Equations"

## Rust Crate Structure

The implementation should be organized in `crates/eunoia-nlm/src/` with the following module structure:

```
src/
├── lib.rs                  # Main API and public exports
├── types.rs                # Type definitions, error types, config structs
├── linalg/                 # Linear algebra operations
│   ├── mod.rs
│   ├── mvmult.rs          # Matrix-vector multiplication (mvmltl, mvmltu, mvmlts)
│   ├── cholesky.rs        # Cholesky decomposition (choldc, chlhsn, lltslv)
│   └── qr.rs              # QR updates (qrupdt, qraux1, qraux2)
├── differentiation/        # Numerical differentiation
│   ├── mod.rs
│   ├── gradient.rs        # Forward/central differences (fstofd, fstocd)
│   ├── hessian.rs         # Hessian computation (fdhess, sndofd)
│   └── checking.rs        # Gradient/Hessian validation (grdchk, heschk)
├── methods/                # Optimization methods
│   ├── mod.rs
│   ├── line_search.rs     # Line search (lnsrch)
│   ├── dogleg.rs          # Dogleg method (dogdrv)
│   └── hook.rs            # More-Hebdon method (hookdrv)
├── updates/                # Hessian updates
│   ├── mod.rs
│   ├── secant.rs          # Secant updates (secfac, secunf)
│   └── trust_region.rs    # Trust region updates (tregup)
├── initialization/         # Initial setup
│   ├── mod.rs
│   ├── hessian.rs         # Hessian initialization (hsnint)
│   └── defaults.rs        # Default values (dfault)
├── driver.rs               # Main optimization driver (optdrv, optdrv_end)
├── stopping.rs             # Stopping criteria (opt_stop)
├── validation.rs           # Option validation (optchk)
└── output.rs               # Result formatting (prt_result)
```

## Function Mapping (C → Rust)

### Public API Functions

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `optif9` | 2550-2614 | `lib.rs` | Main public interface with full control |
| `optif0` | 2506-2545 | `lib.rs` | Simplified interface (optional) |

### Core Driver

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `optdrv` | 2157-2445 | `driver.rs` | Main optimization driver loop |
| `optdrv_end` | 2132-2155 | `driver.rs` | Cleanup and final result preparation |
| `opt_stop` | 1874-1962 | `stopping.rs` | Check stopping criteria |

### Validation & Setup

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `optchk` | 1964-2067 | `validation.rs` | Validate and adjust options |
| `dfault` | 2447-2504 | `initialization/defaults.rs` | Set default parameters |

### Numerical Differentiation

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `fdhess` | 48-114 | `differentiation/hessian.rs` | Finite difference Hessian (A5.6.2) |
| `fstofd` | 1558-1636 | `differentiation/gradient.rs` | Forward difference gradient/Jacobian |
| `fstocd` | 1638-1674 | `differentiation/gradient.rs` | Central difference gradient |
| `sndofd` | 1676-1749 | `differentiation/hessian.rs` | Second-order forward difference Hessian |

### Validation of Derivatives

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `grdchk` | 1751-1793 | `differentiation/checking.rs` | Check analytic gradient |
| `heschk` | 1795-1872 | `differentiation/checking.rs` | Check analytic Hessian |

### Linear Algebra - Cholesky

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `choldc` | 240-317 | `linalg/cholesky.rs` | Cholesky decomp with tolerance |
| `chlhsn` | 1355-1529 | `linalg/cholesky.rs` | Perturb and decompose Hessian |
| `lltslv` | 212-238 | `linalg/cholesky.rs` | Solve with L*L^T factorization |

### Linear Algebra - Matrix-Vector Operations

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `mvmltl` | 130-156 | `linalg/mvmult.rs` | y = L*x (lower triangular) |
| `mvmltu` | 158-179 | `linalg/mvmult.rs` | y = L^T*x (upper triangular) |
| `mvmlts` | 181-210 | `linalg/mvmult.rs` | y = L^T*x (symmetric storage) |

### Linear Algebra - QR Updates

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `qrupdt` | 379-439 | `linalg/qr.rs` | QR update (rank-1 modification) |
| `qraux1` | 319-341 | `linalg/qr.rs` | QR auxiliary function 1 |
| `qraux2` | 343-377 | `linalg/qr.rs` | QR auxiliary function 2 |

### Optimization Methods

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `lnsrch` | 611-835 | `methods/line_search.rs` | Line search with backtracking |
| `dogdrv` | 837-1041 | `methods/dogleg.rs` | Double dogleg driver |
| `hookdrv` | 1043-1141 | `methods/hook.rs` | More-Hebdon (hook step) driver |

### Hessian Updates & Initialization

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `secfac` | 1236-1353 | `updates/secant.rs` | Secant update (factored) |
| `secunf` | 1143-1234 | `updates/secant.rs` | Secant update (unfactored) |
| `tregup` | 441-609 | `updates/trust_region.rs` | Trust region radius update |
| `hsnint` | 1531-1556 | `initialization/hessian.rs` | Initialize Hessian approximation |

### Helper/Dummy Functions

| C Function | Lines | Rust Module | Description |
|------------|-------|-------------|-------------|
| `d1fcn_dum` | 117-121 | Not needed | Dummy gradient (use Option type) |
| `d2fcn_dum` | 123-128 | Not needed | Dummy Hessian (use Option type) |
| `prt_result` | 2069-2130 | `output.rs` | Print iteration results (for debugging) |

## Implementation Guidelines

### Phase 1: Foundation (Linear Algebra)

Implement in this order:
1. **types.rs** - Define all types, error enums, config structs
2. **linalg/mvmult.rs** - Matrix-vector multiplications
3. **linalg/cholesky.rs** - Cholesky operations
4. **linalg/qr.rs** - QR updates

**Testing**: Each function must have unit tests comparing against known results or the original C implementation.

### Phase 2: Differentiation

5. **differentiation/gradient.rs** - Forward and central differences
6. **differentiation/hessian.rs** - Hessian approximations
7. **differentiation/checking.rs** - Gradient and Hessian validation

**Testing**: Test against simple functions with known derivatives (quadratic, Rosenbrock, etc.)

### Phase 3: Updates & Initialization

8. **initialization/defaults.rs** - Default parameter values
9. **initialization/hessian.rs** - Hessian initialization strategies
10. **updates/secant.rs** - BFGS-style secant updates
11. **updates/trust_region.rs** - Trust region radius updates

### Phase 4: Optimization Methods

12. **methods/line_search.rs** - Line search with Armijo condition
13. **methods/dogleg.rs** - Double dogleg method
14. **methods/hook.rs** - More-Hebdon method

**Testing**: Test each method in isolation with simple convex functions

### Phase 5: Driver & Integration

15. **stopping.rs** - Stopping criteria checks
16. **validation.rs** - Option validation and adjustment
17. **driver.rs** - Main optimization loop
18. **lib.rs** - Public API
19. **output.rs** - Result formatting

**Testing**: Integration tests with standard test problems (Rosenbrock, Powell, etc.)

## Code Standards

### Direct Translation Requirements

1. **Algorithm Fidelity**: Translate the C logic directly. Don't "improve" or simplify.
2. **Line-by-line Mapping**: Each C function should map to identifiable Rust code.
3. **Comment Preservation**: Keep original comments from C code, adding Rust-specific notes.
4. **Variable Names**: Use similar variable names where Rust conventions allow.

### Rust Idioms

1. **Error Handling**: Use `Result<T, NlmError>` instead of C error codes
2. **Memory Safety**: Use `nalgebra` types (no raw pointers)
3. **Trait Objects**: Use `dyn Fn` for function pointers
4. **Options**: Use `Option<T>` instead of null pointers or dummy functions

### Matrix Storage

The C code uses column-major storage (`nr` is row dimension for indexing).
Use `nalgebra::DMatrix` with appropriate indexing conversions.

### Testing Requirements

1. **Unit Tests**: Every function must have tests in the same file (`#[cfg(test)]`)
2. **Reference Values**: Compare against C implementation output where possible
3. **Edge Cases**: Test boundary conditions, singular matrices, etc.
4. **Integration Tests**: Full optimization runs in `tests/` directory

### Documentation Requirements

1. **Function Documentation**: Every function needs:
   - Algorithm description
   - Reference to Dennis & Schnabel section if applicable
   - Parameter descriptions
   - Return value description
   - C function mapping (e.g., "Port of `choldc` from nlm.c:240-317")

2. **Module Documentation**: Each module needs overview of contained functionality

## Key Algorithms to Implement Faithfully

### 1. Cholesky with Tolerance (choldc)

Lines 240-317 in nlm.c
- Must handle nearly-singular matrices
- Adds diagonal perturbation when needed
- Returns perturbation amount

### 2. Line Search (lnsrch)

Lines 611-835 in nlm.c
- Implements Goldstein-Armijo conditions
- Handles step size too large case
- Cubic/quadratic interpolation for backtracking

### 3. Dogleg Method (dogdrv)

Lines 837-1041 in nlm.c
- Combines Cauchy point and Newton direction
- Trust region subproblem solution
- Includes internal loop with parameter passing via workspace

### 4. More-Hebdon Method (hookdrv)

Lines 1043-1141 in nlm.c
- More sophisticated trust region method
- Solves: min m(p) s.t. ||D*p|| ≤ delta
- Hook step calculation with parameter search

### 5. Secant Updates (secfac, secunf)

Lines 1143-1353 in nlm.c
- BFGS-style Hessian approximation
- Factored (method 1,2) vs unfactored (method 3) forms
- Skipping logic when updates would be harmful

### 6. Trust Region Update (tregup)

Lines 441-609 in nlm.c
- Adjusts trust region radius based on actual vs predicted reduction
- Handles both increase and decrease cases

## Common Pitfalls to Avoid

1. **Don't simplify the Cholesky routine** - The tolerance handling is critical
2. **Don't skip the QR updates** - Used in secant updates for method 3
3. **Don't merge similar functions** - Keep secfac and secunf separate even if similar
4. **Don't skip parameter validation** - optchk does important adjustments
5. **Don't use simple gradient descent** - Implement the actual trust region methods
6. **Don't skip central differences** - Used as fallback when forward differences fail

## Testing Strategy

### Unit Test Examples

For each module, create tests like:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_choldc_positive_definite() {
        // Test with known positive definite matrix
        // Compare against expected decomposition
    }

    #[test]
    fn test_choldc_nearly_singular() {
        // Test with nearly singular matrix
        // Verify diagonal perturbation is added
    }
}
```

### Integration Test Problems

In `tests/integration_tests.rs`:

1. **Rosenbrock** (n=2): Classic test, banana-shaped valley
2. **Powell** (n=4): Tests ill-conditioning
3. **Wood** (n=4): Multiple local minima
4. **Helical Valley** (n=3): Narrow curved valley
5. **Simple quadratic** (various n): Should converge in 1-2 iterations

Each test should verify:
- Final function value
- Final gradient norm
- Termination code
- Number of iterations

## Performance Considerations

1. **Workspace Management**: Like C code, pre-allocate workspace vectors
2. **Matrix Operations**: Use nalgebra's optimized BLAS when possible
3. **Function Evaluations**: Count and report (expensive operations)
4. **Memory Allocations**: Minimize in inner loops

## Validation Against Original

Once complete, validate by:

1. Running same test problems in R's `nlm()` and Rust implementation
2. Comparing iteration counts, function evaluations
3. Comparing final solutions (should match to numerical precision)
4. Comparing intermediate values (can add debug output)

## Usage Example (Target API)

```rust
use eunoia_nlm::{optimize, NlmConfig, Method};
use nalgebra::DVector;

let rosenbrock = |x: &DVector<f64>| {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
};

let x0 = DVector::from_vec(vec![-1.2, 1.0]);
let config = NlmConfig {
    method: Method::LineSearch,
    max_iter: 150,
    ..Default::default()
};

let result = optimize(&x0, &rosenbrock, None, None, &config)?;
```

## Progress Tracking

Implement functions in phases as outlined above. After each phase:
1. All unit tests pass
2. `cargo clippy` with no warnings
3. `cargo fmt` applied
4. Documentation complete
5. Mark phase complete in this file

---

**Remember**: This is a faithful port. When in doubt, follow the C implementation exactly, even if it seems like there's a "better" way. The algorithm has been extensively tested and validated over decades. Our job is to translate, not improve.
