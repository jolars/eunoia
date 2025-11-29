# eunoia-nlm

Complete Rust port of the Dennis-Schnabel Nonlinear Minimization algorithm from R's `nlm()` function.

## Status

🚧 **UNDER DEVELOPMENT** 🚧

This is a complete, faithful port of the 2,614-line C implementation. No simplifications.

See `IMPLEMENTATION_PROGRESS.md` for current status.

## Goal

Provide a pure Rust implementation of the Dennis-Schnabel optimization algorithm with:

- ✅ Complete algorithm fidelity (no shortcuts)
- ✅ Three optimization methods (line search, dogleg, More-Hebdon)
- ✅ Trust region methods
- ✅ Numerical differentiation (forward and central differences)
- ✅ BFGS-style secant updates
- ✅ Comprehensive testing against R's nlm()

## Reference

**Algorithm**: Dennis & Schnabel (1983) "Numerical Methods for Unconstrained Optimization and Nonlinear Equations"

**Source**: `nlm/nlm.c` in this repository (from R source)

## Documentation

- **Implementation Plan**: `.github/instructions/nlm.instructions.md`
- **Progress Tracking**: `IMPLEMENTATION_PROGRESS.md`
- **Quick Reference**: `QUICK_REFERENCE.md`

## Dependencies

- `nalgebra` - Linear algebra operations
- `thiserror` - Error handling

## Target API (Once Complete)

```rust
use eunoia_nlm::{optimize, NlmConfig, Method};
use nalgebra::DVector;

// Define objective function
let rosenbrock = |x: &DVector<f64>| {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
};

// Initial point
let x0 = DVector::from_vec(vec![-1.2, 1.0]);

// Configure
let config = NlmConfig {
    method: Method::LineSearch,
    max_iter: 150,
    ..Default::default()
};

// Optimize
let result = optimize(&x0, &rosenbrock, None, None, &config)?;

println!("Solution: {:?}", result.x);
println!("Function value: {}", result.f);
println!("Iterations: {}", result.iterations);
```

## Development Phases

1. **Phase 1**: Linear algebra (Cholesky, QR, matrix-vector ops)
2. **Phase 2**: Numerical differentiation (gradients, Hessians)
3. **Phase 3**: Updates & initialization (secant, trust region)
4. **Phase 4**: Optimization methods (line search, dogleg, hook)
5. **Phase 5**: Driver & integration (main loop, stopping criteria)

## Testing

Each function will be tested against:
- Known mathematical results
- C implementation output
- Standard test problems (Rosenbrock, Powell, Wood, etc.)

## License

GPL-2+ (matching original R source)

## Contributing

This is a faithful port project. Changes should maintain algorithm fidelity with the original C implementation.
