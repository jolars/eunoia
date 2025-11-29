# Integration Tests

This directory contains integration tests for the eunoia-nlm optimization library.

## Current Status

**Core algorithms are fully implemented and unit tested** (67 passing unit tests):
- ✅ Linear algebra (Cholesky, solvers)
- ✅ Finite differences (gradient, Hessian)
- ✅ Line search
- ✅ Dogleg trust region
- ✅ Hookstep trust region  
- ✅ BFGS secant updates
- ✅ Stopping criteria

**Integration layer needs debugging**:
- The main optimization driver (`optdrv`) has been implemented but needs debugging
- Individual components work correctly in isolation
- The driver's orchestration of components needs refinement

## Current Tests

- `test_gradient_check()` - Verifies gradient computation
- `test_function_evaluation()` - Verifies test functions

## Future Work

Once the driver is debugged, add tests for:
- Quadratic minimization (all 3 methods)
- Rosenbrock function
- High-dimensional problems
- Different starting points
- Convergence criteria
- Method comparison

The infrastructure is in place, just needs driver fixes.
