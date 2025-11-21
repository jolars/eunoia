# Eunoia - Copilot Instructions

## Project Overview

Eunoia is a Rust library for creating Euler and Venn diagrams using various geometric shapes. It's a modern rewrite of the eulerr R package (which uses C++ as backend), designed to be more flexible and support multiple language bindings.

## Purpose

Generate area-proportional Euler and Venn diagrams by:
1. Computing optimal layouts using multi-dimensional scaling (MDS) with fixed-size shapes
2. Optimizing shape positions and sizes using comprehensive loss functions (RegionError or stress from venneuler)
3. Providing utilities for polygon conversion, intersection computation, and label placement (poles of inaccessibility, centroids)

## Supported Shapes

- **Current**: Circles
- **Planned**: Ellipses, rectangles, triangles

## Target Bindings (Future)

- WebAssembly (WASM) for web applications
- R package (thin wrapper)
- Python package (thin wrapper)
- Julia package (thin wrapper)

## Architecture

### Core Modules

- **`lib.rs`**: Main library interface, defines `Diagram<S>` structure and core traits:
  - `Intersects<S>`: Compute intersection areas between shapes
  - `Parameters`: For optimization (get/set parameters)
  
- **`geometry/`**: Geometric primitives and operations
  - `coord.rs`: 2D coordinate representation
  - `operations.rs`: Geometric operation traits (Area, Centroid, Distance, Contains, Intersects, IntersectionArea, Perimeter)
  - `shapes/`: Shape implementations (currently `circle.rs`)

- **`math.rs`**: Mathematical utilities (currently empty, will contain MDS, optimization algorithms, loss functions)

### Optimization Strategy

1. **Initial Layout** (MDS-based):
   - Convert user input to optimal pairwise distances
   - Use multi-dimensional scaling to place fixed-size shapes
   - Optimize to match distance matrix

2. **Comprehensive Optimization**:
   - Compute all shape intersections in each iteration
   - Compare to user-specified set relationships
   - Minimize loss function: RegionError (preferred) or stress (venneuler-style)
   - Optimize both positions and sizes

3. **Layout Finalization**:
   - Convert shapes to polygons
   - Split into all intersection regions
   - Compute poles of inaccessibility for label placement
   - Calculate centroids and other metrics

## Code Standards

### Documentation
- **Required**: All public functions, structs, traits, and modules must have rustdoc comments
- Use `///` for item documentation and `//!` for module-level docs
- Include examples in doc comments where appropriate
- Target: 100% documentation coverage for public APIs

### Testing
- **Required**: Unit tests for all functions
- Use `#[cfg(test)]` modules in each file
- Target: 100% code coverage
- Test edge cases, especially for geometric operations (tangent circles, containment, no intersection, etc.)
- Property-based testing for geometric invariants (consider using `proptest` crate)

### Code Style
- Follow Rust idioms and conventions
- Use descriptive variable names
- Prefer explicit types for clarity in geometric code
- Keep functions focused and single-purpose
- Use traits for polymorphism over shapes
- **Format code with `cargo fmt`** before committing
- **All code must pass `cargo clippy --all-targets --all-features -- -D warnings`**

### Error Handling
- Use `Result<T, E>` for fallible operations
- Define custom error types as needed
- Avoid panics in library code; use `?` operator and return errors

### Performance Considerations
- Geometric calculations are performance-critical
- Prefer stack allocation where possible
- Consider SIMD optimization for bulk operations (future)
- Profile before optimizing

## Development Guidelines

### Adding New Shapes

1. Create module in `src/geometry/shapes/`
2. Define shape struct with center and size parameters
3. Implement required traits from `operations.rs`:
   - `Area`
   - `Distance`
   - `Contains`
   - `Intersects`
   - `IntersectionArea`
4. Add unit tests covering all trait implementations
5. Update `shapes.rs` to export new module

### Geometric Operations

- All shapes share a common coordinate system (`Coord`)
- Distance calculations should handle edge cases (overlapping, containment)
- Intersection area formulas should be numerically stable
- Consider numerical precision issues (use appropriate tolerances)

### Optimization Code

- Will live in `math.rs` module
- Separate concerns: MDS initialization, loss functions, optimizers
- Make loss functions pluggable (trait-based)
- Support gradient-based and derivative-free optimization

### Plotting (Development Only)

- Consider optional feature flag for plotting dependencies
- Minimal plotting for development/debugging only
- Keep plotting code separate from core library
- Use `plotters` or similar Rust plotting crate
- Do not include plotting in final library API

## Current Status

- ✅ Basic project structure
- ✅ Coordinate system (`Coord`)
- ✅ Circle implementation with geometric operations
- ✅ Trait-based design for operations
- ✅ DiagramBuilder with fluent API for input
- ✅ Comprehensive unit tests (38 tests)
- ✅ Full documentation with doc tests (13 doc tests)
- ✅ Clippy-clean codebase (passes with `-D warnings`)
- ❌ Math module (MDS, optimization) not implemented
- ❌ Other shapes (ellipse, rectangle, triangle) not implemented
- ❌ Polygon conversion utilities not implemented
- ❌ Label placement algorithms not implemented
- ❌ Language bindings not started

## Dependencies

Keep dependencies minimal for core library:
- Standard library for most operations
- Consider `nalgebra` for linear algebra (MDS)
- Consider `argmin` or `optimization` crate for optimization
- Development plotting: `plotters` (feature-gated)

## Working with This Codebase

### When Adding Code
- Write tests first or alongside implementation (TDD encouraged)
- Document public APIs immediately
- **Run `cargo fmt` to format code**
- Run `cargo test` to ensure tests pass
- Run `cargo doc --open` to verify documentation
- **Run `cargo clippy --all-targets --all-features -- -D warnings` for linting (must pass with no warnings)**

### When Reviewing Code
- Verify geometric correctness with test cases
- Check numerical stability (divide by zero, sqrt of negative, etc.)
- Ensure proper error handling
- Confirm documentation is clear and accurate
- Verify code is formatted with `cargo fmt`
- Ensure clippy passes with `-D warnings` flag

### Mathematical References
- Refer to eulerr paper/source for algorithm details
- venneuler for stress loss function
- Classic MDS algorithms for initialization
- Poles of inaccessibility: Polylabel algorithm

## Future Considerations

- WASM compilation targets (`wasm32-unknown-unknown`)
- FFI for R/Python/Julia bindings
- Serialization support for diagram interchange
- Parallel optimization for large diagrams
- GPU acceleration for intersection calculations

## Notes

- This is a library crate, not a binary
- API stability is important for language bindings
- Geometric correctness over performance initially
- Optimize after correctness is established and tested
