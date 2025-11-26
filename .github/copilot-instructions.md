# Eunoia - Copilot Instructions

## Project Overview

Eunoia is a Rust library for creating Euler and Venn diagrams using various
geometric shapes. It's a modern rewrite of the eulerr R package (which uses C++
as backend), designed to be more flexible and support multiple language
bindings.

**Reference Implementation**: The original eulerr R package is included in
`./eulerr/` as a reference for algorithm details, implementation specifics, and
testing strategies.

## Purpose

Generate area-proportional Euler and Venn diagrams by:

1. Computing optimal layouts using multi-dimensional scaling (MDS) with
   fixed-size shapes
2. Optimizing shape positions and sizes using comprehensive loss functions
   (RegionError or stress from venneuler)
3. Providing utilities for polygon conversion, intersection computation, and
   label placement (poles of inaccessibility, centroids)

## Supported Shapes

- **Current**: Circles
- **Planned**: Ellipses, rectangles, triangles

## Target Bindings (Future)

- **WebAssembly (WASM)** for web applications - will be part of this repository
- **R package** (thin wrapper) - separate repository
- **Python package** (thin wrapper) - separate repository
- **Julia package** (thin wrapper) - separate repository

**Note**: Only WASM bindings will be maintained in this repository.
Language-specific bindings (R, Python, Julia) will be maintained in their own
dedicated repositories as thin wrappers around this core library.

## Architecture

### Core Modules

- **`lib.rs`**: Main library interface
- **`diagram/`**: Diagram specification and construction
  - `combination.rs`: Set combination representation
  - `input.rs`: Input type specification (disjoint vs union)
  - `spec.rs`: DiagramSpecBuilder for fluent API
- **`geometry/`**: Geometric primitives and operations
  - `point.rs`: 2D point representation
  - `line.rs`: Line representation
  - `line_segment.rs`: Line segment representation
  - `shapes.rs`: Shape trait definition with common operations (Area, Centroid,
    Distance, Contains, Intersects, IntersectionArea, Perimeter, etc.)
  - `shapes/`: Shape implementations (currently `circle.rs` and `rectangle.rs`)
  - `operations/`: Specialized geometric operations (currently `overlaps.rs`)
- **`fitter/`**: Layout optimization
  - `layout.rs`: Layout representation (result of fitting)
  - `initial_layout.rs`: Initial layout computation
  - `final_layout.rs`: Final layout computation with region discovery and area
    computation
- **`error.rs`**: Error types
- **`math.rs`**: Mathematical utilities
- **`wasm.rs`**: WASM bindings for web integration

### Web Application

- **`web/`**: Interactive diagram viewer built with Svelte + TypeScript
  - **`src/`**: Svelte components and TypeScript code
    - `lib/DiagramViewer.svelte`: Main UI for diagram specification and
      visualization
  - **`pkg/`**: Generated WASM bindings (built with wasm-pack)
  - Uses Vite (rolldown-vite) for development and building
  - Tailwind CSS for styling
  - Real-time diagram updates as specification changes

### Optimization Strategy

1. **Initial Layout**:
   - Compute pairwise relationships between sets
   - Use simple circular arrangement as starting point
   - TODO: Implement MDS-based initialization

2. **Iterative Optimization**:
   - Compute all shape intersections in each iteration
   - Compare to user-specified set relationships
   - Minimize loss function (currently region error)
   - Optimize both positions and sizes using argmin

3. **Layout Finalization**:
   - Return fitted shapes with computed areas
   - TODO: Polygon conversion utilities
   - TODO: Label placement (poles of inaccessibility)
   - TODO: Calculate centroids and other metrics

## Code Standards

### Documentation

- **Required**: All public functions, structs, traits, and modules must have
  rustdoc comments
- Use `///` for item documentation and `//!` for module-level docs
- Include examples in doc comments where appropriate
- Target: 100% documentation coverage for public APIs

### Testing

- **Required**: Unit tests for all functions
- Use `#[cfg(test)]` modules in each file
- Target: 100% code coverage
- Test edge cases, especially for geometric operations (tangent circles,
  containment, no intersection, etc.)
- Property-based testing for geometric invariants (consider using `proptest`
  crate)

### Code Style

- Follow Rust idioms and conventions
- Use descriptive variable names
- Prefer explicit types for clarity in geometric code
- Keep functions focused and single-purpose
- Use traits for polymorphism over shapes
- **Use Rust Edition 2021** (for compatibility with rextendr R bindings)
- **Use modern module organization** (Rust 2018+): `module.rs` + `module/`
  instead of `module/mod.rs`
- **Format code with `cargo fmt`** before committing
- **All code must pass
  `cargo clippy --all-targets --all-features -- -D warnings`**

### Module Organization

Use the Rust 2018+ module system:

```
src/
├── lib.rs
├── diagram.rs           # Module definition
├── diagram/             # Submodules of diagram
│   ├── builder.rs
│   ├── combination.rs
│   └── input.rs
├── solver.rs            # Module definition
└── solver/              # Submodules of solver
    ├── layout.rs
    └── optimize.rs
```

**Do NOT use** the old `mod.rs` style:

```
src/
├── diagram/
│   └── mod.rs          # ❌ Avoid this pattern
```

### Commit Guidelines

- Use [Conventional Commits](https://www.conventionalcommits.org/) format
- Commit message format: `<type>(<scope>): <description>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Examples:
  - `feat(geometry): add ellipse shape implementation`
  - `fix(circle): correct intersection area calculation`
  - `docs(api): update DiagramBuilder examples`
  - `test(shapes): add edge cases for containment`
- Breaking changes: Add `!` after type/scope or `BREAKING CHANGE:` in footer
- Keep commits atomic and focused on a single concern

### Versioning

- Follow [Semantic Versioning](https://semver.org/) (SemVer)
- Version format: `MAJOR.MINOR.PATCH`
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible new functionality
- PATCH: Backwards-compatible bug fixes
- Pre-1.0.0: API is not yet stable (current status)

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
3. Implement the `Shape` trait from `shapes.rs`:
   - `area()`
   - `distance()`
   - `contains()`
   - `intersects()`
   - `intersection_area()`
   - `intersection_points()`
   - `centroid()`
   - `perimeter()`
   - `contains_point()`
   - `bounding_box()`
4. Add unit tests covering all trait implementations
5. Update `shapes.rs` to export new module

### Geometric Operations

- All shapes share a common coordinate system (`Point`)
- All shapes implement the `Shape` trait defined in `shapes.rs`
- Distance calculations should handle edge cases (overlapping, containment)
- Intersection area formulas should be numerically stable
- Consider numerical precision issues (use appropriate tolerances)
- Specialized operations live in `geometry/operations/` module

### Optimization Code

- Lives in `fitter/` module
- Uses argmin for optimization
- Region error loss function implemented in `final_layout.rs`
- Region discovery uses sparse bit-mask approach for efficiency
- Final layout computes disjoint areas using inclusion-exclusion principle
- TODO: Make loss functions pluggable (trait-based)
- TODO: Implement MDS initialization
- TODO: Support stress loss function (venneuler-style)

### Web Development

- **Building WASM**: `wasm-pack build --target web --out-dir web/pkg`
- **Dev server**: `cd web && npm run dev`
- **Building web app**: `cd web && npm run build`
- WASM module exports:
  - `generate_from_spec(specs: Vec<DiagramSpec>)`: Main function for fitting
    diagrams
  - `generate_test_layout()`: Simple test layout
  - `compute_layout(n_sets: usize)`: Generate circular arrangement
- Web app uses reactive Svelte components for real-time updates

## Current Status

- ✅ Basic project structure
- ✅ Point-based coordinate system (`Point`)
- ✅ Circle implementation with full geometric operations
- ✅ Rectangle implementation for bounding boxes
- ✅ Line and LineSegment primitives
- ✅ Trait-based design with `Shape` trait
- ✅ DiagramSpecBuilder with fluent API for input
- ✅ Fitter with argmin-based optimization
- ✅ Region error loss function with sparse region discovery
- ✅ Final layout with disjoint area computation using inclusion-exclusion
- ✅ Layout representation with fitted areas
- ✅ WASM bindings (`wasm.rs`)
- ✅ Web application (Svelte 5 + TypeScript + Vite + Tailwind v4)
- ✅ Interactive diagram viewer with real-time updates
- ✅ Comprehensive unit tests (178 tests passing)
- ✅ Full documentation with doc tests
- ✅ 3-way intersection area computation implemented
- ❌ Stress loss function (venneuler-style) not implemented
- ❌ Ellipse and triangle shapes not implemented
- ❌ Polygon conversion utilities not implemented
- ❌ Label placement algorithms not implemented (poles of inaccessibility)
- ❌ Language bindings not started (will be separate repositories)

## Dependencies

Core dependencies:

- **`nalgebra`** (0.34.1) - Linear algebra for MDS and matrix operations
- **`argmin`** (0.11.0) - Optimization algorithms (gradient-based and
  derivative-free)
- **`argmin-math`** (0.5.1) - Math support for argmin with nalgebra
- Standard library for basic operations

WASM dependencies:

- **`wasm-bindgen`** (0.2) - WASM bindings generation
- **`serde`** (1.0) - Serialization framework
- **`serde-wasm-bindgen`** (0.6) - Serde support for WASM
- **`console_error_panic_hook`** (0.1) - Better error messages in browser
  console
- **`getrandom`** (0.2) - Random number generation for WASM

Web app dependencies:

- **Svelte** (5.43.8) - UI framework
- **TypeScript** (~5.9.3) - Type safety
- **Vite** (rolldown-vite@7.2.5) - Build tool and dev server
- **Tailwind CSS** (4.1.17) - Styling
- **vite-plugin-wasm** (3.5.0) - WASM support for Vite
- **vite-plugin-top-level-await** (1.6.0) - Top-level await support

Keep the dependency footprint minimal. New dependencies should be carefully
considered for:

- Compile time impact
- Binary size (important for WASM)
- Maintenance burden
- License compatibility

**Note**: The project uses **Rust Edition 2021** for compatibility with rextendr
(R bindings framework). This is specified in `Cargo.toml` and should not be
changed without consideration for downstream binding compatibility.

## Working with This Codebase

### Task Runner

The project uses [Task](https://taskfile.dev) for common development workflows.
See `Taskfile.yml` for available tasks:

- `task fmt` - Format code with rustfmt
- `task fmt-check` - Check if code is formatted
- `task lint` - Run clippy with strict warnings
- `task dev` - Full development workflow (format + check + test + lint)
- `task test-debug` - Run tests with debug logging
- `task test-quiet` - Run tests without debug output
- `task coverage` - Generate code coverage report
- `task coverage-open` - Generate and open coverage report
- `task build-release` - Build optimized release binary

You can run these instead of manual cargo commands for convenience.

### WASM and Web App Development

Building WASM:

```bash
wasm-pack build --target web --out-dir web/pkg
```

Running web dev server:

```bash
cd web
npm install --include=dev  # First time only
npm run dev
```

Building web app for production:

```bash
cd web
npm run build
```

**Note**: If you encounter "omit=dev" issues with npm, run:

```bash
npm config delete omit
```

### When Adding Code

- Write tests first or alongside implementation (TDD encouraged)
- Document public APIs immediately
- **Run `cargo fmt` (or `task fmt`) to format code**
- Run `cargo test` to ensure tests pass
- Run `cargo doc --open` to verify documentation
- **Run `cargo clippy --all-targets --all-features -- -D warnings` (or
  `task lint`) for linting (must pass with no warnings)**
- Or simply run `task dev` to run the full development workflow

### When Reviewing Code

- Verify geometric correctness with test cases
- Check numerical stability (divide by zero, sqrt of negative, etc.)
- Ensure proper error handling
- Confirm documentation is clear and accurate
- Verify code is formatted with `cargo fmt`
- Ensure clippy passes with `-D warnings` flag

### Mathematical References

- **eulerr R package** in `./eulerr/` - Reference implementation with C++
  backend
  - Consult for algorithm details, optimization strategies, and edge case
    handling
  - Source code in `./eulerr/src/` and `./eulerr/R/`
  - Tests in `./eulerr/tests/`
- venneuler for stress loss function
- Classic MDS algorithms for initialization
- Poles of inaccessibility: Polylabel algorithm

## Future Considerations

- **Stress loss function** (venneuler-style) as alternative to region error
- **Other shapes**: Ellipses, rectangles, triangles
- **Polygon conversion** utilities for complex visualizations
- **Label placement** algorithms (poles of inaccessibility, centroids)
- Parallel optimization for large diagrams
- GPU acceleration for polygon operations
- Enhanced web UI features (export SVG, PNG, customization options)

## Notes

- This is a library crate, not a binary
- **Uses Rust Edition 2021** for compatibility with rextendr (R bindings)
- **Uses Semantic Versioning (SemVer)** - currently pre-1.0.0 (alpha)
- **Breaking changes are acceptable** in pre-1.0.0 versions
- **Uses Conventional Commits** for clear change history
- API stability will matter after 1.0.0 for language bindings (maintained in
  separate repositories)
- **Only WASM bindings** will be part of this repository
- **R, Python, and Julia bindings** will be separate repositories that depend on
  this core library
- Geometric correctness over performance initially
- Optimize after correctness is established and tested
