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

The project uses a **Cargo workspace** with separate crates:

### Core Library (`crates/eunoia/`)

Pure Rust library with no platform-specific dependencies.

#### Core Modules

- **`lib.rs`**: Main library interface
- **`spec/`**: Diagram specification and construction
  - `combination.rs`: Set combination representation
  - `input.rs`: Input type specification (disjoint vs union)
  - `spec_builder.rs`: DiagramSpecBuilder for fluent API
- **`geometry/`**: Geometric primitives and operations
  - `primitives/`: Basic geometric elements
    - `point.rs`: 2D point representation
    - `line.rs`: Line representation
  - `traits.rs`: Trait definitions for geometric capabilities
    - Composable traits: `Distance`, `Area`, `Centroid`, `Perimeter`, `BoundingBox`
    - `Closed`: Trait for closed shapes (bundles geometric properties + spatial relations)
    - `DiagramShape`: Trait for shapes usable in diagrams (adds optimization methods)
  - `shapes/`: Shape implementations
    - `circle.rs`: Circle shape implementation
    - `rectangle.rs`: Rectangle shape implementation (for bounding boxes)
  - `operations/`: Specialized geometric operations (currently `overlaps.rs`)
  - `diagram.rs`: Diagram-specific logic (region discovery, area computation)
- **`fitter/`**: Layout optimization
  - `layout.rs`: Layout representation (result of fitting)
  - `initial_layout.rs`: Initial layout computation
  - `final_layout.rs`: Final layout computation with region discovery and area
    computation
- **`error.rs`**: Error types
- **`math.rs`**: Mathematical utilities

### WASM Bindings (`crates/eunoia-wasm/`)

WebAssembly bindings that depend on the core library.

- Thin wrapper around `eunoia` crate
- Exports JavaScript-compatible API
- Handles serialization/deserialization for web use
- Built with `wasm-pack` to generate TypeScript bindings

### Web Application (`web/`)

Interactive diagram viewer built with Svelte + TypeScript.

- **`src/`**: Svelte components and TypeScript code
  - `lib/DiagramViewer.svelte`: Main UI for diagram specification and
    visualization
- **`pkg/`**: Generated WASM bindings (built with wasm-pack)
- Uses Vite (rolldown-vite) for development and building
- Tailwind CSS for styling
- Real-time diagram updates as specification changes

### Optimization Strategy

1. **Initial Layout (MDS with circles)**:
   - Compute pairwise relationships between sets
   - Use circles for MDS-based initialization (position + radius for each set)
   - This produces initial `(x, y, r)` parameters for each set
   - **Note**: Initial layout always uses circles, regardless of final shape type (this is what eulerr does)

2. **Convert to Shape-Specific Parameters**:
   - Initial circle parameters are converted to target shape parameters via `DiagramShape::params_from_circle()`
   - Circle: `[x, y, r]` → `[x, y, r]` (identity)
   - Ellipse (future): `[x, y, r]` → `[x, y, r, r, 0.0]` (semi-major=semi-minor=r, angle=0)

3. **Final Optimization (shape-specific)**:
   - Optimizes actual shape parameters (not circles!)
   - Uses `DiagramShape::compute_exclusive_regions()` for exact area computation
   - Minimizes loss function (currently region error) using argmin + Nelder-Mead
   - Constructs final shapes via `DiagramShape::from_params()`

4. **Layout Finalization**:
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
├── spec.rs              # Module definition
├── spec/                # Submodules of spec
│   ├── spec_builder.rs
│   ├── combination.rs
│   └── input.rs
├── geometry.rs          # Module definition
├── geometry/            # Submodules of geometry
│   ├── primitives.rs    # Primitives module definition
│   ├── primitives/
│   │   ├── point.rs
│   │   └── line.rs
│   ├── traits.rs        # All trait definitions
│   ├── shapes.rs        # Shapes module definition
│   ├── shapes/
│   │   ├── circle.rs
│   │   └── rectangle.rs
│   ├── operations.rs
│   ├── operations/
│   └── diagram.rs
├── fitter.rs            # Module definition
└── fitter/              # Submodules of fitter
    ├── initial_layout.rs
    └── final_layout.rs
```

**Do NOT use** the old `mod.rs` style:

```
src/
├── spec/
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

To add a new shape type (e.g., Ellipse):

1. **Create module in `src/geometry/shapes/`**:
   ```rust
   // src/geometry/shapes/ellipse.rs
   use crate::geometry::primitives::Point;
   use crate::geometry::shapes::Rectangle;
   use crate::geometry::traits::{
       Area, BoundingBox, Centroid, Closed, DiagramShape, Distance, Perimeter,
   };
   
   pub struct Ellipse {
       center: Point,
       semi_major: f64,
       semi_minor: f64,
       angle: f64,  // rotation angle in radians
   }
   ```

2. **Implement the component traits**:
   ```rust
   impl Area for Ellipse {
       fn area(&self) -> f64 { ... }
   }
   
   impl Centroid for Ellipse {
       fn centroid(&self) -> (f64, f64) { ... }
   }
   
   impl Distance for Ellipse {
       fn distance(&self, other: &Self) -> f64 { ... }
   }
   
   impl Perimeter for Ellipse {
       fn perimeter(&self) -> f64 { ... }
   }
   
   impl BoundingBox for Ellipse {
       fn bounding_box(&self) -> Rectangle { ... }
   }
   
   impl Closed for Ellipse {
       fn contains(&self, other: &Self) -> bool { ... }
       fn contains_point(&self, point: &Point) -> bool { ... }
       fn intersects(&self, other: &Self) -> bool { ... }
       fn intersection_area(&self, other: &Self) -> f64 { ... }
       fn intersection_points(&self, other: &Self) -> Vec<Point> { ... }
   }
   
   impl DiagramShape for Ellipse {
       fn compute_exclusive_regions(shapes: &[Self]) -> HashMap<RegionMask, f64> {
           // Implement exact ellipse intersection geometry
           // For 3+ ellipses, you may fall back to Monte Carlo
       }
       
       fn params_from_circle(x: f64, y: f64, radius: f64) -> Vec<f64> {
           vec![x, y, radius, radius, 0.0]  // a=b=r, angle=0 (circle is special ellipse)
       }
       
       fn n_params() -> usize {
           5  // x, y, semi_major, semi_minor, angle
       }
       
       fn from_params(params: &[f64]) -> Self {
           Ellipse::new(
               Point::new(params[0], params[1]),
               params[2],  // semi_major
               params[3],  // semi_minor
               params[4],  // angle
           )
       }
   }
   ```

3. **Add unit tests** covering all trait implementations

4. **Export new module** in `shapes.rs`:
   ```rust
   pub mod ellipse;
   ```

5. **Done!** The fitter will automatically:
   - Use circles for initial MDS layout
   - Convert circle parameters to ellipse parameters via `params_from_circle()`
   - Optimize ellipse-specific parameters (x, y, a, b, angle)
   - Use exact ellipse geometry via `compute_exclusive_regions()`
   - Construct final ellipses via `from_params()`

The entire optimization pipeline is **shape-generic** - no changes needed to fitter code!

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

- ✅ Cargo workspace with separate crates for core and WASM
- ✅ Basic project structure
- ✅ Point-based coordinate system (`Point`)
- ✅ **Shape trait with generic parameter system**
- ✅ Circle implementation with full geometric operations and exact area computation
- ✅ Rectangle implementation for bounding boxes
- ✅ Line and LineSegment primitives
- ✅ **Trait-based design with generic `Shape` trait**
  - `compute_exclusive_regions()` - Shape-specific exact area computation
  - `params_from_circle()` - Convert initial circle params to shape params
  - `n_params()` - Number of parameters per shape
  - `from_params()` - Construct shape from parameters
- ✅ DiagramSpecBuilder with fluent API for input (generic over shape type)
- ✅ **Shape-generic fitter with two-phase optimization**
  - Initial layout: MDS with circles (always uses circles)
  - Final optimization: Shape-specific parameter optimization
- ✅ Region error loss function with sparse region discovery
- ✅ Final layout with disjoint area computation using inclusion-exclusion
- ✅ Layout representation with fitted areas (generic over shape type)
- ✅ WASM bindings in separate crate (`crates/eunoia-wasm/`)
- ✅ Web application (Svelte 5 + TypeScript + Vite + Tailwind v4)
- ✅ Interactive diagram viewer with real-time updates
- ✅ Comprehensive unit tests (190 tests passing)
- ✅ Full documentation with doc tests (24 doc tests passing)
- ✅ 3-way intersection area computation implemented
- ✅ Random seed support for reproducible layouts
- ✅ Generic area computation (Monte Carlo for 3+ way intersections on non-Circle shapes)
- ❌ Stress loss function (venneuler-style) not implemented
- ❌ Ellipse and triangle shapes not implemented
- ❌ Polygon conversion utilities not implemented
- ❌ Label placement algorithms not implemented (poles of inaccessibility)

## Dependencies

### Core Library (`crates/eunoia/`)

Core dependencies (platform-independent):

- **`nalgebra`** (0.34.1) - Linear algebra for MDS and matrix operations
- **`argmin`** (0.11.0) - Optimization algorithms (gradient-based and
  derivative-free)
- **`argmin-math`** (0.5.1) - Math support for argmin with nalgebra
- **`rand`** (0.9) - Random number generation
- Standard library for basic operations

### WASM Bindings (`crates/eunoia-wasm/`)

WASM-specific dependencies:

- **`eunoia`** (path dependency) - Core library
- **`wasm-bindgen`** (0.2) - WASM bindings generation
- **`serde`** (1.0) - Serialization framework
- **`serde-wasm-bindgen`** (0.6) - Serde support for WASM
- **`serde_json`** (1.0) - JSON serialization
- **`console_error_panic_hook`** (0.1) - Better error messages in browser
  console
- **`web-sys`** (0.3) - Web APIs

### Web Application (`web/`)

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
# Using Task
task build-wasm

# Or directly with wasm-pack
wasm-pack build crates/eunoia-wasm --target web --out-dir ../../web/pkg
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

### Working with the Workspace

The project uses a Cargo workspace with multiple crates:

- **Build all crates**: `cargo build --workspace`
- **Build core library only**: `cargo build -p eunoia` (default)
- **Build WASM bindings**: `cargo build -p eunoia-wasm`
- **Test core library**: `cargo test -p eunoia`
- **Format all crates**: `cargo fmt` (workspace-level)
- **Lint all crates**: `cargo clippy --workspace --all-targets --all-features -- -D warnings`

### When Adding Code

- Write tests first or alongside implementation (TDD encouraged)
- Document public APIs immediately
- **Run `cargo fmt` (or `task fmt`) to format code**
- Run `cargo test` to ensure tests pass (or `cargo test -p eunoia` for core only)
- Run `cargo doc --open` to verify documentation
- **Run `cargo clippy --workspace --all-targets --all-features -- -D warnings` (or
  `task lint`) for linting (must pass with no warnings)**
- Or simply run `task dev` to run the full development workflow

### When Working with WASM

- WASM bindings live in `crates/eunoia-wasm/`
- Depend on core library via `eunoia = { path = "../eunoia" }`
- Keep WASM layer thin - just exports and type conversions
- Core logic stays in `crates/eunoia/`
- Test WASM builds with: `wasm-pack build crates/eunoia-wasm --target web`

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

- **Uses Cargo workspace** with `crates/eunoia/` (core) and `crates/eunoia-wasm/` (bindings)
- Core library is platform-independent Rust with no WASM dependencies
- WASM bindings are a thin wrapper in a separate crate
- **Uses Rust Edition 2021** for compatibility with rextendr (R bindings)
- **Uses Semantic Versioning (SemVer)** - currently pre-1.0.0 (alpha)
- **Breaking changes are acceptable** in pre-1.0.0 versions
- **Uses Conventional Commits** for clear change history
- API stability will matter after 1.0.0 for language bindings (maintained in
  separate repositories)
- **Only WASM bindings** will be part of this repository
- **R, Python, and Julia bindings** will be separate repositories that depend on
  the core `eunoia` library
- Geometric correctness over performance initially
- Optimize after correctness is established and tested
