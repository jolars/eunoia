# Contributing to Eunoia

We welcome contributions to eunoia! This guide will help you get started.

## Getting Started

### Prerequisites

- Rust toolchain (see `rust-toolchain.toml` for version)
- [Task](https://taskfile.dev) (optional, but recommended for convenience)
- Git

### Setting Up Your Development Environment

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/yourusername/eunoia.git
   cd eunoia
   ```

2. Install dependencies (if needed):

   ```bash
   # The project uses standard Rust tooling
   cargo build
   ```

3. Verify your setup:
   ```bash
   task dev
   # Or manually:
   cargo fmt && cargo check && cargo test && cargo clippy --all-targets --all-features -- -D warnings
   ```

## Development Workflow

### Making Changes

1. **Create a branch** for your work:

   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following the code standards below

3. **Add tests** for any new functionality

4. **Update documentation** for any API changes

5. **Run the development workflow**:
   ```bash
   task dev
   ```

### Code Standards

#### Formatting and Linting

- **Format code**: `cargo fmt` (or `task fmt`)
- **Check formatting**: `cargo fmt -- --check` (or `task fmt-check`)
- **Lint code**: `cargo clippy --all-targets --all-features -- -D warnings` (or
  `task lint`)

All code must pass these checks with no warnings.

#### Testing

- Write unit tests for all functions
- Write doc tests for public APIs with examples
- Run tests: `cargo test`
- Target 100% code coverage
- Test edge cases, especially for geometric operations

#### Documentation

- Document all public APIs with rustdoc (`///` comments)
- Include examples in doc comments where appropriate
- Update README.md if adding major features

#### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(geometry): add ellipse shape implementation
fix(circle): correct intersection area calculation
docs(api): update DiagramSpecBuilder examples
test(shapes): add edge cases for containment
```

**Breaking changes:** Add `!` after type/scope or include `BREAKING CHANGE:` in
the footer:

```
feat(api)!: redesign builder API

BREAKING CHANGE: Builder methods now return Result
```

## Submitting Changes

### Pull Request Process

1. **Ensure all checks pass**:

   ```bash
   task dev
   ```

2. **Push your branch**:

   ```bash
   git push origin feat/your-feature-name
   ```

3. **Create a Pull Request** with:
   - Clear title following conventional commits format
   - Description of changes
   - References to related issues
   - Screenshots/examples if applicable

4. **Address review feedback** if requested

### Before Submitting

Quick checklist:

- [ ] Code is formatted (`cargo fmt -- --check` passes)
- [ ] No clippy warnings
      (`cargo clippy --all-targets --all-features -- -D warnings` passes)
- [ ] All tests pass (`cargo test` succeeds)
- [ ] New features have tests
- [ ] Public APIs are documented
- [ ] Commit messages follow conventional commits
- [ ] Branch is up to date with main

Or simply run:

```bash
task dev
```

## Project Structure

We use a **Cargo workspace** with separate crates for core functionality and WASM bindings:

```
eunoia/
├── Cargo.toml              # Workspace definition
├── crates/
│   ├── eunoia/             # Core library (pure Rust)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs      # Main API surface, re-exports
│   │   │   ├── spec.rs     # Spec module definition
│   │   │   ├── spec/       # Spec submodules
│   │   │   │   ├── spec_builder.rs
│   │   │   │   ├── combination.rs
│   │   │   │   └── input.rs
│   │   │   ├── error.rs    # Error types
│   │   │   ├── geometry.rs # Geometry module definition
│   │   │   ├── geometry/   # Geometric primitives
│   │   │   │   ├── point.rs
│   │   │   │   ├── shapes.rs
│   │   │   │   └── shapes/
│   │   │   │       └── circle.rs
│   │   │   ├── fitter.rs   # Fitter module definition
│   │   │   └── fitter/     # Optimization algorithms
│   │   │       ├── layout.rs
│   │   │       ├── initial_layout.rs
│   │   │       └── final_layout.rs
│   │   └── tests/          # Integration tests
│   └── eunoia-wasm/        # WASM bindings
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs      # WASM exports
├── web/                    # Svelte web application
└── README.md
```

We use the **Rust 2018+ module system** (`module.rs` + `module/` instead of `module/mod.rs`).

### Working with the Workspace

- **Build core library**: `cargo build` (default) or `cargo build -p eunoia`
- **Build WASM**: `cargo build -p eunoia-wasm` or `task build-wasm`
- **Test core library**: `cargo test` or `cargo test -p eunoia`
- **Format all crates**: `cargo fmt` (runs on workspace)
- **Lint all crates**: `cargo clippy --all-targets --all-features -- -D warnings`


## Areas for Contribution

We welcome contributions in these areas:

- **New shapes**: Ellipse, rectangle, triangle implementations
- **Optimization algorithms**: MDS initialization, loss functions
- **Polygon utilities**: Shape conversion, intersection splitting
- **Label placement**: Poles of inaccessibility, centroid calculations
- **Documentation**: Examples, tutorials, API docs
- **Tests**: Edge cases, property-based tests
- **Performance**: Optimization, profiling
- **WASM bindings**: Improvements to WebAssembly API in `crates/eunoia-wasm`

## Getting Help

- Check the [documentation](.github/copilot-instructions.md) for detailed
  guidelines
- Open an issue for questions or discussions
- Review existing issues and PRs for context

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project (see LICENSE file).

## Thank You!

Thank you for contributing to eunoia! Your help makes this project better for
everyone.
