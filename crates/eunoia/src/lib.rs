//! A Rust library for creating area-proportional Euler and Venn diagrams.
//!
//! A fit has two stages:
//!
//! 1. Multi-dimensional scaling places fixed-size shapes.
//! 2. Numerical optimization refines the shapes against the chosen loss.
//!
//! ## Example
//!
//! ```rust
//! use eunoia::{DiagramSpecBuilder, Fitter, InputType};
//! use eunoia::geometry::shapes::Circle;
//!
//! let spec = DiagramSpecBuilder::new()
//!     .set("A", 5.0)
//!     .set("B", 2.0)
//!     .intersection(&["A", "B"], 1.0)
//!     .input_type(InputType::Exclusive)
//!     .build()
//!     .expect("Failed to build diagram specification");
//!
//! let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
//! ```

pub mod error;
pub mod fitter;
pub mod geometry;
pub mod loss;
pub(crate) mod math;
pub mod spec;

pub mod constants;
pub mod venn;

pub mod plotting;

// Internal test utilities. Also reachable from example binaries when the
// `corpus` feature is enabled — only the `corpus` submodule is publicly
// usable in that mode; everything else stays gated on `cfg(test)`.
#[cfg(any(test, feature = "corpus"))]
pub mod test_utils;

pub use error::DiagramError;
/// Internal global-escape solver selector, exposed only under the `corpus`
/// feature for `examples/quality_report` (see [`Fitter::escape_solver`]).
#[cfg(any(test, feature = "corpus"))]
pub use fitter::final_layout::EscapeSolver;
pub use fitter::{Fitter, InitialSampler, Layout, MdsSolver, Optimizer};
pub use spec::{Combination, DiagramSpecBuilder, InputType};
pub use venn::VennDiagram;
