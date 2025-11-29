//! Linear algebra operations for NLM algorithm
//!
//! This module contains matrix-vector operations, Cholesky decomposition,
//! and QR updates used throughout the optimization algorithm.

pub mod cholesky;
pub mod mvmult;
pub mod qr;

pub use cholesky::{chlhsn, choldc, lltslv};
pub use mvmult::{mvmltl, mvmlts, mvmltu};
pub use qr::{qraux1, qraux2, qrupdt};
