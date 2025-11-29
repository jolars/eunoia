//! Numerical differentiation for NLM algorithm
//!
//! This module provides finite difference approximations to gradients and
//! Hessians, as well as validation of user-supplied analytic derivatives.

pub mod checking;
pub mod gradient;
pub mod hessian;

pub use checking::{grdchk, heschk, GrdchkParams, HeschkParams};
pub use gradient::{fstocd, fstofd_gradient, fstofd_hessian};
pub use hessian::{fdhess, sndofd};
