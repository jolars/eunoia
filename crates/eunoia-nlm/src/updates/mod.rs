//! Hessian updates and trust region management
//!
//! This module implements BFGS-style secant updates and trust region radius control.

pub mod secant;
pub mod trust_region;

pub use secant::{secfac, secunf, SecantParams};
pub use trust_region::{tregup, TregupParams, TregupResult, TregupReturnCode};
