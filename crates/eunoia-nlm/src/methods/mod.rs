//! Optimization methods for finding next iterate
//!
//! This module implements the three main methods:
//! - Line search (Method 1)
//! - Double dogleg (Method 2)
//! - More-Hebdon (Method 3)

pub mod dogleg;
pub mod hookstep;
pub mod linesearch;

pub use dogleg::{dog_1step, dogdrv, DoglegState};
pub use hookstep::{hook_1step, hookdrv, HookState};
pub use linesearch::{lnsrch, LnsrchParams, LnsrchResult, LnsrchReturnCode};
