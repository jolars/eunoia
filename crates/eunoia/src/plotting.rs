//! Polygon-based plotting utilities.
//!
//! This module provides utilities for converting fitted diagram shapes into
//! polygons and decomposing them into exclusive regions for visualization.
//!
//! ## Features
//!
//! - Polygon clipping operations (intersection, union, difference, xor)
//! - Region decomposition for exclusive area visualization
//! - Label placement utilities (future)
//!
//! This module is only available when the `plotting` feature is enabled.

#[cfg(feature = "plotting")]
mod clip;
#[cfg(feature = "plotting")]
mod plot_data;
#[cfg(feature = "plotting")]
mod regions;

#[cfg(feature = "plotting")]
pub use clip::{polygon_clip, ClipOperation};
#[cfg(feature = "plotting")]
pub(crate) use plot_data::build_plot_data;
#[cfg(feature = "plotting")]
pub use plot_data::{PlotData, PlotOptions};
#[cfg(feature = "plotting")]
pub use regions::{
    decompose_regions, decompose_regions_with, RegionPiece, RegionPolygons,
    DEFAULT_SLIVER_THRESHOLD,
};
