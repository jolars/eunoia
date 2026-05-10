//! Polygon-based plotting utilities.
//!
//! This module provides utilities for converting fitted diagram shapes into
//! polygons and decomposing them into exclusive regions for visualization.
//!
//! ## Features
//!
//! - Polygon clipping operations (intersection, union, difference, xor)
//! - Region decomposition for exclusive area visualization
//! - Label-anchor utilities (poles of inaccessibility, principal axis)
//! - Label-fit predicates ([`fit_label_in_region`], [`fit_labels_in_regions`])
//! - Strategy-driven label placement ([`place_labels`])
//!
//! Predicate vs placement: [`fit_labels_in_regions`] is a one-shot "does this
//! box fit?" check that omits regions where the answer is no. [`place_labels`]
//! always returns a position for every requested region, with an explicit
//! [`PlacementKind`] discriminator so callers know whether the anchor is
//! inside or outside the region. Most consumers want `place_labels`; reach
//! for the predicate only when rolling your own fallback policy.
//!
//! This module is only available when the `plotting` feature is enabled.

#[cfg(feature = "plotting")]
mod clip;
#[cfg(feature = "plotting")]
mod inscribed;
#[cfg(feature = "plotting")]
mod placement;
#[cfg(feature = "plotting")]
mod plot_data;
#[cfg(feature = "plotting")]
mod regions;

#[cfg(feature = "plotting")]
pub use clip::{polygon_clip, polygon_difference, ClipOperation};
#[cfg(feature = "plotting")]
pub use inscribed::{
    fit_label_in_region, fit_labels_in_regions, largest_inscribed_rect, principal_axis,
};
#[cfg(feature = "plotting")]
pub use placement::{
    place_labels, place_labels_to_fixed_point, placements_bbox, ExteriorPolicy, InteriorPolicy,
    LabelPlacement, PlacementError, PlacementKind, PlacementStrategy,
};
#[cfg(feature = "plotting")]
pub use plot_data::{PlotData, PlotOptions};
#[cfg(feature = "plotting")]
pub use regions::{
    classify_into_pieces, decompose_regions, decompose_regions_with, RegionPiece, RegionPolygons,
    DEFAULT_SLIVER_THRESHOLD,
};
