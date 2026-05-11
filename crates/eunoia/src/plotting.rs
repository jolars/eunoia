//! Polygon-based plotting utilities.
//!
//! This module provides utilities for converting fitted diagram shapes into
//! polygons and decomposing them into exclusive regions for visualization.
//!
//! ## Features
//!
//! - Polygon clipping operations (intersection, union, difference, xor)
//! - Region decomposition for exclusive area visualization
//! - Label-anchor utilities (poles of inaccessibility, principal axis,
//!   inscribed rectangles)
//! - Label placement ([`place_labels`]) — every requested region gets a
//!   position back, interior when the label fits inside the polygon,
//!   exterior otherwise, with a `tether` so callers can draw a leader line
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
pub use clip::{polygon_clip, polygon_difference, polygon_union_many, ClipOperation};
#[cfg(feature = "plotting")]
pub use inscribed::{fit_label_in_region, largest_inscribed_rect, principal_axis};
#[cfg(feature = "plotting")]
pub use placement::{
    place_labels, place_labels_to_fixed_point, placements_bbox, ExteriorPolicy, LabelPlacement,
    PlacementKind, PlacementStrategy, TetherSource,
};
#[cfg(feature = "plotting")]
pub use plot_data::{PlotData, PlotOptions};
#[cfg(feature = "plotting")]
pub use regions::{
    classify_into_pieces, decompose_regions, decompose_regions_with, RegionPiece, RegionPolygons,
    DEFAULT_SLIVER_THRESHOLD,
};
