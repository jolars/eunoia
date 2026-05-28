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

mod clip;
mod inscribed;
mod placement;
mod plot_data;
mod regions;

pub use clip::{ClipOperation, polygon_clip, polygon_difference, polygon_union_many};
pub use inscribed::{fit_label_in_region, largest_inscribed_rect, principal_axis};
pub use placement::{
    ElbowOptions, ExteriorPolicy, LabelPlacement, LeaderStrategy, PlacementKind, PlacementStrategy,
    TetherSource, place_labels, place_labels_to_fixed_point, placements_bbox,
};
pub use plot_data::{PlotData, PlotOptions};
pub use regions::{
    DEFAULT_SLIVER_THRESHOLD, RegionPiece, RegionPolygons, classify_into_pieces, decompose_regions,
    decompose_regions_with,
};
