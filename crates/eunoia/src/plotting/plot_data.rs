//! Bundled plot data — everything a renderer needs in one struct.
//!
//! [`PlotData`] is the convenience output of [`Layout::plot_data`], gathering
//! region polygons, region/set label anchors, and shape outlines into a
//! single value. This is the recommended entry point for binding authors
//! (R, Python, Julia, web) who want to consume eunoia's geometry in one
//! shot rather than calling four separate methods.
//!
//! [`Layout::plot_data`]: crate::Layout::plot_data

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Polygon;
use crate::geometry::traits::{DiagramShape, Polygonize};
use crate::plotting::regions::{decompose_regions, RegionPolygons};
use crate::spec::{Combination, DiagramSpec};
use std::collections::HashMap;

/// Options controlling [`PlotData`] construction.
///
/// Use [`PlotOptions::default`] for sensible defaults (`n_vertices = 200`,
/// `label_precision = 0.01`).
#[derive(Debug, Clone, Copy)]
pub struct PlotOptions {
    /// Number of vertices used when polygonizing each shape (both for region
    /// decomposition and for the per-set outlines). Higher values produce
    /// smoother edges at the cost of more polygon vertices. Matches eulerr's
    /// `n` argument to `plot.euler`, which defaults to `200`.
    pub n_vertices: usize,

    /// Polylabel precision for label-anchor placement, in the same units as
    /// the polygon coordinates. Smaller values yield more accurate anchors
    /// at higher cost.
    pub label_precision: f64,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            n_vertices: 200,
            label_precision: 0.01,
        }
    }
}

/// Everything a renderer needs to draw a fitted diagram.
///
/// All anchors and polygons live in the same coordinate system as the fitted
/// shapes. See [`crate::Layout::plot_data`] for construction.
#[derive(Debug, Clone)]
pub struct PlotData {
    /// Polygons for every non-empty exclusive region, keyed by set
    /// combination. Use these to fill regions.
    pub regions: RegionPolygons,

    /// One anchor per non-empty region — the pole of inaccessibility of the
    /// region's largest-clearance polygon. Use these to place per-region
    /// labels (e.g. element counts).
    pub region_anchors: HashMap<Combination, Point>,

    /// One anchor per set — the pole of inaccessibility of the largest
    /// connected component of the union of regions containing the set. Use
    /// these to place per-set labels (e.g. set names).
    pub set_anchors: HashMap<String, Point>,

    /// Polygonized outline of each set's shape, in the order of
    /// `spec.set_names()`. Use these to draw set boundaries (edges) directly
    /// from the analytical shape rather than from the unioned regions —
    /// avoids visible seams where exclusive regions meet.
    pub shape_outlines: Vec<Polygon>,
}

pub(crate) fn build_plot_data<S>(shapes: &[S], spec: &DiagramSpec, options: PlotOptions) -> PlotData
where
    S: DiagramShape + Polygonize,
{
    let regions = decompose_regions(shapes, spec.set_names(), spec, options.n_vertices);
    let region_anchors = regions.label_points(options.label_precision);
    let set_anchors = regions.set_label_points(spec.set_names(), options.label_precision);
    let shape_outlines = shapes
        .iter()
        .map(|s| s.polygonize(options.n_vertices))
        .collect();

    PlotData {
        regions,
        region_anchors,
        set_anchors,
        shape_outlines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Circle;
    use crate::spec::{DiagramSpecBuilder, InputType};

    #[test]
    fn test_plot_options_default() {
        let opts = PlotOptions::default();
        assert_eq!(opts.n_vertices, 200);
        assert!((opts.label_precision - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_plot_data_two_circles() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let plot = layout.plot_data(&spec, PlotOptions::default());

        // Region polygons + region anchors should agree on which regions exist.
        for combo in plot.regions.iter().map(|(c, _)| c) {
            assert!(plot.region_anchors.contains_key(combo));
        }
        // One outline per set.
        assert_eq!(plot.shape_outlines.len(), spec.set_names().len());
        // One set anchor per set.
        for name in spec.set_names() {
            assert!(plot.set_anchors.contains_key(name));
        }
    }
}
