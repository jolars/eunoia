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
use crate::plotting::clip::{polygon_clip_many, ClipOperation};
use crate::plotting::regions::{decompose_regions, poi_with_holes, RegionPolygons};
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
    let shape_outlines: Vec<Polygon> = shapes
        .iter()
        .map(|s| s.polygonize(options.n_vertices))
        .collect();
    let set_anchors =
        compute_set_anchors(&shape_outlines, spec.set_names(), options.label_precision);

    PlotData {
        regions,
        region_anchors,
        set_anchors,
        shape_outlines,
    }
}

/// Computes a label anchor for each set by finding the pole of inaccessibility
/// of `shape_i \ ⋃_{j≠i} shape_j`, using i_overlay's polygon difference. This
/// preserves hole structure (a CCW outer ring with CW hole rings), so when
/// other sets nest inside set `i` they are treated as holes rather than being
/// flattened away — producing a correct "S only" anchor instead of `S`'s
/// geometric centre.
fn compute_set_anchors(
    shape_outlines: &[Polygon],
    set_names: &[String],
    precision: f64,
) -> HashMap<String, Point> {
    let mut result = HashMap::new();
    if shape_outlines.is_empty() {
        return result;
    }

    for (i, name) in set_names.iter().enumerate() {
        if i >= shape_outlines.len() {
            continue;
        }

        // Build the union of all other shapes (one polygon, possibly with
        // disconnected pieces and/or holes — all preserved as separate rings).
        let mut others_union: Vec<Polygon> = Vec::new();
        for (j, other) in shape_outlines.iter().enumerate() {
            if j == i {
                continue;
            }
            if others_union.is_empty() {
                others_union.push(other.clone());
            } else {
                others_union = polygon_clip_many(&others_union, other, ClipOperation::Union);
            }
        }

        // shape_i minus the union of the rest. If others_union is empty
        // (single-set diagram), the result is shape_i itself.
        let exclusive_rings: Vec<Polygon> = if others_union.is_empty() {
            vec![shape_outlines[i].clone()]
        } else {
            let mut acc = vec![shape_outlines[i].clone()];
            for clip in &others_union {
                acc = polygon_clip_many(&acc, clip, ClipOperation::Difference);
                if acc.is_empty() {
                    break;
                }
            }
            acc
        };

        // Try the proper hole-aware POI first.
        let anchor = poi_with_holes(&exclusive_rings, precision).map(|(p, _)| p);

        // Fallback: set is fully covered by others. Use the shape's own POI —
        // not ideal, but the only sensible default with no exclusive area to
        // place a label in.
        let anchor = anchor.or_else(|| {
            // Try non-empty regions containing the set in any combo. This is
            // a last-ditch placeholder; callers can provide their own logic.
            let (p, _) = shape_outlines[i].pole_of_inaccessibility_with_distance(precision);
            Some(p)
        });

        if let Some(point) = anchor {
            result.insert(name.clone(), point);
        }
    }

    result
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
    fn test_clip_difference_with_nested_circle_returns_two_rings() {
        use crate::geometry::primitives::Point;
        use crate::geometry::shapes::Circle;
        use crate::geometry::traits::Polygonize;
        use crate::plotting::clip::{polygon_clip, ClipOperation};

        let a = Circle::new(Point::new(0.0, 0.0), 5.0).polygonize(64);
        let b = Circle::new(Point::new(0.0, 0.0), 1.0).polygonize(32);
        let result = polygon_clip(&a, &b, ClipOperation::Difference);

        let mut pos = 0;
        let mut neg = 0;
        for r in &result {
            let mut s = 0.0;
            let v = r.vertices();
            for i in 0..v.len() {
                let j = (i + 1) % v.len();
                s += v[i].x() * v[j].y() - v[j].x() * v[i].y();
            }
            if s > 0.0 {
                pos += 1;
            } else if s < 0.0 {
                neg += 1;
            }
        }
        eprintln!(
            "polygon_clip(A=r5, B=r1, Difference) returned {} rings: {} CCW (outer), {} CW (hole)",
            result.len(),
            pos,
            neg
        );
        for (i, r) in result.iter().enumerate() {
            let v = r.vertices();
            let mut s = 0.0;
            for j in 0..v.len() {
                let k = (j + 1) % v.len();
                s += v[j].x() * v[k].y() - v[k].x() * v[j].y();
            }
            eprintln!(
                "  ring {}: {} verts, signed_area={:.3}, first vertex=({:.3}, {:.3})",
                i,
                v.len(),
                s / 2.0,
                v[0].x(),
                v[0].y()
            );
        }
        // Expect at least one outer + one hole.
        assert!(pos >= 1, "expected at least one outer (CCW) ring");
        assert!(neg >= 1, "expected at least one hole (CW) ring");
    }

    /// When B, C, D are nested inside A, the set anchor for A must land in
    /// A's exclusive ("A only") lobe — not at A's geometric centre, where
    /// the inner circles overlap it. Regression test for the `eulerr`
    /// comparison reported by users after the eunoia-backend switch.
    #[test]
    fn test_set_anchor_avoids_nested_inner_sets() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 30.0)
            .intersection(&["A", "B"], 3.0)
            .intersection(&["A", "C"], 3.0)
            .intersection(&["A", "D"], 3.0)
            .intersection(&["A", "B", "C"], 0.6)
            .intersection(&["A", "B", "D"], 0.6)
            .intersection(&["A", "C", "D"], 0.6)
            .intersection(&["A", "B", "C", "D"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(1).fit().unwrap();
        let plot = layout.plot_data(&spec, PlotOptions::default());

        let a_anchor = plot.set_anchors.get("A").expect("missing label for A");
        let a_circle = layout.shape_for_set("A").unwrap();

        for inner in ["B", "C", "D"] {
            let c = layout.shape_for_set(inner).unwrap();
            let dx = a_anchor.x() - c.center().x();
            let dy = a_anchor.y() - c.center().y();
            assert!(
                dx * dx + dy * dy > c.radius() * c.radius(),
                "label A at ({:.3}, {:.3}) overlaps inner set {} (center=({:.3}, {:.3}), r={:.3})",
                a_anchor.x(),
                a_anchor.y(),
                inner,
                c.center().x(),
                c.center().y(),
                c.radius(),
            );
        }
        let dx = a_anchor.x() - a_circle.center().x();
        let dy = a_anchor.y() - a_circle.center().y();
        assert!(dx * dx + dy * dy <= a_circle.radius() * a_circle.radius());
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
