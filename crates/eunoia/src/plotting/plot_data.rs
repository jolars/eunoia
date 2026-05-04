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
use crate::plotting::regions::{
    classify_into_pieces, decompose_regions, poi_with_holes, RegionPolygons,
};
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
/// All anchors and polygons live in the same coordinate system as the
/// fitted shapes (no normalisation is applied — the renderer chooses the
/// transform). See [`crate::Layout::plot_data`] for construction and
/// [`crate::plotting::RegionPiece`] for the rendering contract on
/// `regions`.
#[derive(Debug, Clone)]
pub struct PlotData {
    /// Per-region pieces (outer + holes per connected component) keyed by
    /// set combination. Renderers should fill each piece with the SVG /
    /// Canvas default `fill-rule: nonzero`; orientations are normalised
    /// for that — see [`crate::plotting::RegionPiece`].
    pub regions: RegionPolygons,

    /// Hole-aware label anchor for every non-empty region (one anchor per
    /// region, even when the region is fragmented). Computed by
    /// [`RegionPolygons::label_points`] — the pole of inaccessibility of
    /// the highest-clearance piece. Use these for per-region labels such
    /// as element counts.
    pub region_anchors: HashMap<Combination, Point>,
    /// Label anchor for every set, with the eulerr-style fallback chain:
    ///
    /// 1. **Hole-aware POI of `shape_i \ ⋃_{j≠i} shape_j`** — the natural
    ///    "S only" lobe, accounting for nested sets as holes.
    /// 2. **Largest containing region's anchor** — for sets with no
    ///    exclusive area (e.g. B fully nested in A → label lands in
    ///    `A&B`), so the label still appears inside the set's actual
    ///    coverage.
    /// 3. **Shape's own POI** — last-resort default; reachable only when
    ///    the set is geometrically empty.
    ///
    /// Use these for per-set labels (e.g. set names). For the un-fallback'd
    /// version that only considers regions, see
    /// [`RegionPolygons::set_label_points`].
    pub set_anchors: HashMap<String, Point>,

    /// Polygonised outline of each set's shape, in the order of
    /// `spec.set_names()`. Use these to draw set boundaries (edges)
    /// directly from the analytical shape rather than from the unioned
    /// regions — avoids visible seams where exclusive regions meet under
    /// stroke rendering.
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
    let set_anchors = compute_set_anchors(
        &shape_outlines,
        spec.set_names(),
        &regions,
        &region_anchors,
        options.label_precision,
    );

    PlotData {
        regions,
        region_anchors,
        set_anchors,
        shape_outlines,
    }
}

/// Computes a label anchor for each set by finding the pole of inaccessibility
/// of `shape_i \ ⋃_{j≠i} shape_j`, using i_overlay's polygon difference and
/// the same piece classifier the region decomposition uses. So an "A only"
/// region with B/C/D as holes gets a hole-aware POI in the donut, and a
/// fully-nested B falls back to the largest containing region (e.g. `A&B`).
///
/// Sliver filtering is handled inside the classifier-and-clipper pipeline:
/// pieces below the relative-area threshold are rejected during piece
/// construction, so callers don't need a separate threshold here.
fn compute_set_anchors(
    shape_outlines: &[Polygon],
    set_names: &[String],
    regions: &RegionPolygons,
    region_anchors: &HashMap<Combination, Point>,
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

        // shape_i minus the union of the rest, then classify the resulting
        // ring set into outer-with-holes pieces. If `others_union` is empty
        // (single-set diagram) the result is just `shape_i`.
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

        let pieces = classify_into_pieces(exclusive_rings);

        // Drop pieces that are tiny relative to the shape itself — these are
        // polygonization seam artifacts when the set is geometrically nested
        // inside another, and would otherwise drag the label onto a sliver.
        let shape_area = shape_outlines[i].area();
        let max_piece_area = pieces.iter().map(|p| p.area()).fold(0.0_f64, f64::max);
        let kept: Vec<_> = if shape_area > 0.0 && max_piece_area < shape_area * 1e-3 {
            Vec::new()
        } else {
            pieces.into_iter().filter(|p| p.area() > 0.0).collect()
        };

        // Hole-aware POI on the (possibly hole-bearing) exclusive pieces.
        let anchor = poi_with_holes(&kept, precision).map(|(p, _)| p);

        // Fallback chain: largest-area containing region (eulerr behaviour
        // for nested sets), then the shape's own POI as a last resort.
        let anchor =
            anchor.or_else(|| largest_containing_region_anchor(name, regions, region_anchors));
        let anchor = anchor.or_else(|| {
            let (p, _) = shape_outlines[i].pole_of_inaccessibility_with_distance(precision);
            Some(p)
        });

        if let Some(point) = anchor {
            result.insert(name.clone(), point);
        }
    }

    result
}

/// Picks the anchor of the largest-area region whose combination contains
/// `name`, or `None` if the set never appears in any non-empty region.
/// "Largest area" sums the absolute polygon areas of every piece in the
/// region — matches the JS `fallbackSetLabels` heuristic this replaces.
fn largest_containing_region_anchor(
    name: &str,
    regions: &RegionPolygons,
    region_anchors: &HashMap<Combination, Point>,
) -> Option<Point> {
    let mut best: Option<(&Combination, f64)> = None;
    for (combo, polys) in regions.iter() {
        if !combo.sets().iter().any(|s| s == name) {
            continue;
        }
        let area: f64 = polys.iter().map(|p| p.area()).sum();
        if best.map(|(_, a)| area > a).unwrap_or(true) {
            best = Some((combo, area));
        }
    }
    best.and_then(|(combo, _)| region_anchors.get(combo).copied())
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
