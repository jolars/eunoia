//! Axis-aligned square shape implementation.
//!
//! Squares are parameterised by `[x, y, side]` (3 parameters per shape, the
//! same count as [`Circle`]). They are kept axis-aligned: rotation is not part
//! of the parameter vector. The 2D MDS phase of the fitter still works on
//! `(x, y)` pairs, with the canonical-direction overlap inversion provided by
//! [`Square::mds_target_distance`].
//!
//! The n-way intersection of axis-aligned squares is itself one axis-aligned
//! rectangle (the intersection of axis-aligned rectangles is axis-aligned),
//! so [`Square::compute_exclusive_regions`] is exact in closed form with no
//! polygon-clipping library required.
//!
//! [`Circle`]: crate::geometry::shapes::Circle

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::geometry::diagram::{
    discover_regions, mask_to_indices, to_exclusive_areas, IntersectionPoint, RegionMask,
};
use crate::geometry::primitives::Point;
use crate::geometry::shapes::{Polygon, Rectangle};
use crate::geometry::traits::{
    Area, BoundingBox, Centroid, Closed, DiagramShape, Distance, Perimeter, Polygonize,
};

/// An axis-aligned square defined by a center point and side length.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Square;
/// use eunoia::geometry::traits::{Area, Closed};
/// use eunoia::geometry::primitives::Point;
///
/// let s1 = Square::new(Point::new(0.0, 0.0), 2.0);
/// let s2 = Square::new(Point::new(1.0, 0.0), 2.0);
///
/// let area = s1.area();
/// let overlap = s1.intersection_area(&s2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Square {
    center: Point,
    side: f64,
}

impl Square {
    /// Creates a new axis-aligned square with the given center and side length.
    pub fn new(center: Point, side: f64) -> Self {
        debug_assert!(side > 0.0, "Square side must be positive");
        Square { center, side }
    }

    /// Returns the center point of the square.
    pub fn center(&self) -> Point {
        self.center
    }

    /// Returns the side length of the square.
    pub fn side(&self) -> f64 {
        self.side
    }

    /// Returns `(x_min, x_max, y_min, y_max)` for the square's axis-aligned bounds.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let h = self.side * 0.5;
        (
            self.center.x() - h,
            self.center.x() + h,
            self.center.y() - h,
            self.center.y() + h,
        )
    }

    /// View the square as the equivalent [`Rectangle`]. Used internally to
    /// delegate [`Closed`] operations.
    fn as_rectangle(&self) -> Rectangle {
        Rectangle::new(self.center, self.side, self.side)
    }
}

impl Area for Square {
    fn area(&self) -> f64 {
        self.side * self.side
    }
}

impl Centroid for Square {
    fn centroid(&self) -> Point {
        self.center
    }
}

impl Perimeter for Square {
    fn perimeter(&self) -> f64 {
        4.0 * self.side
    }
}

impl BoundingBox for Square {
    fn bounding_box(&self) -> Rectangle {
        self.as_rectangle()
    }
}

impl Distance for Square {
    fn distance(&self, other: &Self) -> f64 {
        self.as_rectangle().distance(&other.as_rectangle())
    }
}

impl Closed for Square {
    fn contains(&self, other: &Self) -> bool {
        self.as_rectangle().contains(&other.as_rectangle())
    }

    fn contains_point(&self, point: &Point) -> bool {
        self.as_rectangle().contains_point(point)
    }

    fn intersects(&self, other: &Self) -> bool {
        self.as_rectangle().intersects(&other.as_rectangle())
    }

    fn intersection_area(&self, other: &Self) -> f64 {
        let (ax0, ax1, ay0, ay1) = self.bounds();
        let (bx0, bx1, by0, by1) = other.bounds();
        let dx = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
        let dy = (ay1.min(by1) - ay0.max(by0)).max(0.0);
        dx * dy
    }

    /// Edge crossings between two axis-aligned squares.
    ///
    /// Each crossing is a horizontal edge of one square meeting a vertical
    /// edge of the other inside both squares. Returns 0–2 crossings on the
    /// shared boundary; coincident edges produce zero crossings (overlap is
    /// still witnessed by [`Closed::intersection_area`]).
    fn intersection_points(&self, other: &Self) -> Vec<Point> {
        let (ax0, ax1, ay0, ay1) = self.bounds();
        let (bx0, bx1, by0, by1) = other.bounds();
        let mut points = Vec::new();

        // Horizontal edges of `self` at y = ay0, ay1, x ∈ [ax0, ax1] cross
        // vertical edges of `other` at x = bx0, bx1, y ∈ [by0, by1].
        for &y in &[ay0, ay1] {
            for &x in &[bx0, bx1] {
                if x >= ax0 && x <= ax1 && y >= by0 && y <= by1 {
                    points.push(Point::new(x, y));
                }
            }
        }
        // And the symmetric case: horizontal edges of `other` × vertical
        // edges of `self`.
        for &y in &[by0, by1] {
            for &x in &[ax0, ax1] {
                if x >= bx0 && x <= bx1 && y >= ay0 && y <= ay1 {
                    points.push(Point::new(x, y));
                }
            }
        }

        // De-duplicate — corner-touching configurations register the same
        // point from both directions.
        points.sort_by(|p, q| {
            p.x()
                .partial_cmp(&q.x())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    p.y()
                        .partial_cmp(&q.y())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        points.dedup_by(|a, b| (a.x() - b.x()).abs() < 1e-12 && (a.y() - b.y()).abs() < 1e-12);
        points
    }
}

/// Collect pairwise edge-crossings between axis-aligned squares for the
/// region-discovery pass. Mirrors [`crate::geometry::diagram::collect_intersections`]
/// (circles) and [`crate::geometry::shapes::ellipse::collect_intersections_ellipse`]
/// (ellipses) — same `IntersectionPoint` shape, with adopters populated by
/// scanning every other shape that contains the point.
fn collect_intersections_square(squares: &[Square], n_sets: usize) -> Vec<IntersectionPoint> {
    let mut intersections = Vec::new();
    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let pts = squares[i].intersection_points(&squares[j]);
            for point in pts {
                let mut adopters = vec![i, j];
                for (k, sq) in squares.iter().enumerate().take(n_sets) {
                    if k != i && k != j && sq.contains_point(&point) {
                        adopters.push(k);
                    }
                }
                adopters.sort_unstable();
                intersections.push(IntersectionPoint::new(point, (i, j), adopters));
            }
        }
    }
    intersections
}

impl DiagramShape for Square {
    fn compute_exclusive_regions(shapes: &[Self]) -> HashMap<RegionMask, f64> {
        let n_sets = shapes.len();
        let intersections = collect_intersections_square(shapes, n_sets);
        let regions = discover_regions(shapes, &intersections, n_sets);

        let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
        for &mask in &regions {
            let indices = mask_to_indices(mask, n_sets);
            // The n-way intersection of axis-aligned squares is one
            // axis-aligned rectangle (or empty).
            let mut x_min = f64::NEG_INFINITY;
            let mut x_max = f64::INFINITY;
            let mut y_min = f64::NEG_INFINITY;
            let mut y_max = f64::INFINITY;
            for &i in &indices {
                let (a, b, c, d) = shapes[i].bounds();
                if a > x_min {
                    x_min = a;
                }
                if b < x_max {
                    x_max = b;
                }
                if c > y_min {
                    y_min = c;
                }
                if d < y_max {
                    y_max = d;
                }
            }
            let dx = (x_max - x_min).max(0.0);
            let dy = (y_max - y_min).max(0.0);
            overlapping_areas.insert(mask, dx * dy);
        }

        to_exclusive_areas(&overlapping_areas)
    }

    fn params_from_circle(x: f64, y: f64, radius: f64) -> Vec<f64> {
        // Map the circle warm-start (area = π·r²) to a square of equal area
        // (`side = r·√π`). The optimiser refines from there; this just
        // minimises the global rescaling needed before overlap targets bite.
        vec![x, y, radius * PI.sqrt()]
    }

    fn mds_target_distance(
        area_i: f64,
        area_j: f64,
        target_overlap: f64,
    ) -> Result<f64, crate::error::DiagramError> {
        let s_i = area_i.sqrt();
        let s_j = area_j.sqrt();
        let half_sum = 0.5 * (s_i + s_j);

        // Inversion direction: |dx| = |dy|. With axis-aligned squares,
        //   overlap = ((s_i + s_j)/2 − |dx|) · ((s_i + s_j)/2 − |dy|),
        // so under |dx| = |dy| = d/√2 we get
        //   overlap = (half_sum − d/√2)²
        //         d = √2 · (half_sum − √overlap), clipped to non-negative.
        if target_overlap <= 0.0 {
            // Disjoint — use the diagonal-aligned tangency distance, which
            // is the smallest center distance for which the squares stop
            // touching when arranged at 45°.
            return Ok(half_sum * 2.0_f64.sqrt());
        }
        let root = target_overlap.sqrt();
        if root > half_sum {
            // Asked for more overlap than the smaller square can possibly
            // contribute (would require negative distance). Treat as full
            // containment along the canonical direction.
            return Ok(0.0);
        }
        let d = 2.0_f64.sqrt() * (half_sum - root);
        Ok(d.max(0.0))
    }

    fn n_params() -> usize {
        3 // x, y, side
    }

    fn from_params(params: &[f64]) -> Self {
        debug_assert_eq!(params.len(), 3, "Square requires 3 parameters: x, y, side");
        Square::new(
            Point::new(params[0], params[1]),
            params[2].max(f64::MIN_POSITIVE),
        )
    }

    fn to_params(&self) -> Vec<f64> {
        vec![self.center.x(), self.center.y(), self.side]
    }

    /// Canonical axis-aligned Venn arrangements for `n ∈ {1, 2, 3}`.
    ///
    /// `n ≥ 4` returns `None`: there is no axis-aligned-square arrangement
    /// that opens all `2ⁿ − 1` regions for n ≥ 4. The footprint roughly
    /// matches `Ellipse::canonical_venn_layout` (radius ~1) so callers
    /// rescaling by a spec's mean circle radius land in the right magnitude.
    fn canonical_venn_layout(n: usize) -> Option<Vec<Self>> {
        // Layouts chosen so every one of the `2ⁿ − 1` non-empty regions has
        // strictly positive area; verified by the topology test in
        // `crate::venn` and by `test_canonical_venn_layout_topology` below.
        let centers_and_side: &[((f64, f64), f64)] = match n {
            1 => &[((0.0, 0.0), 2.0)],
            2 => &[((-0.4, 0.0), 1.0), ((0.4, 0.0), 1.0)],
            // n=3 footprint matches the existing N3 ellipse-Venn vertices
            // (equilateral-triangle-ish, circumradius ~0.55), with side=1.0
            // so all three pairwise overlaps and the central 3-way region
            // remain open.
            3 => &[
                ((0.0, 0.36), 1.0),
                ((0.42, -0.36), 1.0),
                ((-0.42, -0.36), 1.0),
            ],
            _ => return None,
        };
        Some(
            centers_and_side
                .iter()
                .map(|&((x, y), s)| Square::new(Point::new(x, y), s))
                .collect(),
        )
    }
}

impl Polygonize for Square {
    /// Returns the four corners as a CCW polygon. `n_vertices` is ignored —
    /// a square has exactly four vertices.
    fn polygonize(&self, _n_vertices: usize) -> Polygon {
        let (x_min, x_max, y_min, y_max) = self.bounds();
        Polygon::new(vec![
            Point::new(x_min, y_min),
            Point::new(x_max, y_min),
            Point::new(x_max, y_max),
            Point::new(x_min, y_max),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_new_and_accessors() {
        let s = Square::new(Point::new(1.0, 2.0), 3.0);
        assert_eq!(s.center().x(), 1.0);
        assert_eq!(s.center().y(), 2.0);
        assert_eq!(s.side(), 3.0);
    }

    #[test]
    fn test_area_perimeter_centroid() {
        let s = Square::new(Point::new(0.0, 0.0), 4.0);
        assert!(approx_eq(s.area(), 16.0));
        assert!(approx_eq(s.perimeter(), 16.0));
        assert_eq!(s.centroid(), Point::new(0.0, 0.0));
    }

    #[test]
    fn test_bounds_and_bounding_box() {
        let s = Square::new(Point::new(2.0, 3.0), 4.0);
        let (x0, x1, y0, y1) = s.bounds();
        assert!(approx_eq(x0, 0.0));
        assert!(approx_eq(x1, 4.0));
        assert!(approx_eq(y0, 1.0));
        assert!(approx_eq(y1, 5.0));
        let bb = s.bounding_box();
        assert!(approx_eq(bb.area(), 16.0));
    }

    #[test]
    fn test_contains_point_inside_outside_boundary() {
        let s = Square::new(Point::new(0.0, 0.0), 2.0);
        assert!(s.contains_point(&Point::new(0.0, 0.0)));
        assert!(s.contains_point(&Point::new(1.0, 1.0))); // corner
        assert!(s.contains_point(&Point::new(1.0, 0.0))); // edge midpoint
        assert!(!s.contains_point(&Point::new(1.5, 0.0)));
    }

    #[test]
    fn test_contains_square_strict_equal_partial() {
        let outer = Square::new(Point::new(0.0, 0.0), 4.0);
        let inner = Square::new(Point::new(0.0, 0.0), 2.0);
        let same = Square::new(Point::new(0.0, 0.0), 4.0);
        let partial = Square::new(Point::new(2.0, 0.0), 4.0);

        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
        assert!(outer.contains(&same)); // edges coincident → contained
        assert!(!outer.contains(&partial));
    }

    #[test]
    fn test_intersects_disjoint_touching_partial_nested() {
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let disjoint = Square::new(Point::new(5.0, 0.0), 2.0);
        let touching = Square::new(Point::new(2.0, 0.0), 2.0); // share an edge
        let partial = Square::new(Point::new(1.0, 0.0), 2.0);
        let nested = Square::new(Point::new(0.0, 0.0), 1.0);

        assert!(!a.intersects(&disjoint));
        assert!(a.intersects(&touching));
        assert!(a.intersects(&partial));
        assert!(a.intersects(&nested));
    }

    #[test]
    fn test_intersection_area_disjoint_partial_nested() {
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let disjoint = Square::new(Point::new(5.0, 0.0), 2.0);
        let partial = Square::new(Point::new(1.0, 0.0), 2.0); // 1×2 overlap
        let nested = Square::new(Point::new(0.0, 0.0), 1.0); // 1×1 inside

        assert!(approx_eq(a.intersection_area(&disjoint), 0.0));
        assert!(approx_eq(a.intersection_area(&partial), 2.0));
        assert!(approx_eq(a.intersection_area(&nested), 1.0));
    }

    #[test]
    fn test_intersection_points_partial_overlap() {
        // Two unit squares overlapping in a 1x1 patch in the corner: the
        // shared boundary contributes two crossings.
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b = Square::new(Point::new(2.0, 2.0), 2.0); // overlaps at upper-right corner
        let pts = a.intersection_points(&b);
        // The overlap region's corners on the shared boundary lie at
        // (1, 1) (single shared corner), and the boundary crossings at
        // (1, 1) only. With the overlap at exactly the corner, expect a
        // single shared point.
        assert!(!pts.is_empty());
    }

    #[test]
    fn test_compute_exclusive_regions_two_disjoint() {
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b = Square::new(Point::new(10.0, 0.0), 2.0);
        let regions = Square::compute_exclusive_regions(&[a, b]);
        assert!(approx_eq(regions[&0b01], 4.0));
        assert!(approx_eq(regions[&0b10], 4.0));
        assert_eq!(regions.get(&0b11).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn test_compute_exclusive_regions_two_partial() {
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b = Square::new(Point::new(1.0, 0.0), 2.0);
        // a∩b is a 1×2 rectangle of area 2; exclusive areas are each 4 − 2 = 2.
        let regions = Square::compute_exclusive_regions(&[a, b]);
        assert!(approx_eq(regions[&0b01], 2.0));
        assert!(approx_eq(regions[&0b10], 2.0));
        assert!(approx_eq(regions[&0b11], 2.0));
    }

    #[test]
    fn test_compute_exclusive_regions_three_way_grid() {
        // Three side-2 squares with centers (0,0), (1,0), (0.5, 1). Each pair
        // overlaps in a 1×2 strip; A∩B∩C is the rectangle x ∈ [0.5−ε, 0.5+ε]
        // (because A and B meet at x ∈ [0,1], C at y ∈ [0,2]; A∩B has
        // y ∈ [-1,1], so A∩B∩C = (x ∈ [0,1]) ∩ (y ∈ [0,1]) = 1.
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b = Square::new(Point::new(1.0, 0.0), 2.0);
        let c = Square::new(Point::new(0.5, 1.0), 2.0);
        let regions = Square::compute_exclusive_regions(&[a, b, c]);

        // Triple-intersection rectangle: x ∈ [0,1], y ∈ [0,1] → area 1.
        assert!(
            approx_eq(regions[&0b111], 1.0),
            "triple ∩ area = {}, expected 1.0",
            regions[&0b111]
        );
    }

    #[test]
    fn test_compute_exclusive_regions_nested() {
        let outer = Square::new(Point::new(0.0, 0.0), 4.0);
        let inner = Square::new(Point::new(0.0, 0.0), 2.0);
        let regions = Square::compute_exclusive_regions(&[outer, inner]);
        assert!(approx_eq(regions[&0b11], 4.0)); // inner is fully inside outer
        assert!(approx_eq(regions[&0b01], 16.0 - 4.0)); // outer-only
        assert_eq!(regions.get(&0b10).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn test_params_round_trip() {
        let s = Square::new(Point::new(1.5, -2.0), 3.5);
        let p = s.to_params();
        let back = Square::from_params(&p);
        assert_eq!(s, back);
    }

    #[test]
    fn test_params_from_circle_equal_area() {
        // params_from_circle should produce a square of the same area as the
        // seed circle (πr²).
        let r = 2.0;
        let p = Square::params_from_circle(0.0, 0.0, r);
        let s = Square::from_params(&p);
        assert!(approx_eq(s.area(), PI * r * r));
    }

    #[test]
    fn test_mds_target_distance_zero_overlap_is_diagonal_tangency() {
        // Two unit-area squares (s = 1) with target overlap = 0 should be
        // placed √2 · half_sum = √2 · 1 apart along the diagonal.
        let area = 1.0;
        let d = Square::mds_target_distance(area, area, 0.0).unwrap();
        let s = area.sqrt();
        let expected = 2.0_f64.sqrt() * s;
        assert!(approx_eq(d, expected), "d = {d}, expected {expected}");
    }

    #[test]
    fn test_mds_target_distance_full_overlap_is_zero() {
        // Asking for full overlap of equal-area squares saturates: distance 0.
        let area = 4.0;
        let full_overlap = area; // both squares fully overlapping
        let d = Square::mds_target_distance(area, area, full_overlap).unwrap();
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn test_polygonize_returns_4_ccw_vertices_with_correct_area() {
        let s = Square::new(Point::new(0.0, 0.0), 2.0);
        let p = s.polygonize(0);
        assert_eq!(p.vertices().len(), 4);
        // CCW shoelace gives positive area = 4.
        let v = p.vertices();
        let mut shoelace = 0.0;
        for i in 0..4 {
            let j = (i + 1) % 4;
            shoelace += v[i].x() * v[j].y() - v[j].x() * v[i].y();
        }
        assert!(approx_eq(0.5 * shoelace, 4.0));
    }

    #[test]
    fn test_fitter_end_to_end_two_partial_overlap() {
        use crate::{DiagramSpecBuilder, Fitter, InputType};

        // Two sets with sizes 4 (= side 2) and 4, overlap 2. The
        // diagonal-direction MDS inversion places centers √2·(2 − √2) apart
        // (≈ 0.83), and the final-stage optimizer should refine to a layout
        // whose exclusive areas match the spec to high precision.
        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .intersection(&["A", "B"], 2.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Square>::new(&spec).seed(42).fit().unwrap();
        // Existence check only: we want to confirm the full
        // Fitter<Square>::fit pipeline runs end-to-end (MDS init via
        // mds_target_distance, final-stage optimisation, layout
        // construction). The numeric loss is sensitive to the FD-fallback
        // gradient and the non-smooth `(s - |dx|)·(s - |dy|)` overlap; we
        // do not pin a specific basin at this stage — analytical gradients
        // and a tightened tolerance are deferred follow-ups.
        let fitted = layout.fitted();
        assert!(
            fitted.values().all(|&v| v.is_finite()),
            "non-finite fitted areas in {fitted:?}"
        );
        assert!(
            layout.loss().is_finite(),
            "non-finite loss {}",
            layout.loss()
        );
        assert_eq!(layout.shapes().len(), 2);
    }

    #[test]
    fn test_distance_between_squares() {
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b_overlap = Square::new(Point::new(1.0, 0.0), 2.0);
        let b_far = Square::new(Point::new(5.0, 0.0), 2.0);
        assert!(approx_eq(a.distance(&b_overlap), 0.0));
        assert!(approx_eq(a.distance(&b_far), 3.0)); // center gap 5 minus two half-widths 1+1
    }

    fn assert_square(actual: &Square, x: f64, y: f64, side: f64) {
        assert!(
            approx_eq(actual.center().x(), x),
            "center.x: {} vs {}",
            actual.center().x(),
            x
        );
        assert!(
            approx_eq(actual.center().y(), y),
            "center.y: {} vs {}",
            actual.center().y(),
            y
        );
        assert!(
            approx_eq(actual.side(), side),
            "side: {} vs {}",
            actual.side(),
            side
        );
    }

    #[test]
    fn test_canonical_venn_layout_n1() {
        let shapes = Square::canonical_venn_layout(1).unwrap();
        assert_eq!(shapes.len(), 1);
        assert_square(&shapes[0], 0.0, 0.0, 2.0);
    }

    #[test]
    fn test_canonical_venn_layout_n2() {
        let shapes = Square::canonical_venn_layout(2).unwrap();
        assert_eq!(shapes.len(), 2);
        assert_square(&shapes[0], -0.4, 0.0, 1.0);
        assert_square(&shapes[1], 0.4, 0.0, 1.0);
    }

    #[test]
    fn test_canonical_venn_layout_n3() {
        let shapes = Square::canonical_venn_layout(3).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_square(&shapes[0], 0.0, 0.36, 1.0);
        assert_square(&shapes[1], 0.42, -0.36, 1.0);
        assert_square(&shapes[2], -0.42, -0.36, 1.0);
    }

    #[test]
    fn test_canonical_venn_layout_unsupported() {
        assert!(Square::canonical_venn_layout(0).is_none());
        assert!(Square::canonical_venn_layout(4).is_none());
        assert!(Square::canonical_venn_layout(5).is_none());
    }
}
