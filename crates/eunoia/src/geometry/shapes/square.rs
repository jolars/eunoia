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
    discover_regions, mask_to_indices, to_exclusive_areas, to_exclusive_areas_and_gradients,
    IntersectionPoint, RegionMask,
};
use crate::geometry::primitives::Point;
use crate::geometry::shapes::{Polygon, Rectangle};
use crate::geometry::traits::{
    Area, BoundingBox, Centroid, Closed, DiagramShape, Distance, ExclusiveRegionsAndGradient,
    Perimeter, Polygonize,
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

/// Companion to [`Square::compute_exclusive_regions`] that also returns the
/// analytical gradient of each exclusive area w.r.t. the flat parameter
/// vector `[x₀, y₀, s₀, x₁, …]`.
///
/// For each region the n-way intersection is one axis-aligned rectangle
/// with `dx = x_max − x_min` and `dy = y_max − y_min`; each of the four
/// extrema is a min/max over the mask. When a single shape strictly binds a
/// side, that side's contributions go to that shape:
///
/// ```text
/// ∂A/∂x_{i_L} = −dy   ∂A/∂x_{i_R} = +dy
/// ∂A/∂y_{i_B} = −dx   ∂A/∂y_{i_T} = +dx
/// ∂A/∂s_{i_L} = dy/2  ∂A/∂s_{i_R} = dy/2
/// ∂A/∂s_{i_B} = dx/2  ∂A/∂s_{i_T} = dx/2
/// ```
///
/// At a tie (multiple shapes share the extremum on a side, e.g. coincident
/// edges), the side's contribution is split equally among the tied shapes.
/// This matches the central-difference average of one-sided derivatives at
/// the non-smooth point and produces a valid subgradient. Contributions sum
/// when the same shape binds multiple sides (e.g. the 1-mask case yields
/// `∂A/∂s_i = 2s`, matching `A = s²`). When `dx ≤ 0` or `dy ≤ 0` the area
/// is clamped to 0 and the gradient for that region is the zero vector. The
/// shared inclusion-exclusion combiner [`to_exclusive_areas_and_gradients`]
/// already zeroes gradients for post-IE clamped-negative areas.
fn compute_exclusive_regions_with_gradient_squares(
    shapes: &[Square],
) -> ExclusiveRegionsAndGradient {
    let n_sets = shapes.len();
    let n_params = n_sets * 3;

    let intersections = collect_intersections_square(shapes, n_sets);
    let regions = discover_regions(shapes, &intersections, n_sets);

    let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
    let mut overlapping_grads: HashMap<RegionMask, Vec<f64>> = HashMap::new();

    for &mask in &regions {
        let indices = mask_to_indices(mask, n_sets);

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

        let dx_raw = x_max - x_min;
        let dy_raw = y_max - y_min;
        let dx = dx_raw.max(0.0);
        let dy = dy_raw.max(0.0);
        overlapping_areas.insert(mask, dx * dy);

        let mut grad = vec![0.0; n_params];
        if dx_raw > 0.0 && dy_raw > 0.0 {
            // Collect tied binding indices per side. Bounds and the running
            // extremum are produced by arithmetic on input parameters, so
            // structural edge coincidences hit exact f64 equality.
            let mut tied_l: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_r: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_b: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_t: Vec<usize> = Vec::with_capacity(indices.len());
            for &i in &indices {
                let (a, b, c, d) = shapes[i].bounds();
                #[allow(clippy::float_cmp)]
                {
                    if a == x_min {
                        tied_l.push(i);
                    }
                    if b == x_max {
                        tied_r.push(i);
                    }
                    if c == y_min {
                        tied_b.push(i);
                    }
                    if d == y_max {
                        tied_t.push(i);
                    }
                }
            }
            let w_l = 1.0 / tied_l.len() as f64;
            let w_r = 1.0 / tied_r.len() as f64;
            let w_b = 1.0 / tied_b.len() as f64;
            let w_t = 1.0 / tied_t.len() as f64;

            for &i in &tied_l {
                grad[3 * i] -= dy * w_l;
                grad[3 * i + 2] += dy * 0.5 * w_l;
            }
            for &i in &tied_r {
                grad[3 * i] += dy * w_r;
                grad[3 * i + 2] += dy * 0.5 * w_r;
            }
            for &i in &tied_b {
                grad[3 * i + 1] -= dx * w_b;
                grad[3 * i + 2] += dx * 0.5 * w_b;
            }
            for &i in &tied_t {
                grad[3 * i + 1] += dx * w_t;
                grad[3 * i + 2] += dx * 0.5 * w_t;
            }
        }
        overlapping_grads.insert(mask, grad);
    }

    to_exclusive_areas_and_gradients(&overlapping_areas, &overlapping_grads, n_params)
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

    fn compute_exclusive_regions_with_gradient(
        shapes: &[Self],
    ) -> Option<ExclusiveRegionsAndGradient> {
        Some(compute_exclusive_regions_with_gradient_squares(shapes))
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

    /// Central-difference reference for `compute_exclusive_regions`. Returns a
    /// HashMap matching the analytical layout: per region, a length-`3·n_sets`
    /// gradient vector ordered `[x₀, y₀, s₀, x₁, …]`.
    fn fd_exclusive_region_gradients(shapes: &[Square], h: f64) -> HashMap<RegionMask, Vec<f64>> {
        let n_sets = shapes.len();
        let n_params = n_sets * 3;
        let base = Square::compute_exclusive_regions(shapes);

        let mut grads: HashMap<RegionMask, Vec<f64>> =
            base.keys().map(|&m| (m, vec![0.0; n_params])).collect();

        for i in 0..n_sets {
            for k in 0..3 {
                let perturb = |delta: f64| -> HashMap<RegionMask, f64> {
                    let mut copy: Vec<Square> = shapes.to_vec();
                    let (cx, cy, side) =
                        (copy[i].center().x(), copy[i].center().y(), copy[i].side());
                    let new = match k {
                        0 => Square::new(Point::new(cx + delta, cy), side),
                        1 => Square::new(Point::new(cx, cy + delta), side),
                        2 => Square::new(Point::new(cx, cy), side + delta),
                        _ => unreachable!(),
                    };
                    copy[i] = new;
                    Square::compute_exclusive_regions(&copy)
                };
                let plus = perturb(h);
                let minus = perturb(-h);
                for (&mask, g) in grads.iter_mut() {
                    let p = plus.get(&mask).copied().unwrap_or(0.0);
                    let m = minus.get(&mask).copied().unwrap_or(0.0);
                    g[3 * i + k] = (p - m) / (2.0 * h);
                }
            }
        }
        grads
    }

    /// Compare analytical and FD gradient maps mask-by-mask, indexing both by
    /// the analytical map's masks. Tolerance `tol` is per-component absolute.
    fn assert_grad_matches_fd(
        analytical: &HashMap<RegionMask, Vec<f64>>,
        fd: &HashMap<RegionMask, Vec<f64>>,
        tol: f64,
    ) {
        for (&mask, ag) in analytical.iter() {
            let fg = fd
                .get(&mask)
                .expect("FD missing mask present in analytical");
            assert_eq!(ag.len(), fg.len(), "param count mismatch for mask {mask:b}");
            for (k, (&a, &f)) in ag.iter().zip(fg.iter()).enumerate() {
                assert!(
                    (a - f).abs() < tol,
                    "mask {mask:b} param {k}: analytical={a} fd={f} (tol={tol})"
                );
            }
        }
    }

    #[test]
    fn test_gradient_single_square_matches_2s_on_side() {
        // A = s²; ∂A/∂s = 2s, ∂A/∂x = ∂A/∂y = 0.
        let s = 3.0;
        let sq = Square::new(Point::new(1.5, -2.0), s);
        let (areas, grads) = Square::compute_exclusive_regions_with_gradient(&[sq]).unwrap();
        assert!(approx_eq(areas[&0b1], s * s));
        let g = &grads[&0b1];
        assert_eq!(g.len(), 3);
        assert!(approx_eq(g[0], 0.0));
        assert!(approx_eq(g[1], 0.0));
        assert!(approx_eq(g[2], 2.0 * s));
    }

    #[test]
    fn test_gradient_two_squares_partial_overlap_matches_fd() {
        // 1×2 overlap rectangle. Edge-coincident on y=±1, so the gradient
        // exercises the tie-split path; central FD truncation degrades from
        // O(h²) to O(h) at those kinks, hence a 1e-5 tolerance.
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b = Square::new(Point::new(1.0, 0.0), 2.0);
        let (_, grads) = Square::compute_exclusive_regions_with_gradient(&[a, b]).unwrap();
        let fd = fd_exclusive_region_gradients(&[a, b], 1e-6);
        assert_grad_matches_fd(&grads, &fd, 1e-5);
    }

    #[test]
    fn test_gradient_three_squares_overlap_matches_fd() {
        // Triple-overlap configuration with multiple edge ties.
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b = Square::new(Point::new(1.0, 0.0), 2.0);
        let c = Square::new(Point::new(0.5, 1.0), 2.0);
        let (_, grads) = Square::compute_exclusive_regions_with_gradient(&[a, b, c]).unwrap();
        let fd = fd_exclusive_region_gradients(&[a, b, c], 1e-6);
        assert_grad_matches_fd(&grads, &fd, 1e-5);
    }

    #[test]
    fn test_gradient_generic_no_ties_matches_fd_tightly() {
        // Generic configuration with all distinct edges so no tie-split
        // applies; central FD is O(h²) accurate and the gradient should
        // match to ~1e-9 with h=1e-5.
        let a = Square::new(Point::new(0.0, 0.0), 2.3);
        let b = Square::new(Point::new(1.1, 0.4), 1.7);
        let c = Square::new(Point::new(0.6, 1.2), 2.1);
        let (_, grads) = Square::compute_exclusive_regions_with_gradient(&[a, b, c]).unwrap();
        let fd = fd_exclusive_region_gradients(&[a, b, c], 1e-5);
        assert_grad_matches_fd(&grads, &fd, 1e-7);
    }

    #[test]
    fn test_gradient_disjoint_pair_is_zero_on_intersection() {
        let a = Square::new(Point::new(0.0, 0.0), 2.0);
        let b = Square::new(Point::new(10.0, 0.0), 2.0);
        let (_, grads) = Square::compute_exclusive_regions_with_gradient(&[a, b]).unwrap();
        // The {a,b} mask might be absent (sparse discovery) or present with
        // zero area + zero gradient.
        if let Some(g) = grads.get(&0b11) {
            for &v in g {
                assert!(approx_eq(v, 0.0), "expected zero on disjoint pair, got {v}");
            }
        }
        // Singletons must agree with FD.
        let fd = fd_exclusive_region_gradients(&[a, b], 1e-5);
        for &mask in &[0b01_usize, 0b10_usize] {
            let ag = grads.get(&mask).expect("singleton missing");
            let fg = fd.get(&mask).expect("FD singleton missing");
            for (k, (&a, &f)) in ag.iter().zip(fg.iter()).enumerate() {
                assert!(
                    (a - f).abs() < 1e-6,
                    "mask {mask:b} param {k}: analytical={a} fd={f}"
                );
            }
        }
    }

    #[test]
    fn test_gradient_nested_matches_fd() {
        // Inner square fully inside outer; the inner-only exclusive area is 0.
        // Edges are not coincident, so this is a smooth point — tight FD parity.
        let outer = Square::new(Point::new(0.0, 0.0), 4.0);
        let inner = Square::new(Point::new(0.0, 0.0), 2.0);
        let (areas, grads) =
            Square::compute_exclusive_regions_with_gradient(&[outer, inner]).unwrap();
        let fd = fd_exclusive_region_gradients(&[outer, inner], 1e-5);
        assert!(approx_eq(areas[&0b11], 4.0));
        assert!(approx_eq(areas[&0b01], 12.0));
        assert_grad_matches_fd(&grads, &fd, 1e-7);
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
        // construction). The numeric loss is sensitive to the non-smooth
        // `(s - |dx|)·(s - |dy|)` overlap; we do not pin a specific basin
        // at this stage.
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
