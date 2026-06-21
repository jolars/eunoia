//! Oriented (rotated) rectangle shape implementation.
//!
//! `RotatedRectangle` is the rotation-bearing sibling of
//! [`Rectangle`] — the same width/height
//! box plus a rotation angle `φ`, exactly as [`Ellipse`] extends
//! [`Circle`](crate::geometry::shapes::Circle). It is a fittable
//! [`DiagramShape`].
//!
//! Unlike the axis-aligned [`Rectangle`], the overlap of two oriented boxes is
//! a general convex polygon, so areas are computed by **convex polygon
//! clipping** (Sutherland–Hodgman) plus the shoelace formula rather than a
//! closed-form AABB shortcut. The resulting area is *exact* but only
//! piecewise-C¹ (it has kinks where the contact combinatorics change), so this
//! shape deliberately does **not** provide analytical gradients
//! (`compute_exclusive_regions_with_gradient` stays `None`) and is fitted with
//! derivative-free optimizers. See `SUPPORTS_ANALYTIC_GRADIENT` on
//! [`DiagramShape`] and the capability-driven default pool in
//! [`Fitter`](crate::fitter::Fitter).
//!
//! Two parameter encodings are exposed, mirroring `Rectangle` with `φ`
//! appended unchanged:
//!
//! - **Geometric** (`to_params` / `from_params`): `[x, y, w, h, φ]`.
//! - **Optimizer** (`to_optimizer_params` / `from_optimizer_params`):
//!   `[x, y, ln(w·h), ln(w/h), φ]`. The log-space area/aspect split is the
//!   same conditioning trick `Rectangle` uses; `φ` passes through unconstrained
//!   (periodic, like `Ellipse`).
//!
//! [`Ellipse`]: crate::geometry::shapes::Ellipse

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::geometry::diagram::{
    IntersectionPoint, RegionMask, discover_regions, mask_to_indices, to_exclusive_areas,
};
use crate::geometry::primitives::{Bounds, Point};
use crate::geometry::shapes::Polygon;
use crate::geometry::shapes::Rectangle;
use crate::geometry::shapes::polygon::shoelace_area;
use crate::geometry::traits::{
    Area, BoundingBox, Centroid, Closed, DiagramShape, Distance, Perimeter, Polygonize,
};

/// Tolerance for boundary-inclusive point containment, in diagram coordinates.
/// Loose enough to absorb the round-trip floating-point error of the local-frame
/// rotation (so `r.contains(&r)` holds), tight enough not to merge distinct
/// shapes at the ~unit scale the fitter works in.
const CONTAINS_EPS: f64 = 1e-9;

/// An oriented rectangle defined by a center, width, height, and rotation.
///
/// The rotation is in radians, counterclockwise from the +x axis, and applies
/// to the local frame whose axes are the width (local x) and height (local y).
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::RotatedRectangle;
/// use eunoia::geometry::traits::{Area, Closed};
/// use eunoia::geometry::primitives::Point;
///
/// let r1 = RotatedRectangle::new(Point::new(0.0, 0.0), 4.0, 2.0, 0.0);
/// let r2 = RotatedRectangle::new(Point::new(1.0, 0.0), 2.0, 2.0, std::f64::consts::FRAC_PI_4);
///
/// let area1 = r1.area();
/// let overlap = r1.intersection_area(&r2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRectangle {
    center: Point,
    width: f64,
    height: f64,
    rotation: f64,
}

#[allow(dead_code)]
impl RotatedRectangle {
    /// Creates a new oriented rectangle.
    ///
    /// `rotation` is in radians, counterclockwise from the +x axis. Like
    /// [`Rectangle::new`], this does not validate the extents; use
    /// [`RotatedRectangle::try_new`] for untrusted input.
    pub fn new(center: Point, width: f64, height: f64, rotation: f64) -> Self {
        RotatedRectangle {
            center,
            width,
            height,
            rotation,
        }
    }

    /// Fallible constructor: returns
    /// [`crate::error::DiagramError::InvalidShapeParameter`] when `width <= 0`
    /// or `height <= 0`. Use this when constructing from untrusted input (e.g.
    /// across an FFI boundary).
    pub fn try_new(
        center: Point,
        width: f64,
        height: f64,
        rotation: f64,
    ) -> Result<Self, crate::error::DiagramError> {
        if width <= 0.0 {
            return Err(crate::error::DiagramError::InvalidShapeParameter {
                shape: "RotatedRectangle",
                param: "width",
                value: width,
            });
        }
        if height <= 0.0 {
            return Err(crate::error::DiagramError::InvalidShapeParameter {
                shape: "RotatedRectangle",
                param: "height",
                value: height,
            });
        }
        Ok(RotatedRectangle {
            center,
            width,
            height,
            rotation,
        })
    }

    /// Returns a reference to the rectangle's center point.
    pub fn center(&self) -> &Point {
        &self.center
    }

    /// Returns the rectangle's width (extent along the local x axis).
    pub fn width(&self) -> f64 {
        self.width
    }

    /// Returns the rectangle's height (extent along the local y axis).
    pub fn height(&self) -> f64 {
        self.height
    }

    /// Returns the rotation angle in radians.
    pub fn rotation(&self) -> f64 {
        self.rotation
    }

    /// Sets the center of the rectangle.
    pub fn set_center(&mut self, center: Point) {
        self.center = center;
    }

    /// Returns the four corner points in counterclockwise order, in world
    /// coordinates (the local CCW ring `(−,−), (+,−), (+,+), (−,+)` rotated by
    /// `φ` about the center). A proper rotation preserves orientation, so the
    /// result is convex and CCW — the form the clipper and shoelace expect.
    pub fn corners(&self) -> [Point; 4] {
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let hw = self.width / 2.0;
        let hh = self.height / 2.0;
        let local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
        local.map(|(lx, ly)| {
            Point::new(
                self.center.x() + lx * cos - ly * sin,
                self.center.y() + lx * sin + ly * cos,
            )
        })
    }
}

impl Area for RotatedRectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
}

impl Perimeter for RotatedRectangle {
    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }
}

impl BoundingBox for RotatedRectangle {
    /// Axis-aligned bounds of the rotated box. The extreme corners give
    /// half-extents `hw·|cos| + hh·|sin|` (x) and `hw·|sin| + hh·|cos|` (y) —
    /// the rectangle analog of the ellipse bounding-box projection (which uses
    /// an L2 sum because its boundary is smooth, where a box's corners give an
    /// L1 sum).
    fn bounds(&self) -> Bounds {
        let cos = self.rotation.cos().abs();
        let sin = self.rotation.sin().abs();
        let hw = self.width / 2.0;
        let hh = self.height / 2.0;
        let half_width = hw * cos + hh * sin;
        let half_height = hw * sin + hh * cos;
        Bounds::new(
            self.center.x() - half_width,
            self.center.x() + half_width,
            self.center.y() - half_height,
            self.center.y() + half_height,
        )
    }
}

impl Centroid for RotatedRectangle {
    fn centroid(&self) -> Point {
        self.center
    }
}

impl Distance for RotatedRectangle {
    /// Minimum distance between the two boundaries, `0.0` if they overlap or
    /// touch. For disjoint convex boxes the edges do not cross, so the minimum
    /// is attained at an endpoint and reduces to the smallest
    /// vertex-to-edge distance over all edge pairs.
    fn distance(&self, other: &Self) -> f64 {
        if self.intersects(other) {
            return 0.0;
        }
        let a = self.corners();
        let b = other.corners();
        let mut min_d = f64::INFINITY;
        for i in 0..4 {
            let a1 = a[i];
            let a2 = a[(i + 1) % 4];
            for j in 0..4 {
                let b1 = b[j];
                let b2 = b[(j + 1) % 4];
                min_d = min_d.min(segment_segment_distance(a1, a2, b1, b2));
            }
        }
        min_d
    }
}

#[allow(dead_code)]
impl Closed for RotatedRectangle {
    /// True iff `other` lies entirely within `self`. Both boxes are convex, so
    /// this holds exactly when every corner of `other` is inside `self`.
    fn contains(&self, other: &Self) -> bool {
        other.corners().iter().all(|p| self.contains_point(p))
    }

    fn contains_point(&self, point: &Point) -> bool {
        // Rotate the offset into the box's local frame (inverse rotation R(−φ),
        // matching `Ellipse::contains_point`), then axis-aligned check.
        let dx = point.x() - self.center.x();
        let dy = point.y() - self.center.y();
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let x_local = dx * cos + dy * sin;
        let y_local = -dx * sin + dy * cos;
        x_local.abs() <= self.width / 2.0 + CONTAINS_EPS
            && y_local.abs() <= self.height / 2.0 + CONTAINS_EPS
    }

    fn intersects(&self, other: &Self) -> bool {
        let a = self.corners();
        let b = other.corners();
        !(has_separating_axis(&a, &b) || has_separating_axis(&b, &a))
    }

    /// Exact overlap area via convex polygon clipping + shoelace.
    fn intersection_area(&self, other: &Self) -> f64 {
        let clipped = clip_convex(&self.corners(), &other.corners());
        shoelace_area(&clipped)
    }

    /// Vertices of the overlap polygon (the convex region `self ∩ other`),
    /// or empty when the boxes are disjoint or merely touch. These are the
    /// candidate points fed to region discovery.
    fn intersection_points(&self, other: &Self) -> Vec<Point> {
        let clipped = clip_convex(&self.corners(), &other.corners());
        if shoelace_area(&clipped) <= CONTAINS_EPS {
            return vec![];
        }
        clipped
    }
}

// ---------------------------------------------------------------------------
// Convex-polygon geometry helpers (private to this module).
// ---------------------------------------------------------------------------

/// Signed twice-area of triangle `(a, b, p)`; positive when `p` is left of the
/// directed edge `a → b`.
#[inline]
fn cross(a: Point, b: Point, p: Point) -> f64 {
    (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x())
}

/// Intersection of the segment `p1 → p2` with the infinite line through
/// `a → b`. Returns `None` when (near-)parallel.
fn line_segment_intersection(p1: Point, p2: Point, a: Point, b: Point) -> Option<Point> {
    let dx = p2.x() - p1.x();
    let dy = p2.y() - p1.y();
    let sx = b.x() - a.x();
    let sy = b.y() - a.y();
    let denom = dx * sy - dy * sx;
    if denom.abs() < 1e-15 {
        return None;
    }
    let t = ((a.x() - p1.x()) * sy - (a.y() - p1.y()) * sx) / denom;
    Some(Point::new(p1.x() + t * dx, p1.y() + t * dy))
}

/// Sutherland–Hodgman clip of convex `subject` against convex CCW `clip`,
/// returning the (convex) intersection ring. Both inputs must be
/// counterclockwise; the result may be empty (no overlap) or degenerate
/// (touching), in which case [`shoelace_area`] yields `0.0`.
fn clip_convex(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }
    let mut output = subject.to_vec();
    let n = clip.len();
    for e in 0..n {
        if output.is_empty() {
            break;
        }
        let a = clip[e];
        let b = clip[(e + 1) % n];
        let input = std::mem::take(&mut output);
        let m = input.len();
        for j in 0..m {
            let prev = input[(j + m - 1) % m];
            let cur = input[j];
            let cur_in = cross(a, b, cur) >= 0.0;
            let prev_in = cross(a, b, prev) >= 0.0;
            if cur_in {
                if !prev_in && let Some(ip) = line_segment_intersection(prev, cur, a, b) {
                    output.push(ip);
                }
                output.push(cur);
            } else if prev_in && let Some(ip) = line_segment_intersection(prev, cur, a, b) {
                output.push(ip);
            }
        }
    }
    output
}

/// Separating-axis test: returns true if some edge normal of `poly` separates
/// `poly` from `other`. Touching (zero-gap) projections are *not* separated,
/// matching [`Rectangle`]'s touching-counts-as-intersecting convention.
fn has_separating_axis(poly: &[Point], other: &[Point]) -> bool {
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        // Outward normal of edge a → b (unnormalized; consistent across both
        // projections so magnitude is irrelevant).
        let nx = -(b.y() - a.y());
        let ny = b.x() - a.x();
        let (mut min1, mut max1) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in poly {
            let d = p.x() * nx + p.y() * ny;
            min1 = min1.min(d);
            max1 = max1.max(d);
        }
        let (mut min2, mut max2) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in other {
            let d = p.x() * nx + p.y() * ny;
            min2 = min2.min(d);
            max2 = max2.max(d);
        }
        if max1 < min2 || max2 < min1 {
            return true;
        }
    }
    false
}

/// Distance from point `p` to segment `a → b`.
fn point_segment_distance(p: Point, a: Point, b: Point) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len2 = dx * dx + dy * dy;
    if len2 <= f64::MIN_POSITIVE {
        return p.distance(&a);
    }
    let t = (((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / len2).clamp(0.0, 1.0);
    let proj = Point::new(a.x() + t * dx, a.y() + t * dy);
    p.distance(&proj)
}

/// Distance between two segments that are known not to cross (the disjoint
/// convex case): the minimum of the four endpoint-to-segment distances.
fn segment_segment_distance(a1: Point, a2: Point, b1: Point, b2: Point) -> f64 {
    point_segment_distance(a1, b1, b2)
        .min(point_segment_distance(a2, b1, b2))
        .min(point_segment_distance(b1, a1, a2))
        .min(point_segment_distance(b2, a1, a2))
}

/// Area of the convex intersection of the boxes selected by `indices`, via
/// iterated clipping (the intersection of convex sets is convex, so each clip
/// keeps the running polygon convex). Empty `indices` ⇒ `0.0`.
fn intersection_area_of(shapes: &[RotatedRectangle], indices: &[usize]) -> f64 {
    let Some((&first, rest)) = indices.split_first() else {
        return 0.0;
    };
    let mut poly = shapes[first].corners().to_vec();
    for &i in rest {
        poly = clip_convex(&poly, &shapes[i].corners());
        if poly.is_empty() {
            return 0.0;
        }
    }
    shoelace_area(&poly)
}

/// Pairwise overlap vertices for the region-discovery pass. Mirrors
/// [`Rectangle`]'s `collect_intersections_rectangle`: each overlap vertex is
/// tagged with every box that contains it.
fn collect_intersections_rotated_rectangle(
    rects: &[RotatedRectangle],
    n_sets: usize,
) -> Vec<IntersectionPoint> {
    let mut intersections = Vec::new();
    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let pts = rects[i].intersection_points(&rects[j]);
            for point in pts {
                let mut adopters = vec![i, j];
                for (k, r) in rects.iter().enumerate().take(n_sets) {
                    if k != i && k != j && r.contains_point(&point) {
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

/// Clipped exclusive areas against an axis-aligned `container`. Mask `0` is
/// seeded with the container area; inclusion-exclusion then yields
/// `complement = container.area − area(⋃ rectangles ∩ container)`. The
/// container participates as one more convex clip polygon.
pub(crate) fn compute_exclusive_regions_clipped_rotated_rectangles(
    shapes: &[RotatedRectangle],
    container: &Rectangle,
) -> HashMap<RegionMask, f64> {
    let n_sets = shapes.len();
    let intersections = collect_intersections_rotated_rectangle(shapes, n_sets);
    let regions = discover_regions(shapes, &intersections, n_sets);

    let container_poly = container.corners().to_vec();
    let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
    overlapping_areas.insert(0, container.area());

    for &mask in &regions {
        let indices = mask_to_indices(mask, n_sets);
        let mut poly = container_poly.clone();
        for &i in &indices {
            poly = clip_convex(&poly, &shapes[i].corners());
            if poly.is_empty() {
                break;
            }
        }
        overlapping_areas.insert(mask, shoelace_area(&poly));
    }

    to_exclusive_areas(&overlapping_areas)
}

impl Polygonize for RotatedRectangle {
    /// Returns the four rotated corners as a CCW polygon. `n_vertices` is
    /// ignored — a rectangle has exactly four vertices.
    fn polygonize(&self, _n_vertices: usize) -> Polygon {
        Polygon::new(self.corners().to_vec())
    }
}

impl DiagramShape for RotatedRectangle {
    fn compute_exclusive_regions(shapes: &[Self]) -> HashMap<RegionMask, f64> {
        let n_sets = shapes.len();
        let intersections = collect_intersections_rotated_rectangle(shapes, n_sets);
        let regions = discover_regions(shapes, &intersections, n_sets);

        let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
        for &mask in &regions {
            let indices = mask_to_indices(mask, n_sets);
            overlapping_areas.insert(mask, intersection_area_of(shapes, &indices));
        }

        to_exclusive_areas(&overlapping_areas)
    }

    fn optimizer_params_from_circle(x: f64, y: f64, radius: f64) -> Vec<f64> {
        // Map the circle warm-start (area = π·r²) to a square of equal area.
        // In optimizer encoding: u = ln(π·r²), v = 0 (w = h), φ = 0.
        let u = PI.ln() + 2.0 * radius.ln();
        vec![x, y, u, 0.0, 0.0]
    }

    fn mds_target_distance(
        area_i: f64,
        area_j: f64,
        target_overlap: f64,
    ) -> Result<f64, crate::error::DiagramError> {
        // Same equal-area-square inversion as Rectangle: MDS only fixes 2D
        // positions; aspect ratio and rotation are recovered by the final
        // stage.
        let s_i = area_i.sqrt();
        let s_j = area_j.sqrt();
        let half_sum = 0.5 * (s_i + s_j);

        if target_overlap <= 0.0 {
            return Ok(half_sum * 2.0_f64.sqrt());
        }
        let root = target_overlap.sqrt();
        if root > half_sum {
            return Ok(0.0);
        }
        let d = 2.0_f64.sqrt() * (half_sum - root);
        Ok(d.max(0.0))
    }

    fn n_params() -> usize {
        5 // x, y, w, h, φ (geometric); x, y, ln(area), ln(ratio), φ (optimizer)
    }

    fn from_params(params: &[f64]) -> Self {
        debug_assert_eq!(
            params.len(),
            5,
            "RotatedRectangle requires 5 parameters: x, y, width, height, rotation"
        );
        RotatedRectangle::new(
            Point::new(params[0], params[1]),
            params[2].max(f64::MIN_POSITIVE),
            params[3].max(f64::MIN_POSITIVE),
            params[4],
        )
    }

    fn to_params(&self) -> Vec<f64> {
        vec![
            self.center.x(),
            self.center.y(),
            self.width,
            self.height,
            self.rotation,
        ]
    }

    fn from_optimizer_params(params: &[f64]) -> Self {
        debug_assert_eq!(
            params.len(),
            5,
            "RotatedRectangle optimizer params: x, y, ln(area), ln(ratio), rotation"
        );
        let u = params[2];
        let v = params[3];
        let w = ((u + v) * 0.5).exp();
        let h = ((u - v) * 0.5).exp();
        let w = if w.is_finite() && w > 0.0 {
            w
        } else {
            f64::MIN_POSITIVE
        };
        let h = if h.is_finite() && h > 0.0 {
            h
        } else {
            f64::MIN_POSITIVE
        };
        RotatedRectangle::new(Point::new(params[0], params[1]), w, h, params[4])
    }

    fn to_optimizer_params(&self) -> Vec<f64> {
        let u = (self.width * self.height).ln();
        let v = (self.width / self.height).ln();
        vec![self.center.x(), self.center.y(), u, v, self.rotation]
    }

    fn compute_exclusive_regions_clipped(
        shapes: &[Self],
        container: &Rectangle,
    ) -> Option<HashMap<RegionMask, f64>> {
        Some(compute_exclusive_regions_clipped_rotated_rectangles(
            shapes, container,
        ))
    }

    // `compute_exclusive_regions_with_gradient` and
    // `compute_exclusive_regions_clipped_with_gradient` intentionally use the
    // trait default (`None`): the exact convex-clip area is only piecewise-C¹,
    // so this shape is fitted derivative-free (see module docs).

    /// Canonical Venn arrangements for `n ∈ {1, 2, 3}`, reusing
    /// [`Rectangle`]'s axis-aligned (φ = 0) square footprints. `n ≥ 4` returns
    /// `None`: no equal-box arrangement opens all `2ⁿ − 1` regions for `n ≥ 4`.
    fn canonical_venn_layout(n: usize) -> Option<Vec<Self>> {
        let centers_and_side: &[((f64, f64), f64)] = match n {
            1 => &[((0.0, 0.0), 2.0)],
            2 => &[((-0.4, 0.0), 1.0), ((0.4, 0.0), 1.0)],
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
                .map(|&((x, y), s)| RotatedRectangle::new(Point::new(x, y), s, s, 0.0))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    const EPS: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn area_and_perimeter() {
        let r = RotatedRectangle::new(Point::new(0.0, 0.0), 4.0, 3.0, 0.7);
        assert!(approx(r.area(), 12.0));
        assert!(approx(r.perimeter(), 14.0));
    }

    #[test]
    fn corners_unrotated_match_axis_aligned() {
        let r = RotatedRectangle::new(Point::new(0.0, 0.0), 2.0, 2.0, 0.0);
        let c = r.corners();
        let expected = [
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(-1.0, 1.0),
        ];
        for (got, want) in c.iter().zip(expected.iter()) {
            assert!(approx(got.x(), want.x()) && approx(got.y(), want.y()));
        }
    }

    #[test]
    fn bounds_90_degrees_swaps_extents() {
        // A 4×2 box rotated 90° has a 2×4 axis-aligned bounding box.
        let r = RotatedRectangle::new(Point::new(0.0, 0.0), 4.0, 2.0, FRAC_PI_2);
        let b = r.bounds();
        assert!(approx(b.x_max - b.x_min, 2.0));
        assert!(approx(b.y_max - b.y_min, 4.0));
    }

    #[test]
    fn bounds_45_degrees() {
        // Unit square rotated 45° → bounding box half-extent = √2/2 + √2/2 = √2.
        let r = RotatedRectangle::new(Point::new(0.0, 0.0), 1.0, 1.0, std::f64::consts::FRAC_PI_4);
        let b = r.bounds();
        let diag = 2.0_f64.sqrt();
        assert!(approx(b.x_max - b.x_min, diag));
        assert!(approx(b.y_max - b.y_min, diag));
    }

    #[test]
    fn contains_point_rotated_frame() {
        let r = RotatedRectangle::new(Point::new(0.0, 0.0), 4.0, 2.0, FRAC_PI_2);
        // After 90° rotation, the long axis is vertical.
        assert!(r.contains_point(&Point::new(0.0, 1.9)));
        assert!(!r.contains_point(&Point::new(1.9, 0.0)));
        assert!(r.contains_point(&Point::new(0.9, 0.0)));
    }

    #[test]
    fn contains_self_and_smaller() {
        let big = RotatedRectangle::new(Point::new(0.0, 0.0), 4.0, 4.0, 0.6);
        let small = RotatedRectangle::new(Point::new(0.0, 0.0), 1.0, 1.0, 0.6);
        assert!(big.contains(&big));
        assert!(big.contains(&small));
        assert!(!small.contains(&big));
    }

    // ---- φ = 0 must match Rectangle exactly (the key regression guard) ----

    fn rect(cx: f64, cy: f64, w: f64, h: f64) -> Rectangle {
        Rectangle::new(Point::new(cx, cy), w, h)
    }
    fn rrect(cx: f64, cy: f64, w: f64, h: f64) -> RotatedRectangle {
        RotatedRectangle::new(Point::new(cx, cy), w, h, 0.0)
    }

    #[test]
    fn unrotated_intersection_area_matches_rectangle() {
        let cases = [
            ((0.0, 0.0, 4.0, 4.0), (2.0, 0.0, 4.0, 4.0)),
            ((0.0, 0.0, 10.0, 10.0), (1.0, 0.0, 4.0, 4.0)),
            ((0.0, 0.0, 2.0, 2.0), (10.0, 0.0, 2.0, 2.0)),
            ((0.0, 0.0, 4.0, 3.0), (1.5, 0.5, 3.0, 2.0)),
        ];
        for (a, b) in cases {
            let ra = rect(a.0, a.1, a.2, a.3);
            let rb = rect(b.0, b.1, b.2, b.3);
            let rra = rrect(a.0, a.1, a.2, a.3);
            let rrb = rrect(b.0, b.1, b.2, b.3);
            assert!(
                approx(rra.intersection_area(&rrb), ra.intersection_area(&rb)),
                "mismatch for {a:?} vs {b:?}: {} != {}",
                rra.intersection_area(&rrb),
                ra.intersection_area(&rb)
            );
        }
    }

    #[test]
    fn unrotated_exclusive_regions_match_rectangle() {
        let rects = [
            rect(0.0, 0.0, 4.0, 4.0),
            rect(2.0, 0.0, 4.0, 4.0),
            rect(1.0, 2.0, 4.0, 4.0),
        ];
        let rrects = [
            rrect(0.0, 0.0, 4.0, 4.0),
            rrect(2.0, 0.0, 4.0, 4.0),
            rrect(1.0, 2.0, 4.0, 4.0),
        ];
        let want = Rectangle::compute_exclusive_regions(&rects);
        let got = RotatedRectangle::compute_exclusive_regions(&rrects);
        assert_eq!(want.len(), got.len());
        for (mask, w) in &want {
            assert!(
                approx(*got.get(mask).unwrap_or(&f64::NAN), *w),
                "region {mask:b}: {:?} != {w}",
                got.get(mask)
            );
        }
    }

    // ---- rotated clip sanity ----

    #[test]
    fn diamond_inside_square_full_overlap() {
        // Unit-ish square fully containing a 45°-rotated small square.
        let big = RotatedRectangle::new(Point::new(0.0, 0.0), 10.0, 10.0, 0.0);
        let diamond =
            RotatedRectangle::new(Point::new(0.0, 0.0), 2.0, 2.0, std::f64::consts::FRAC_PI_4);
        assert!(approx(big.intersection_area(&diamond), diamond.area()));
    }

    #[test]
    fn disjoint_rotated_boxes_zero_overlap() {
        let a = RotatedRectangle::new(Point::new(0.0, 0.0), 2.0, 2.0, 0.5);
        let b = RotatedRectangle::new(Point::new(10.0, 0.0), 2.0, 2.0, 0.5);
        assert!(!a.intersects(&b));
        assert!(approx(a.intersection_area(&b), 0.0));
        assert!(a.distance(&b) > 0.0);
    }

    #[test]
    fn rotation_invariance_of_overlap() {
        // Rotating both boxes and their relative position by the same angle
        // about the shared center leaves the overlap area unchanged.
        let theta = 0.9_f64;
        let a0 = RotatedRectangle::new(Point::new(0.0, 0.0), 4.0, 2.0, 0.0);
        let b0 = RotatedRectangle::new(Point::new(1.5, 0.0), 3.0, 2.0, 0.0);
        let base = a0.intersection_area(&b0);
        let (c, s) = (theta.cos(), theta.sin());
        let rot = |p: &Point| Point::new(p.x() * c - p.y() * s, p.x() * s + p.y() * c);
        let a1 = RotatedRectangle::new(rot(a0.center()), 4.0, 2.0, theta);
        let b1 = RotatedRectangle::new(rot(b0.center()), 3.0, 2.0, theta);
        assert!(approx(a1.intersection_area(&b1), base));
    }

    // ---- parameter round-trips ----

    #[test]
    fn geometric_params_round_trip() {
        let r = RotatedRectangle::new(Point::new(1.5, -2.0), 3.0, 4.0, 0.7);
        let p = r.to_params();
        assert_eq!(p, vec![1.5, -2.0, 3.0, 4.0, 0.7]);
        assert_eq!(RotatedRectangle::from_params(&p), r);
    }

    #[test]
    fn optimizer_params_round_trip() {
        let r = RotatedRectangle::new(Point::new(1.5, -2.0), 3.0, 4.0, -0.4);
        let p = r.to_optimizer_params();
        let back = RotatedRectangle::from_optimizer_params(&p);
        assert!(approx(back.center().x(), 1.5));
        assert!(approx(back.center().y(), -2.0));
        assert!(approx(back.width(), 3.0));
        assert!(approx(back.height(), 4.0));
        assert!(approx(back.rotation(), -0.4));
    }

    #[test]
    fn optimizer_params_from_circle_equal_area() {
        let radius = 2.0;
        let p = RotatedRectangle::optimizer_params_from_circle(0.0, 0.0, radius);
        assert!(approx(p[2], (PI * radius * radius).ln()));
        assert!(approx(p[3], 0.0));
        assert!(approx(p[4], 0.0));
    }

    #[test]
    fn try_new_rejects_nonpositive() {
        assert!(RotatedRectangle::try_new(Point::new(0.0, 0.0), 0.0, 1.0, 0.0).is_err());
        assert!(RotatedRectangle::try_new(Point::new(0.0, 0.0), 1.0, -1.0, 0.0).is_err());
        assert!(RotatedRectangle::try_new(Point::new(0.0, 0.0), 1.0, 1.0, 0.3).is_ok());
    }

    #[test]
    fn canonical_venn_layouts() {
        for n in 1..=3 {
            let layout = RotatedRectangle::canonical_venn_layout(n).unwrap();
            assert_eq!(layout.len(), n);
            assert!(layout.iter().all(|r| r.rotation() == 0.0));
        }
        assert!(RotatedRectangle::canonical_venn_layout(4).is_none());
    }

    #[test]
    fn clipped_complement_runs() {
        let shapes = [rrect(-0.5, 0.0, 1.0, 1.0), rrect(0.5, 0.0, 1.0, 1.0)];
        let container = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let regions =
            RotatedRectangle::compute_exclusive_regions_clipped(&shapes, &container).unwrap();
        // Complement (mask 0) present and equals container minus the union.
        let complement = regions[&0];
        let union: f64 = regions
            .iter()
            .filter(|(m, _)| **m != 0)
            .map(|(_, a)| a)
            .sum();
        assert!(approx(complement + union, container.area()));
    }

    #[test]
    fn capability_flag_is_derivative_free() {
        // RotatedRectangle has no analytic gradient (piecewise-C¹ overlap), so
        // it must report a different capability than the gradient-providing
        // Rectangle — that difference is what routes the two to different
        // default optimizer pools. Concretely RotatedRectangle returns `None`
        // from the gradient method.
        assert_ne!(
            RotatedRectangle::SUPPORTS_ANALYTIC_GRADIENT,
            Rectangle::SUPPORTS_ANALYTIC_GRADIENT
        );
        assert!(RotatedRectangle::compute_exclusive_regions_with_gradient(&[]).is_none());
    }

    // ---- end-to-end fits (derivative-free pool) ----

    #[test]
    fn fit_two_set_partial_overlap() {
        use crate::{DiagramSpecBuilder, Fitter, InputType};

        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .intersection(&["A", "B"], 2.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<RotatedRectangle>::new(&spec)
            .seed(7)
            .fit()
            .unwrap();
        assert_eq!(layout.shapes().len(), 2);
        assert!(layout.loss().is_finite() && layout.loss() < 0.05);
        assert!(layout.fitted().values().all(|v| v.is_finite()));
    }

    #[test]
    fn fit_three_set() {
        use crate::{DiagramSpecBuilder, Fitter, InputType};

        let spec = DiagramSpecBuilder::new()
            .set("A", 6.0)
            .set("B", 6.0)
            .set("C", 6.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["A", "C"], 2.0)
            .intersection(&["B", "C"], 2.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<RotatedRectangle>::new(&spec)
            .seed(7)
            .fit()
            .unwrap();
        assert_eq!(layout.shapes().len(), 3);
        assert!(layout.loss().is_finite());
        assert!(layout.fitted().values().all(|v| v.is_finite()));
    }

    #[test]
    fn fit_with_complement_runs_derivative_free() {
        use crate::{DiagramSpecBuilder, Fitter, InputType};

        let spec = DiagramSpecBuilder::new()
            .set("A", 3.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .complement(10.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<RotatedRectangle>::new(&spec)
            .seed(7)
            .fit()
            .unwrap();
        assert_eq!(layout.shapes().len(), 2);
        assert!(layout.loss().is_finite());
    }
}
