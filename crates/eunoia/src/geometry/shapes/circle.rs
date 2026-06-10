//! Circle shape implementation.

use std::f64::consts::PI;

use crate::error::DiagramError;
use crate::geometry::diagram::IntersectionPoint;
use crate::geometry::primitives::Point;
use crate::geometry::primitives::point;
use crate::geometry::shapes::{Polygon, Rectangle};
use crate::geometry::traits::{
    Area, BoundingBox, Centroid, Closed, DiagramShape, Distance, Perimeter, Polygonize,
};

/// A circle defined by a center point and radius.
///
/// Circles are the simplest shape for Euler and Venn diagrams and are often
/// sufficient for many use cases. They have the advantage of being rotationally
/// symmetric, which simplifies some computations.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::geometry::traits::{Area, Closed};
/// use eunoia::geometry::primitives::Point;
///
/// let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
/// let c2 = Circle::new(Point::new(3.0, 0.0), 1.0);
///
/// let area1 = c1.area();
/// let overlap = c1.intersection_area(&c2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    center: Point,
    radius: f64,
}

impl Area for Circle {
    /// Computes the area of the circle using the formula A = πr².
    fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }
}

impl Centroid for Circle {
    fn centroid(&self) -> Point {
        self.center
    }
}

impl Distance for Circle {
    /// Computes the minimum distance between the boundaries of two circles.
    ///
    /// Returns 0.0 if the circles overlap or touch.
    fn distance(&self, other: &Self) -> f64 {
        let center_distance = self.center.distance(&other.center);
        let radius_sum = self.radius + other.radius;

        if center_distance > radius_sum {
            center_distance - radius_sum
        } else {
            0.0
        }
    }
}

impl Perimeter for Circle {
    /// Compute the perimeter of the circle.
    fn perimeter(&self) -> f64 {
        2.0 * PI * self.radius
    }
}

impl BoundingBox for Circle {
    fn bounding_box(&self) -> Rectangle {
        let width = 2.0 * self.radius;
        let height = 2.0 * self.radius;

        Rectangle::new(self.center, width, height)
    }
}

impl Closed for Circle {
    fn contains(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        center_distance + other.radius <= self.radius
    }

    fn contains_point(&self, point: &Point) -> bool {
        let dist = self.center.distance(point);
        dist <= self.radius
    }

    /// Checks if two circles intersect (share any common points).
    ///
    /// Note: This implementation returns `true` if circles are separate,
    /// which appears to be inverted from the typical definition. This may
    /// need correction.
    fn intersects(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        // Circles intersect if distance is less than sum of radii
        // Also check for containment case (distance < |r1 - r2|)
        center_distance < self.radius + other.radius
            && center_distance > (self.radius - other.radius).abs()
    }

    /// Computes the area of intersection between two circles.
    ///
    /// Uses the standard geometric formula for circle-circle intersection:
    /// - Returns 0 if circles don't overlap
    /// - Returns area of smaller circle if one contains the other
    /// - Otherwise computes the lens-shaped intersection area
    ///
    /// # Algorithm
    ///
    /// For two circles with radii r1 and r2 separated by distance d, the
    /// intersection area is computed using the formula involving circular
    /// segments from both circles.
    fn intersection_area(&self, other: &Self) -> f64 {
        let d = self.center.distance(&other.center);

        if d >= self.radius + other.radius {
            return 0.0; // No intersection
        }

        if d <= (self.radius - other.radius).abs() {
            // One circle is completely inside the other
            let smaller_radius = self.radius.min(other.radius);
            return PI * smaller_radius * smaller_radius;
        }

        let r1 = self.radius;
        let r2 = other.radius;

        let part1 = r1 * r1 * (((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos());
        let part2 = r2 * r2 * (((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos());
        let part3 = 0.5 * ((r1 + r2 - d) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).sqrt();

        part1 + part2 - part3
    }

    /// Computes the points of intersection between two circles.
    fn intersection_points(&self, other: &Self) -> Vec<Point> {
        let d = self.center.distance(&other.center);

        if d > self.radius + other.radius || d < (self.radius - other.radius).abs() {
            return vec![]; // No intersection points
        }

        let a = (self.radius * self.radius - other.radius * other.radius + d * d) / (2.0 * d);
        let h = (self.radius * self.radius - a * a).sqrt();

        let p2_x = self.center.x() + a * (other.center.x() - self.center.x()) / d;
        let p2_y = self.center.y() + a * (other.center.y() - self.center.y()) / d;

        let rx = -(other.center.y() - self.center.y()) * (h / d);
        let ry = (other.center.x() - self.center.x()) * (h / d);

        let intersection1 = Point::new(p2_x + rx, p2_y + ry);
        let intersection2 = Point::new(p2_x - rx, p2_y - ry);

        if intersection1 == intersection2 {
            vec![intersection1]
        } else {
            vec![intersection1, intersection2]
        }
    }
}

impl DiagramShape for Circle {
    fn compute_exclusive_regions(
        shapes: &[Self],
    ) -> std::collections::HashMap<crate::geometry::diagram::RegionMask, f64> {
        crate::geometry::diagram::compute_exclusive_regions(shapes)
    }

    fn optimizer_params_from_circle(x: f64, y: f64, radius: f64) -> Vec<f64> {
        vec![x, y, radius]
    }

    fn mds_target_distance(
        area_i: f64,
        area_j: f64,
        target_overlap: f64,
    ) -> Result<f64, crate::error::DiagramError> {
        let r1 = (area_i / PI).sqrt();
        let r2 = (area_j / PI).sqrt();
        distance_for_overlap(r1, r2, target_overlap, None, None).map_err(|_| {
            crate::error::DiagramError::InvalidCombination(format!(
                "Could not compute target distance for areas {area_i} / {area_j} \
                 and target overlap {target_overlap}"
            ))
        })
    }

    fn n_params() -> usize {
        3 // x, y, radius
    }

    fn from_params(params: &[f64]) -> Self {
        debug_assert_eq!(
            params.len(),
            3,
            "Circle requires 3 parameters: x, y, radius"
        );
        // Floor at `f64::MIN_POSITIVE` so the optimizer's exploratory steps
        // (which can briefly propose `radius <= 0`) still produce a valid
        // shape — `Circle::new` asserts `radius > 0`. Mirrors the
        // `Square::from_params` and `Ellipse::from_optimizer_params` clamps.
        Circle::new(
            Point::new(params[0], params[1]),
            params[2].max(f64::MIN_POSITIVE),
        )
    }

    fn to_params(&self) -> Vec<f64> {
        vec![self.center.x(), self.center.y(), self.radius]
    }

    fn compute_exclusive_regions_with_gradient(
        shapes: &[Self],
    ) -> Option<crate::geometry::traits::ExclusiveRegionsAndGradient> {
        Some(crate::geometry::diagram::compute_exclusive_regions_with_gradient_circles(shapes))
    }

    fn compute_exclusive_regions_clipped(
        shapes: &[Self],
        container: &Rectangle,
    ) -> Option<std::collections::HashMap<crate::geometry::diagram::RegionMask, f64>> {
        Some(compute_exclusive_regions_clipped(shapes, container))
    }

    fn compute_exclusive_regions_clipped_with_gradient(
        shapes: &[Self],
        container: &Rectangle,
    ) -> Option<crate::geometry::traits::ExclusiveRegionsAndGradient> {
        Some(compute_exclusive_regions_clipped_with_gradient(
            shapes, container,
        ))
    }

    /// Canonical circle Venn arrangements for `n ∈ {1, 2, 3}` — the classic
    /// one-, two-, and three-circle diagrams.
    ///
    /// `n ≥ 4` returns `None`: equal circles cannot open all `2ⁿ − 1`
    /// regions beyond three sets (the standard obstruction that motivates
    /// the Venn ellipse layouts for `n ∈ {4, 5}`). For `n = 3` the circles
    /// sit on an equilateral triangle whose circumradius (`0.45`) is smaller
    /// than the radius (`0.6`), so every circle covers the centroid (opening
    /// the 3-way region) while each pairwise lens still pokes outside the
    /// third circle (opening the three 2-way regions). The footprint is
    /// kept near radius ~1 to match the other shapes' canonical layouts.
    fn canonical_venn_layout(n: usize) -> Option<Vec<Self>> {
        // Layouts chosen so every one of the `2ⁿ − 1` non-empty regions has
        // strictly positive area; verified by `test_topology_is_true_venn_circle`
        // in `crate::venn` and by `test_canonical_venn_layout_*` below.
        let centers_and_radius: &[((f64, f64), f64)] = match n {
            1 => &[((0.0, 0.0), 1.0)],
            2 => &[((-0.35, 0.0), 0.6), ((0.35, 0.0), 0.6)],
            // Equilateral triangle, circumradius 0.45, radius 0.6 (vertices at
            // 90°, 210°, 330°). x = ±0.45·cos(30°) ≈ ±0.389711, y = {0.45, −0.225}.
            3 => &[
                ((0.0, 0.45), 0.6),
                ((-0.389711, -0.225), 0.6),
                ((0.389711, -0.225), 0.6),
            ],
            _ => return None,
        };
        Some(
            centers_and_radius
                .iter()
                .map(|&((x, y), r)| Circle::new(Point::new(x, y), r))
                .collect(),
        )
    }
}

impl Polygonize for Circle {
    fn polygonize(&self, n_vertices: usize) -> Polygon {
        use std::f64::consts::PI;

        let n_vertices = n_vertices.max(3);
        let mut vertices = Vec::with_capacity(n_vertices);

        for i in 0..n_vertices {
            let angle = 2.0 * PI * (i as f64) / (n_vertices as f64);
            let x = self.center.x() + self.radius * angle.cos();
            let y = self.center.y() + self.radius * angle.sin();
            vertices.push(Point::new(x, y));
        }

        Polygon::new(vertices)
    }
}

struct SeparationCost {
    r1: f64,
    r2: f64,
    target_overlap: f64,
    /// Lower / upper of the feasible centre-distance bracket, stored so
    /// `basin::Brent` can read them via [`basin::BoxConstraints`].
    lower: f64,
    upper: f64,
}

impl basin::CostFunction for SeparationCost {
    type Param = f64;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, distance: &f64) -> Result<f64, std::convert::Infallible> {
        let c1 = Circle::new(Point::new(0.0, 0.0), self.r1);
        let c2 = Circle::new(Point::new(*distance, 0.0), self.r2);

        let current_overlap = c1.intersection_area(&c2);
        Ok((current_overlap - self.target_overlap).powi(2))
    }
}

impl basin::BoxConstraints for SeparationCost {
    fn lower(&self) -> &f64 {
        &self.lower
    }

    fn upper(&self) -> &f64 {
        &self.upper
    }
}

impl Circle {
    /// Creates a new circle with the specified center and radius.
    ///
    /// # Arguments
    ///
    /// * `center` - The center point of the circle
    /// * `radius` - The radius of the circle (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `radius <= 0`. Use [`Circle::try_new`] to handle invalid
    /// input as a [`DiagramError`] instead of a panic — bindings authors
    /// writing FFI wrappers should reach for `try_new`.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Circle;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let circle = Circle::new(Point::new(1.0, 2.0), 3.0);
    /// ```
    pub fn new(center: Point, radius: f64) -> Self {
        assert!(radius > 0.0, "Circle radius must be > 0, got {radius}");
        Circle { center, radius }
    }

    /// Fallible constructor: returns
    /// [`DiagramError::InvalidShapeParameter`] when `radius <= 0` instead of
    /// panicking. Use this when constructing circles from untrusted input
    /// (e.g. across an FFI boundary).
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Circle;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// assert!(Circle::try_new(Point::new(0.0, 0.0), 1.0).is_ok());
    /// assert!(Circle::try_new(Point::new(0.0, 0.0), 0.0).is_err());
    /// assert!(Circle::try_new(Point::new(0.0, 0.0), -1.0).is_err());
    /// ```
    pub fn try_new(center: Point, radius: f64) -> Result<Self, DiagramError> {
        if radius > 0.0 {
            Ok(Circle { center, radius })
        } else {
            Err(DiagramError::InvalidShapeParameter {
                shape: "Circle",
                param: "radius",
                value: radius,
            })
        }
    }

    /// Returns a reference to the circle's center point.
    pub fn center(&self) -> &Point {
        &self.center
    }

    /// Returns the circle's radius.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Sets the center of the circle.
    pub fn set_center(&mut self, center: Point) {
        self.center = center;
    }

    /// Circular sector area given the radius and angle in radians.
    pub fn sector_area(&self, angle_rad: f64) -> f64 {
        0.5 * self.radius * self.radius * angle_rad
    }

    /// Circular segment area from radius and angle
    pub fn segment_area_from_angle(&self, angle_rad: f64) -> f64 {
        0.5 * self.radius * self.radius * (angle_rad - angle_rad.sin())
    }

    /// Circular segment area from radius and chord length
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `chord_length > 2 * radius` (impossible geometry).
    /// In release builds, this validation is skipped for performance.
    pub fn segment_area_from_chord(&self, chord_length: f64) -> f64 {
        let r = self.radius;
        debug_assert!(
            chord_length <= 2.0 * r,
            "Chord length {} cannot exceed diameter {}",
            chord_length,
            2.0 * r
        );
        let theta = 2.0 * (chord_length / (2.0 * r)).asin();
        self.segment_area_from_angle(theta)
    }

    pub fn segment_area_from_points(&self, p1: &Point, p2: &Point) -> f64 {
        let chord_length = p1.distance(p2);
        self.segment_area_from_chord(chord_length)
    }
}

/// Computes the distance required between two circles to achieve a specified overlap area.
pub(crate) fn distance_for_overlap(
    r1: f64,
    r2: f64,
    overlap: f64,
    tol: Option<f64>,
    max_iter: Option<u64>,
) -> Result<f64, DiagramError> {
    let min_distance = (r1 - r2).abs();
    let max_distance = r1 + r2;

    // If the desired overlap is zero, then the circles should
    // at most be touching.
    if overlap <= 0.0 {
        return Ok(max_distance);
    }

    let cost_fun = SeparationCost {
        r1,
        r2,
        target_overlap: overlap,
        lower: min_distance,
        upper: max_distance,
    };

    // `tol` is the relative x-tolerance on the centre distance. The default
    // `sqrt(machine epsilon)` matches eulerr; `Brent::new` uses the same
    // relative tolerance with a `1e-12` absolute floor. basin's `Brent` reads
    // the `[min_distance, max_distance]` bracket from `BoxConstraints`, seeded
    // at the bracket midpoint. `state.param` tracks the incumbent minimiser,
    // so the final value is the best distance found.
    let solver = match tol {
        Some(t) => basin::Brent::with_tol(t, 1e-12),
        None => basin::Brent::new(),
    };
    let x0 = 0.5 * (min_distance + max_distance);
    let result = basin::Executor::new(cost_fun, solver, basin::ScalarState::new(x0))
        .max_iter(max_iter.unwrap_or(1000))
        .run()
        .expect("solver problem is infallible");

    Ok(*result.param())
}

#[deprecated(
    since = "0.3.1",
    note = "Returns wrong area when one circle in the mask contains the others' lens. Use `crate::geometry::diagram::compute_exclusive_regions` or the boundary-arc helpers (`region_boundary_arcs` + `area_from_boundary_arcs`) instead."
)]
pub fn multiple_overlap_areas(circles: &[Circle], points: &[IntersectionPoint]) -> f64 {
    let n_circles = circles.len();

    // Filter to only points that are in ALL circles (full intersection)
    let full_intersection_points: Vec<&IntersectionPoint> = points
        .iter()
        .filter(|ip| ip.adopters().len() == n_circles)
        .collect();

    if full_intersection_points.is_empty() {
        return 0.0;
    }

    let n_points = full_intersection_points.len();

    // Sort the points by their angles around the centroid
    let centroid = point::centroid(
        &full_intersection_points
            .iter()
            .map(|ip| *ip.point())
            .collect::<Vec<Point>>(),
    );

    let mut indices: Vec<usize> = (0..n_points).collect();
    indices.sort_by(|&i, &j| {
        full_intersection_points[i]
            .point()
            .angle_to(&centroid)
            .partial_cmp(&full_intersection_points[j].point().angle_to(&centroid))
            .unwrap_or(std::cmp::Ordering::Less)
    });

    let mut area = 0.0;

    let mut l = n_points - 1;

    for k in 0..n_points {
        let i = indices[k];
        let j = indices[l];

        let p1 = &full_intersection_points[i].point();
        let p2 = &full_intersection_points[j].point();

        // Now we need to discover which of the circles the two points are
        // coming from so that we can compute the segment area.
        // This should be the set intersection of the parents of both points.
        // In some cases, the intersection may be of length 2, in which
        // case we need to compute both segment areas and pick the
        // smaller one.
        let parents1 = &full_intersection_points[i].parents();
        let parents2 = &full_intersection_points[j].parents();

        let common_parents: Vec<usize> = vec![parents1.0, parents1.1]
            .into_iter()
            .filter(|p| *p == parents2.0 || *p == parents2.1)
            .collect();

        let mut segment_areas = Vec::with_capacity(common_parents.len());

        if common_parents.is_empty() {
            // This should not happen in a well-formed set of intersection points
            panic!("No common parent circles found for intersection points");
        }

        for &circle_index in &common_parents {
            let circle = &circles[circle_index];
            let seg_area = circle.segment_area_from_points(p1, p2);

            debug_assert!(seg_area >= 0.0, "Segment area should be non-negative");

            segment_areas.push(seg_area);
        }

        let triangle_area = 0.5 * ((p1.x() + p2.x()) * (p1.y() - p2.y()));
        // Note: triangle_area can be negative (signed area from shoelace algorithm)
        // We take abs() at the end to get the final area

        let min_segment = segment_areas
            .into_iter()
            .fold(f64::INFINITY, |a, b| a.min(b));

        area += triangle_area;
        area += min_segment;

        l = k;
    }

    area.abs()
}

/// Compute the area of the intersection region for a subset of circles.
///
/// This is similar to `multiple_overlap_areas` but allows specifying which circles
/// to consider for the "full intersection". This is needed for computing 3-way
/// intersections in a 4+ circle diagram where some intersection points may be
/// in more than just the 3 circles of interest.
///
/// # Arguments
/// * `circles` - All circles in the diagram
/// * `points` - Intersection points (with adopters referencing indices in `circles`)
/// * `circle_indices` - Indices of the circles that define this region
#[deprecated(
    since = "0.3.1",
    note = "Returns wrong area when one circle in the mask contains the others' lens. Use `crate::geometry::diagram::compute_exclusive_regions` or the boundary-arc helpers (`region_boundary_arcs` + `area_from_boundary_arcs`) instead."
)]
pub fn multiple_overlap_areas_with_mask(
    circles: &[Circle],
    points: &[IntersectionPoint],
    circle_indices: &[usize],
) -> f64 {
    // Filter to only points that are in ALL of the specified circles
    // A point is in the region if all circle_indices are present in its adopters
    let region_points: Vec<&IntersectionPoint> = points
        .iter()
        .filter(|ip| {
            circle_indices
                .iter()
                .all(|&idx| ip.adopters().contains(&idx))
        })
        .collect();

    if region_points.is_empty() {
        return 0.0;
    }

    let n_points = region_points.len();

    // Sort the points by their angles around the centroid
    let centroid = point::centroid(
        &region_points
            .iter()
            .map(|ip| *ip.point())
            .collect::<Vec<Point>>(),
    );

    let mut indices: Vec<usize> = (0..n_points).collect();
    indices.sort_by(|&i, &j| {
        region_points[i]
            .point()
            .angle_to(&centroid)
            .partial_cmp(&region_points[j].point().angle_to(&centroid))
            .unwrap_or(std::cmp::Ordering::Less)
    });

    let mut area = 0.0;

    let mut l = n_points - 1;

    for k in 0..n_points {
        let i = indices[k];
        let j = indices[l];

        let p1 = &region_points[i].point();
        let p2 = &region_points[j].point();

        // Find which of the region circles these points come from
        let parents1 = &region_points[i].parents();
        let parents2 = &region_points[j].parents();

        let common_parents: Vec<usize> = vec![parents1.0, parents1.1]
            .into_iter()
            .filter(|p| *p == parents2.0 || *p == parents2.1)
            .filter(|p| circle_indices.contains(p)) // Only consider circles in our region
            .collect();

        let mut segment_areas = Vec::with_capacity(common_parents.len());

        if common_parents.is_empty() {
            // Try to find any circle in the region that contains both points
            // This can happen when points come from different pairs but are connected
            // through the region
            for &circle_idx in circle_indices {
                let circle = &circles[circle_idx];
                if circle.contains_point(p1) && circle.contains_point(p2) {
                    let seg_area = circle.segment_area_from_points(p1, p2);
                    segment_areas.push(seg_area);
                }
            }

            if segment_areas.is_empty() {
                // No circle in the region connects these points - use straight line
                // This shouldn't normally happen in a well-formed region
                l = k;
                continue;
            }
        } else {
            for &circle_index in &common_parents {
                let circle = &circles[circle_index];
                let seg_area = circle.segment_area_from_points(p1, p2);

                debug_assert!(seg_area >= 0.0, "Segment area should be non-negative");

                segment_areas.push(seg_area);
            }
        }

        let triangle_area = 0.5 * ((p1.x() + p2.x()) * (p1.y() - p2.y()));

        let min_segment = segment_areas
            .into_iter()
            .fold(f64::INFINITY, |a, b| a.min(b));

        area += triangle_area;
        area += min_segment;

        l = k;
    }

    area.abs()
}

/// One circular arc on the boundary of a region, oriented to be traversed in the
/// CCW direction around the region.
///
/// `phi_start` and `phi_end` are the standard `atan2` angles (in `(-π, π]`) of
/// the arc endpoints relative to the owning circle's centre. `delta_phi` is the
/// signed short-arc angular delta from `phi_start` to `phi_end` in `(-π, π]`
/// (a full-circle arc uses `2π`); its sign matches the traversal direction
/// around the owning circle (`+1` = CCW, `−1` = CW). The pair `(phi_start,
/// delta_phi)` is sufficient to drive the boundary integral; `phi_end` is
/// stored only as a convenience for the closed-form gradient formulas.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundaryArc {
    pub circle: usize,
    pub phi_start: f64,
    pub phi_end: f64,
    pub delta_phi: f64,
}

/// Implicit-form value `((p.x − c.x)² + (p.y − c.y)²) / r²` for point `p` and
/// circle `c`. `< 1` strictly inside, `= 1` on the boundary, `> 1` outside.
#[inline]
fn circle_implicit_value(p: &Point, c: &Circle) -> f64 {
    let dx = p.x() - c.center().x();
    let dy = p.y() - c.center().y();
    (dx * dx + dy * dy) / (c.radius() * c.radius())
}

/// Decide whether an arc whose midpoint is on `∂C_j` is owned by `j` for
/// the purpose of region-boundary contributions. The midpoint must be inside
/// every other mask circle; when it lies on another mask circle's boundary
/// (boundaries coincide), the smaller-index circle owns the arc to avoid
/// double-counting (the identical-circles case in particular would otherwise
/// emit one full-circle arc per circle).
///
/// Tolerance is scaled per comparator circle: a geometric near-tangency of
/// `~δ` translates to an implicit-value deviation of `~2δ/r`, so we pick
/// `eps_l = 2·BOUNDARY_COINCIDENCE_GEOM_TOL / r_l` (clamped against
/// floating-point noise) so the threshold corresponds to a roughly constant
/// geometric gap regardless of circle scale.
fn arc_midpoint_owned_by_j(j: usize, mid: &Point, indices: &[usize], circles: &[Circle]) -> bool {
    for &l in indices {
        if l == j {
            continue;
        }
        let cl = &circles[l];
        let eps = (2.0 * BOUNDARY_COINCIDENCE_GEOM_TOL / cl.radius()).max(IMPLICIT_VALUE_FP_TOL);
        let v = circle_implicit_value(mid, cl);
        if v > 1.0 + eps {
            return false;
        }
        if (v - 1.0).abs() <= eps && l < j {
            return false;
        }
    }
    true
}

/// Geometric tolerance (in world-frame distance units) for treating two
/// circle boundaries as coincident at a probe point. Converted to a per-circle
/// implicit-value tolerance inside the tiebreaker.
const BOUNDARY_COINCIDENCE_GEOM_TOL: f64 = 1e-7;

/// Floor for the per-circle implicit-value tolerance, guarding against
/// floating-point noise on huge circles where the geometric tolerance would
/// otherwise map to a sub-ulp implicit-value threshold.
const IMPLICIT_VALUE_FP_TOL: f64 = 1e-12;

/// Build the CCW-oriented list of boundary arcs for an overlapping region.
///
/// For each circle in the mask, the function identifies the IPs that lie on
/// that circle's boundary and inside every other circle in the mask, sorts
/// them by angle around the circle's centre, and emits an arc for every
/// inter-IP segment whose midpoint sits inside every other mask circle. When
/// no such IPs exist on a circle, a probe point is used to decide whether the
/// circle contributes a full-circle arc (i.e. it is wholly inside every
/// other) or no arc.
///
/// All emitted arcs have `delta_phi > 0` (CCW around the owning circle), since
/// for a region that is the intersection of disks, R sits inside each
/// boundary circle and CCW-around-Cₖ on the boundary coincides with CCW
/// around R. This is robust across degenerate cases that the older
/// "polygon + min-segment" decomposition would miss — most notably 3+-way
/// regions where the IPs all come from a single pair of circles because a
/// third circle fully contains their lens.
///
/// When circle boundaries coincide (identical circles, or arcs shared between
/// two circles), an index tiebreaker — `arc_midpoint_owned_by_j` — hands the
/// arc to the smallest-index circle so the area is not multi-counted.
pub(crate) fn region_boundary_arcs(
    mask: crate::geometry::diagram::RegionMask,
    circles: &[Circle],
    intersections: &[IntersectionPoint],
    n_sets: usize,
) -> Vec<BoundaryArc> {
    use crate::geometry::diagram::{adopters_to_mask, mask_to_indices};
    let circle_count = mask.count_ones();
    if circle_count == 0 {
        return Vec::new();
    }
    if circle_count == 1 {
        let idx = mask.trailing_zeros() as usize;
        return vec![BoundaryArc {
            circle: idx,
            phi_start: 0.0,
            phi_end: 0.0,
            delta_phi: 2.0 * PI,
        }];
    }

    let indices = mask_to_indices(mask, n_sets);
    let mut arcs = Vec::new();
    let two_pi = 2.0 * PI;
    // Numerical guard for collapsing arcs at tangencies / shared IPs.
    let arc_eps = 1e-9;

    for &j in &indices {
        let cj = &circles[j];
        // IPs on ∂C_j (j as a parent) that are inside every other mask circle.
        let mut j_phis: Vec<f64> = intersections
            .iter()
            .filter(|ip| {
                let (p1, p2) = ip.parents();
                if p1 != j && p2 != j {
                    return false;
                }
                let am = adopters_to_mask(ip.adopters());
                (mask & am) == mask
            })
            .map(|ip| {
                let p = ip.point();
                (p.y() - cj.center().y()).atan2(p.x() - cj.center().x())
            })
            .collect();

        if j_phis.is_empty() {
            // Either C_j is wholly inside every other mask circle (full-circle
            // arc) or wholly outside the region (no arc). Probe at φ = 0; if
            // the probe lies on ∂C_l for some l ≠ j (boundaries coincide),
            // the index tiebreaker hands the arc to the smallest-index circle
            // so identical / coincident-boundary circles don't all emit
            // duplicate full-circle arcs.
            let probe = Point::new(cj.center().x() + cj.radius(), cj.center().y());
            if arc_midpoint_owned_by_j(j, &probe, &indices, circles) {
                arcs.push(BoundaryArc {
                    circle: j,
                    phi_start: 0.0,
                    phi_end: 0.0,
                    delta_phi: two_pi,
                });
            }
            continue;
        }

        j_phis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = j_phis.len();

        for k in 0..m {
            let phi_a = j_phis[k];
            let phi_b = j_phis[(k + 1) % m];
            // CCW arc length from phi_a to phi_b.
            let mut delta = phi_b - phi_a;
            if delta <= 0.0 {
                delta += two_pi;
            }
            if delta < arc_eps || delta > two_pi - arc_eps {
                continue;
            }
            // Midpoint of the candidate arc; include only if it sits inside
            // every other circle in the mask (i.e. on R's boundary). The
            // index tiebreaker handles boundary-coincident arcs the same way
            // as the empty-IP branch above.
            let phi_mid = phi_a + delta * 0.5;
            let mid = Point::new(
                cj.center().x() + cj.radius() * phi_mid.cos(),
                cj.center().y() + cj.radius() * phi_mid.sin(),
            );
            if !arc_midpoint_owned_by_j(j, &mid, &indices, circles) {
                continue;
            }
            // Canonicalise phi_end to (-π, π] so sin/cos in the area &
            // gradient formulas operate on raw atan2 values; the periodicity
            // of sin/cos makes phi_end == phi_a + delta (continuous) and
            // phi_end (canonical) interchangeable.
            let phi_end_canonical = wrap_angle(phi_a + delta);
            arcs.push(BoundaryArc {
                circle: j,
                phi_start: phi_a,
                phi_end: phi_end_canonical,
                delta_phi: delta,
            });
        }
    }

    arcs
}

/// Compute the area enclosed by a CCW boundary arc list via the line-integral
/// `A = (1/2) ∮ (x dy − y dx)`. For an arc on circle k parametrised by φ:
/// ```text
/// x dy − y dx = (xₖ rₖ cos φ + yₖ rₖ sin φ + rₖ²) dφ
/// ```
/// integrating from `phi_start` to `phi_start + delta_phi` (continuous).
pub(crate) fn area_from_boundary_arcs(arcs: &[BoundaryArc], circles: &[Circle]) -> f64 {
    let mut total = 0.0;
    for arc in arcs {
        let cj = &circles[arc.circle];
        let xk = cj.center().x();
        let yk = cj.center().y();
        let r = cj.radius();
        let phi_a = arc.phi_start;
        let phi_b = phi_a + arc.delta_phi;
        total += xk * r * (phi_b.sin() - phi_a.sin());
        total += yk * r * (phi_a.cos() - phi_b.cos());
        total += r * r * arc.delta_phi;
    }
    0.5 * total
}

/// Wrap an angle into `(-π, π]`.
#[inline]
pub(crate) fn wrap_angle(x: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let y = x.rem_euclid(two_pi);
    if y > PI { y - two_pi } else { y }
}

/// Accumulate the gradient of an overlapping region's area into `grad`, where
/// `grad` is a length-`3 · n_sets` vector laid out as `[x₀, y₀, r₀, x₁, …]`.
///
/// Each boundary arc on circle `k` contributes:
/// ```text
/// ∂A/∂xₖ += sign · rₖ · (sin φ_end − sin φ_start)
/// ∂A/∂yₖ −= sign · rₖ · (cos φ_end − cos φ_start)
/// ∂A/∂rₖ += rₖ · |delta_phi|
/// ```
/// where `sign = sign(delta_phi)`, derived from the boundary-velocity identity
/// `dA/dθ = ∮_∂R (v_θ · n) ds` with `θ ∈ {xₖ, yₖ, rₖ}`.
pub(crate) fn accumulate_region_overlap_gradient(
    arcs: &[BoundaryArc],
    circles: &[Circle],
    grad: &mut [f64],
) {
    for arc in arcs {
        let r = circles[arc.circle].radius();
        let sign = arc.delta_phi.signum();
        let off = arc.circle * 3;
        grad[off] += sign * r * (arc.phi_end.sin() - arc.phi_start.sin());
        grad[off + 1] -= sign * r * (arc.phi_end.cos() - arc.phi_start.cos());
        grad[off + 2] += r * arc.delta_phi.abs();
    }
}

/// Compute the per-mask exclusive areas of `circles` clipped to an
/// axis-aligned `container` rectangle, including the all-zeros region (mask
/// `0`) representing `container ∖ ⋃ circles` (the complement target).
///
/// Algorithm: for each overlap region `R_m = ⋂_{i ∈ m} D_i`, compute
/// `area(R_m ∩ container)` via Green's theorem. The boundary
/// `∂(R_m ∩ container)` decomposes into:
///
/// - sub-arcs of `∂R_m` that lie inside the container, and
/// - sub-segments of `∂container` (axis-aligned edges) that lie inside `R_m`.
///
/// Both pieces are integrated in closed form. After that, inclusion-exclusion
/// (`to_exclusive_areas`) converts the overlap areas to per-mask exclusive
/// areas; mask `0` is seeded with `area(container)` so the IE pass produces
/// the complement (`container.area − area(⋃ disks ∩ container)`) for free.
///
/// See `area_from_clipped_arcs` (private) for the per-region implementation.
pub fn compute_exclusive_regions_clipped(
    circles: &[Circle],
    container: &Rectangle,
) -> std::collections::HashMap<crate::geometry::diagram::RegionMask, f64> {
    use crate::geometry::diagram::{collect_intersections, discover_regions, to_exclusive_areas};

    let n_sets = circles.len();
    let intersections = collect_intersections(circles, n_sets);
    let regions = discover_regions(circles, &intersections, n_sets);

    let mut overlapping_areas = std::collections::HashMap::new();
    // Seed mask 0 with the container area. After inclusion-exclusion the
    // exclusive value at mask 0 becomes `container.area − area(⋃ disks ∩
    // container)`, i.e. the complement.
    overlapping_areas.insert(0, container.area());

    for &mask in &regions {
        let arcs = region_boundary_arcs(mask, circles, &intersections, n_sets);
        let area = area_from_clipped_arcs(&arcs, circles, container, mask, n_sets);
        overlapping_areas.insert(mask, area);
    }

    to_exclusive_areas(&overlapping_areas)
}

/// Gradient-aware companion to [`compute_exclusive_regions_clipped`].
///
/// Returns `(exclusive_areas, exclusive_grads)` where each gradient vector has
/// length `n_sets · 3 + 4` and is laid out as
/// `[x₀, y₀, r₀, x₁, …, x_c, y_c, u, v]` — the trailing four entries are the
/// container's optimizer encoding (`u = ln(w·h)`, `v = ln(w/h)`).
///
/// The shape-param gradient comes from the same boundary-velocity identity
/// `dA/dθ = ∮_∂R (v_θ · n) ds` as the unclipped path
/// (`accumulate_region_overlap_gradient`), restricted to inside-container
/// sub-arcs. Box-edge sub-segments contribute to the container-param block
/// only: each edge moves rigidly with `(x_c, y_c, u, v)`, so the line
/// integral collapses to length-times-velocity. Endpoints where an arc meets a
/// box edge cancel pairwise (same point, same `(v · n)`, opposite ds direction
/// across the two boundary pieces), so this builds a consistent gradient
/// without tracking trim-point motion.
///
/// Mask `0` (the complement) is seeded with `container.area()` and a gradient
/// `∂(w·h)/∂(x_c, y_c, u, v) = [0, 0, w·h, 0]` so inclusion-exclusion produces
/// `complement = container.area − area(⋃ disks ∩ container)` along with its
/// matching gradient.
pub(crate) fn compute_exclusive_regions_clipped_with_gradient(
    circles: &[Circle],
    container: &Rectangle,
) -> (
    std::collections::HashMap<crate::geometry::diagram::RegionMask, f64>,
    std::collections::HashMap<crate::geometry::diagram::RegionMask, Vec<f64>>,
) {
    use crate::geometry::diagram::{
        collect_intersections, discover_regions, to_exclusive_areas_and_gradients,
    };

    let n_sets = circles.len();
    let n_params = n_sets * 3 + 4;
    let intersections = collect_intersections(circles, n_sets);
    let regions = discover_regions(circles, &intersections, n_sets);

    let mut overlapping_areas = std::collections::HashMap::new();
    let mut overlapping_grads: std::collections::HashMap<
        crate::geometry::diagram::RegionMask,
        Vec<f64>,
    > = std::collections::HashMap::new();

    // Seed mask 0 with `container.area()` and its gradient. ∂(w·h)/∂u = w·h
    // (chain through `u = ln(w·h)`); other partials are zero.
    let container_area = container.area();
    overlapping_areas.insert(0, container_area);
    let mut zero_grad = vec![0.0; n_params];
    zero_grad[n_sets * 3 + 2] = container_area;
    overlapping_grads.insert(0, zero_grad);

    for &mask in &regions {
        let arcs = region_boundary_arcs(mask, circles, &intersections, n_sets);
        let mut grad = vec![0.0; n_params];
        let area =
            area_and_gradient_from_clipped_arcs(&arcs, circles, container, mask, n_sets, &mut grad);
        overlapping_areas.insert(mask, area);
        overlapping_grads.insert(mask, grad);
    }

    to_exclusive_areas_and_gradients(&overlapping_areas, &overlapping_grads, n_params)
}

/// Compute `area(R_m ∩ container)` from the CCW boundary arcs of `R_m` and
/// the axis-aligned `container`, where `R_m = ⋂_{i ∈ m} disks[i]`.
///
/// Boundary decomposition:
/// - Each input arc contributes its inside-container sub-arcs (split at
///   crossings with the four box edges, kept iff the sub-arc midpoint lies
///   inside the container).
/// - Each container edge contributes its inside-`R_m` sub-segments
///   (intersection of per-disk projections of the edge into each disk in the
///   mask).
///
/// The two pieces share Green's theorem `A = ½ ∮ (x dy − y dx)`; arc and
/// segment contributions are summed and halved.
pub(crate) fn area_from_clipped_arcs(
    arcs: &[BoundaryArc],
    circles: &[Circle],
    container: &Rectangle,
    mask: crate::geometry::diagram::RegionMask,
    n_sets: usize,
) -> f64 {
    use crate::geometry::diagram::mask_to_indices;

    let (x_min, x_max, y_min, y_max) = container.bounds();

    // Defensive: a degenerate (zero-area) box can only intersect things with
    // measure zero. The arc and edge integrals would produce 0 anyway, but
    // bailing avoids wasted work and any divide-by-zero risk in helpers.
    if x_max <= x_min || y_max <= y_min {
        return 0.0;
    }

    let mut total = 0.0; // ∫ (x dy − y dx); area = 0.5 · total.

    // 1. Arc contributions.
    for arc in arcs {
        total += clipped_arc_integral(arc, circles, x_min, x_max, y_min, y_max);
    }

    // 2. Container-edge contributions, each oriented CCW around the container
    // interior. For each edge we find the inside-`R_m` sub-interval (the
    // intersection of the per-disk projections onto that edge), then add the
    // closed-form line integral.
    let indices = mask_to_indices(mask, n_sets);

    // Bottom edge: y = y_min, walk x from x_min → x_max. ∫(−y dx) = −y_min·Δx.
    if let Some((a, b)) = horizontal_edge_inside_interval(y_min, x_min, x_max, &indices, circles) {
        total += -y_min * (b - a);
    }
    // Right edge: x = x_max, walk y from y_min → y_max. ∫(x dy) = x_max·Δy.
    if let Some((a, b)) = vertical_edge_inside_interval(x_max, y_min, y_max, &indices, circles) {
        total += x_max * (b - a);
    }
    // Top edge: y = y_max, walk x from x_max → x_min. ∫(−y dx) = +y_max·Δx
    // because dx is negative when walking right→left.
    if let Some((a, b)) = horizontal_edge_inside_interval(y_max, x_min, x_max, &indices, circles) {
        total += y_max * (b - a);
    }
    // Left edge: x = x_min, walk y from y_max → y_min. ∫(x dy) = −x_min·Δy
    // because dy is negative when walking top→bottom.
    if let Some((a, b)) = vertical_edge_inside_interval(x_min, y_min, y_max, &indices, circles) {
        total += -x_min * (b - a);
    }

    0.5 * total
}

/// Gradient-aware companion to [`area_from_clipped_arcs`]. Computes
/// `area(R_m ∩ container)` and accumulates `∂area / ∂θ` into `grad`
/// (length `n_sets · 3 + 4`, layout `[x₀, y₀, r₀, …, x_c, y_c, u, v]`).
///
/// The arc and box-edge decompositions are identical to the area-only path;
/// the gradient is produced inline using the per-piece boundary-velocity
/// formulas — see the function docs of
/// [`compute_exclusive_regions_clipped_with_gradient`] for the derivation
/// summary.
pub(crate) fn area_and_gradient_from_clipped_arcs(
    arcs: &[BoundaryArc],
    circles: &[Circle],
    container: &Rectangle,
    mask: crate::geometry::diagram::RegionMask,
    n_sets: usize,
    grad: &mut [f64],
) -> f64 {
    use crate::geometry::diagram::mask_to_indices;

    let (x_min, x_max, y_min, y_max) = container.bounds();

    if x_max <= x_min || y_max <= y_min {
        return 0.0;
    }

    let w = x_max - x_min;
    let h = y_max - y_min;
    let container_off = n_sets * 3;

    let mut total = 0.0; // ∫ (x dy − y dx); area = 0.5 · total.

    // 1. Arc contributions — area + shape-param gradient.
    for arc in arcs {
        total += clipped_arc_integral_with_gradient(arc, circles, x_min, x_max, y_min, y_max, grad);
    }

    // 2. Container-edge contributions. Each edge is a line of length `L`
    // inside `R_m` (intersection of the per-disk projections onto the edge).
    // Area integral as in `area_from_clipped_arcs`; container-param gradient
    // from the rigid-edge boundary velocity (see fn docs above).
    let indices = mask_to_indices(mask, n_sets);

    if let Some((a, b)) = horizontal_edge_inside_interval(y_min, x_min, x_max, &indices, circles) {
        let l = b - a;
        // Area: bottom edge walked left→right; ∫(−y dx) = −y_min·L.
        total += -y_min * l;
        // Gradient: outward normal (0, −1); v·n = −∂y_min/∂θ.
        // ∂y_min/∂(x_c, y_c, u, v) = (0, 1, −h/4, h/4)
        grad[container_off + 1] -= l;
        grad[container_off + 2] += (h / 4.0) * l;
        grad[container_off + 3] -= (h / 4.0) * l;
    }
    if let Some((a, b)) = vertical_edge_inside_interval(x_max, y_min, y_max, &indices, circles) {
        let l = b - a;
        total += x_max * l;
        // Right edge: outward normal (+1, 0); v·n = ∂x_max/∂θ.
        // ∂x_max/∂(x_c, y_c, u, v) = (1, 0, w/4, w/4)
        grad[container_off] += l;
        grad[container_off + 2] += (w / 4.0) * l;
        grad[container_off + 3] += (w / 4.0) * l;
    }
    if let Some((a, b)) = horizontal_edge_inside_interval(y_max, x_min, x_max, &indices, circles) {
        let l = b - a;
        total += y_max * l;
        // Top edge: outward normal (0, +1); v·n = ∂y_max/∂θ.
        // ∂y_max/∂(x_c, y_c, u, v) = (0, 1, h/4, −h/4)
        grad[container_off + 1] += l;
        grad[container_off + 2] += (h / 4.0) * l;
        grad[container_off + 3] -= (h / 4.0) * l;
    }
    if let Some((a, b)) = vertical_edge_inside_interval(x_min, y_min, y_max, &indices, circles) {
        let l = b - a;
        total += -x_min * l;
        // Left edge: outward normal (−1, 0); v·n = −∂x_min/∂θ.
        // ∂x_min/∂(x_c, y_c, u, v) = (1, 0, −w/4, −w/4)
        grad[container_off] -= l;
        grad[container_off + 2] += (w / 4.0) * l;
        grad[container_off + 3] += (w / 4.0) * l;
    }

    0.5 * total
}

/// Sum the Green's-theorem arc integral over the inside-container sub-arcs of
/// `arc`, splitting at crossings with the four box edges.
fn clipped_arc_integral(
    arc: &BoundaryArc,
    circles: &[Circle],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> f64 {
    let cj = &circles[arc.circle];
    let xc = cj.center().x();
    let yc = cj.center().y();
    let r = cj.radius();

    let phi_a = arc.phi_start;
    let phi_b = phi_a + arc.delta_phi; // continuous (not wrapped)

    // Collect crossing parameters strictly inside (phi_a, phi_b). Endpoints
    // bookend the sub-arc walk separately.
    let mut crossings: Vec<f64> = Vec::with_capacity(8);

    // Vertical edges: cos φ = (X − xc) / r → φ = ±acos(c) (+ 2πk).
    for &x_edge in &[x_min, x_max] {
        let c = (x_edge - xc) / r;
        if c.abs() <= 1.0 {
            let phi0 = c.acos(); // ∈ [0, π]
            collect_periodic_in_range(&mut crossings, phi0, phi_a, phi_b);
            collect_periodic_in_range(&mut crossings, -phi0, phi_a, phi_b);
        }
    }
    // Horizontal edges: sin φ = (Y − yc) / r → φ = asin(s) or π − asin(s).
    for &y_edge in &[y_min, y_max] {
        let s = (y_edge - yc) / r;
        if s.abs() <= 1.0 {
            let phi0 = s.asin(); // ∈ [−π/2, π/2]
            let phi1 = std::f64::consts::PI - phi0;
            collect_periodic_in_range(&mut crossings, phi0, phi_a, phi_b);
            collect_periodic_in_range(&mut crossings, phi1, phi_a, phi_b);
        }
    }

    let mut breaks: Vec<f64> = Vec::with_capacity(crossings.len() + 2);
    breaks.push(phi_a);
    breaks.extend(crossings);
    breaks.push(phi_b);
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut sum = 0.0;
    for w in breaks.windows(2) {
        let a = w[0];
        let b = w[1];
        let delta = b - a;
        if delta <= 1e-12 {
            continue;
        }
        let mid = 0.5 * (a + b);
        let mx = xc + r * mid.cos();
        let my = yc + r * mid.sin();
        // Generous geometric tolerance to keep boundary-aligned midpoints
        // (e.g. arcs that exit and immediately re-enter via a corner) on the
        // "inside" side. Mismatched signs here would lose a slice of area.
        let tol = 1e-9;
        let inside =
            mx >= x_min - tol && mx <= x_max + tol && my >= y_min - tol && my <= y_max + tol;
        if !inside {
            continue;
        }
        sum += xc * r * (b.sin() - a.sin()) + yc * r * (a.cos() - b.cos()) + r * r * (b - a);
    }
    sum
}

/// Gradient-aware companion to [`clipped_arc_integral`]. Returns the same
/// area-integrand contribution to `2A` while also accumulating shape-param
/// gradient entries for the owning circle into `grad` (length `≥ 3 · n_sets +
/// 4`, layout `[x₀, y₀, r₀, …]`).
///
/// Per inside-container sub-arc on circle `k` with continuous endpoints
/// `(a, b)`, `b > a`:
///
/// ```text
/// ∂A/∂x_k += r · (sin b − sin a)
/// ∂A/∂y_k += r · (cos a − cos b)
/// ∂A/∂r_k += r · (b − a)
/// ```
///
/// derived from the boundary-velocity identity (`v_θ · n = cos φ`, `sin φ`,
/// `1` for `θ = x_k, y_k, r_k`) integrated against `ds = r dφ`. Trim-point
/// motion at box-edge endpoints is captured by the matching box-edge
/// integral (same point, opposite-sign `ds`), so the per-sub-arc formula is
/// self-contained.
fn clipped_arc_integral_with_gradient(
    arc: &BoundaryArc,
    circles: &[Circle],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    grad: &mut [f64],
) -> f64 {
    let cj = &circles[arc.circle];
    let xc = cj.center().x();
    let yc = cj.center().y();
    let r = cj.radius();

    let phi_a = arc.phi_start;
    let phi_b = phi_a + arc.delta_phi;

    let mut crossings: Vec<f64> = Vec::with_capacity(8);

    for &x_edge in &[x_min, x_max] {
        let c = (x_edge - xc) / r;
        if c.abs() <= 1.0 {
            let phi0 = c.acos();
            collect_periodic_in_range(&mut crossings, phi0, phi_a, phi_b);
            collect_periodic_in_range(&mut crossings, -phi0, phi_a, phi_b);
        }
    }
    for &y_edge in &[y_min, y_max] {
        let s = (y_edge - yc) / r;
        if s.abs() <= 1.0 {
            let phi0 = s.asin();
            let phi1 = std::f64::consts::PI - phi0;
            collect_periodic_in_range(&mut crossings, phi0, phi_a, phi_b);
            collect_periodic_in_range(&mut crossings, phi1, phi_a, phi_b);
        }
    }

    let mut breaks: Vec<f64> = Vec::with_capacity(crossings.len() + 2);
    breaks.push(phi_a);
    breaks.extend(crossings);
    breaks.push(phi_b);
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let off = arc.circle * 3;
    let mut sum = 0.0;
    for w in breaks.windows(2) {
        let a = w[0];
        let b = w[1];
        let delta = b - a;
        if delta <= 1e-12 {
            continue;
        }
        let mid = 0.5 * (a + b);
        let mx = xc + r * mid.cos();
        let my = yc + r * mid.sin();
        let tol = 1e-9;
        let inside =
            mx >= x_min - tol && mx <= x_max + tol && my >= y_min - tol && my <= y_max + tol;
        if !inside {
            continue;
        }
        let s_diff = b.sin() - a.sin();
        let c_diff = b.cos() - a.cos();
        // 2A integrand (matches the area-only path).
        sum += xc * r * s_diff - yc * r * c_diff + r * r * delta;
        // Boundary-velocity gradient on circle k. Sub-arcs are traversed CCW
        // (delta > 0), so sign = +1.
        grad[off] += r * s_diff;
        grad[off + 1] -= r * c_diff;
        grad[off + 2] += r * delta;
    }
    sum
}

/// For an angle `cand` solving a circle/edge equation, push every periodic
/// copy `cand + 2πk` strictly inside `(phi_a, phi_b)` into `out`. Arc spans
/// can reach 2π so at most one copy per shift offset is in range; iterating
/// `k ∈ {-1, 0, 1}` covers every case for `delta_phi ≤ 2π`.
fn collect_periodic_in_range(out: &mut Vec<f64>, cand: f64, phi_a: f64, phi_b: f64) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let tol = 1e-12;
    for k in -1..=1 {
        let p = cand + (k as f64) * two_pi;
        if p > phi_a + tol && p < phi_b - tol {
            out.push(p);
        }
    }
}

/// Inside-`R_m` sub-interval of a horizontal edge `y = const` over
/// `x ∈ [x_min, x_max]`. Returns `None` if the edge does not touch every
/// disk in `indices`. With `indices` empty, the whole `[x_min, x_max]` is
/// "inside" (intersection over an empty constraint set is the universe),
/// which is exactly what mask `0` (the complement) wants.
fn horizontal_edge_inside_interval(
    y: f64,
    x_min: f64,
    x_max: f64,
    indices: &[usize],
    circles: &[Circle],
) -> Option<(f64, f64)> {
    let mut lo = x_min;
    let mut hi = x_max;
    for &i in indices {
        let c = &circles[i];
        let dy = y - c.center().y();
        let r = c.radius();
        let r2 = r * r;
        if dy * dy > r2 {
            return None;
        }
        let dx = (r2 - dy * dy).max(0.0).sqrt();
        let disk_lo = c.center().x() - dx;
        let disk_hi = c.center().x() + dx;
        if disk_lo > lo {
            lo = disk_lo;
        }
        if disk_hi < hi {
            hi = disk_hi;
        }
        if hi <= lo {
            return None;
        }
    }
    if hi > lo { Some((lo, hi)) } else { None }
}

/// Inside-`R_m` sub-interval of a vertical edge `x = const` over
/// `y ∈ [y_min, y_max]`. Mirror of `horizontal_edge_inside_interval`.
fn vertical_edge_inside_interval(
    x: f64,
    y_min: f64,
    y_max: f64,
    indices: &[usize],
    circles: &[Circle],
) -> Option<(f64, f64)> {
    let mut lo = y_min;
    let mut hi = y_max;
    for &i in indices {
        let c = &circles[i];
        let dx = x - c.center().x();
        let r = c.radius();
        let r2 = r * r;
        if dx * dx > r2 {
            return None;
        }
        let dy = (r2 - dx * dx).max(0.0).sqrt();
        let disk_lo = c.center().y() - dy;
        let disk_hi = c.center().y() + dy;
        if disk_lo > lo {
            lo = disk_lo;
        }
        if disk_hi < hi {
            hi = disk_hi;
        }
        if hi <= lo {
            return None;
        }
    }
    if hi > lo { Some((lo, hi)) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_circle_new() {
        let center = Point::new(1.0, 2.0);
        let circle = Circle::new(center, 5.0);
        assert_eq!(circle.radius(), 5.0);
        assert_eq!(circle.center().x(), 1.0);
        assert_eq!(circle.center().y(), 2.0);
    }

    #[test]
    fn test_circle_try_new_accepts_positive() {
        let circle = Circle::try_new(Point::new(0.0, 0.0), 1.0).unwrap();
        assert_eq!(circle.radius(), 1.0);
    }

    #[test]
    fn test_circle_try_new_rejects_zero_and_negative() {
        let err = Circle::try_new(Point::new(0.0, 0.0), 0.0).unwrap_err();
        assert!(matches!(
            err,
            crate::error::DiagramError::InvalidShapeParameter {
                shape: "Circle",
                param: "radius",
                ..
            }
        ));
        let err = Circle::try_new(Point::new(0.0, 0.0), -1.5).unwrap_err();
        assert!(matches!(
            err,
            crate::error::DiagramError::InvalidShapeParameter {
                shape: "Circle",
                param: "radius",
                ..
            }
        ));
    }

    #[test]
    #[should_panic(expected = "Circle radius must be > 0")]
    fn test_circle_new_panics_on_zero_radius() {
        let _ = Circle::new(Point::new(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_circle_area() {
        let circle = Circle::new(Point::new(0.0, 0.0), 1.0);
        assert!(approx_eq(circle.area(), PI));

        let circle2 = Circle::new(Point::new(0.0, 0.0), 2.0);
        assert!(approx_eq(circle2.area(), 4.0 * PI));

        let circle3 = Circle::new(Point::new(5.0, 5.0), 3.0);
        assert!(approx_eq(circle3.area(), 9.0 * PI));
    }

    #[test]
    fn test_circle_distance_no_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(5.0, 0.0), 1.0);
        assert_eq!(circle1.distance(&circle2), 3.0);
    }

    #[test]
    fn test_circle_distance_touching() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        assert_eq!(circle1.distance(&circle2), 0.0);
    }

    #[test]
    fn test_circle_distance_overlapping() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(1.0, 0.0), 2.0);
        assert_eq!(circle1.distance(&circle2), 0.0);
    }

    #[test]
    fn test_circle_contains_smaller() {
        let large = Circle::new(Point::new(0.0, 0.0), 5.0);
        let small = Circle::new(Point::new(1.0, 1.0), 2.0);
        assert!(large.contains(&small));
    }

    #[test]
    fn test_circle_contains_self() {
        let circle = Circle::new(Point::new(0.0, 0.0), 3.0);
        assert!(circle.contains(&circle));
    }

    #[test]
    fn test_circle_not_contains() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(5.0, 0.0), 2.0);
        assert!(!circle1.contains(&circle2));
    }

    #[test]
    fn test_circle_not_contains_partial_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 2.0);
        assert!(!circle1.contains(&circle2));
    }

    #[test]
    fn test_circle_intersects_separate() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(5.0, 0.0), 1.0);
        // Distance = 5, sum of radii = 2, circles are separated
        assert!(!circle1.intersects(&circle2));
    }

    #[test]
    fn test_circle_intersects_touching() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        // Distance = 2, sum of radii = 2, circles touch at exactly one point
        // This is a boundary case - could be either true or false depending on definition
        // We treat tangent as NOT intersecting (no area overlap)
        assert!(!circle1.intersects(&circle2));
    }

    #[test]
    fn test_circle_intersects_overlapping() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(1.0, 0.0), 2.0);
        // Distance = 1, sum of radii = 4, circles overlap
        assert!(circle1.intersects(&circle2));
    }

    #[test]
    fn test_intersection_area_no_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(10.0, 0.0), 1.0);
        assert_eq!(circle1.intersection_area(&circle2), 0.0);
    }

    #[test]
    fn test_intersection_area_touching() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);
        assert!(approx_eq(area, 0.0));
    }

    #[test]
    fn test_intersection_area_complete_overlap_same_size() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let expected = PI * 4.0;
        assert!(approx_eq(circle1.intersection_area(&circle2), expected));
    }

    #[test]
    fn test_intersection_area_one_inside_other() {
        let large = Circle::new(Point::new(0.0, 0.0), 5.0);
        let small = Circle::new(Point::new(1.0, 0.0), 2.0);
        let expected = PI * 4.0; // Area of smaller circle
        assert!(approx_eq(large.intersection_area(&small), expected));
        assert!(approx_eq(small.intersection_area(&large), expected));
    }

    #[test]
    fn test_intersection_area_partial_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(1.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);

        // For two unit circles with centers 1 apart, there's a known formula
        // The intersection area should be positive and less than π
        assert!(area > 0.0);
        assert!(area < PI);
    }

    #[test]
    fn test_intersection_area_symmetric() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(1.5, 0.0), 1.5);
        let area1 = circle1.intersection_area(&circle2);
        let area2 = circle2.intersection_area(&circle1);
        assert!(approx_eq(area1, area2));
    }

    #[test]
    fn test_intersection_area_different_sizes() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);

        // Should be positive and at most the smaller circle's area
        assert!(area > 0.0);
        assert!(area <= PI * 1.0 * 1.0);
    }

    #[test]
    fn test_distance_for_overlap_zero_overlap() {
        let r1 = 2.0;
        let r2 = 1.5;
        let overlap = 0.0;

        let distance = distance_for_overlap(r1, r2, overlap, None, None).unwrap();

        // Should return the sum of radii (circles just touching)
        assert!(approx_eq(distance, r1 + r2));
    }

    #[test]
    fn test_distance_for_overlap_negative_overlap() {
        let r1 = 2.0;
        let r2 = 1.5;
        let overlap = -1.0;

        let distance = distance_for_overlap(r1, r2, overlap, None, None).unwrap();

        // Should return the sum of radii (circles separated)
        assert!(approx_eq(distance, r1 + r2));
    }

    #[test]
    fn test_distance_for_overlap_full_overlap() {
        let r1 = 3.0;
        let r2 = 2.0;
        let overlap = PI * r2 * r2; // Full area of smaller circle

        let distance = distance_for_overlap(r1, r2, overlap, None, None).unwrap();

        // Distance should be close to |r1 - r2| (one inside the other)
        assert!(distance <= (r1 - r2).abs() + 0.1); // Allow small tolerance
    }

    #[test]
    fn test_distance_for_overlap_partial_overlap_equal_radii() {
        let r1 = 2.0;
        let r2 = 2.0;
        let target_overlap = 2.0; // Some specific overlap area

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        // Verify the result by computing the actual overlap at this distance
        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        // Should match target within tolerance (relaxed for optimization convergence)
        assert!((actual_overlap - target_overlap).abs() < 1e-2);
    }

    #[test]
    fn test_distance_for_overlap_partial_overlap_different_radii() {
        let r1 = 3.0;
        let r2 = 1.5;
        let target_overlap = 1.0;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        // Verify the result
        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-2);
    }

    #[test]
    fn test_distance_for_overlap_custom_tolerance() {
        let r1 = 2.0;
        let r2 = 1.0;
        let target_overlap = 0.5;
        let custom_tol = 1e-8;

        let distance =
            distance_for_overlap(r1, r2, target_overlap, Some(custom_tol), None).unwrap();

        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        // Should be very close to target with custom tolerance
        assert!((actual_overlap - target_overlap).abs() < 1e-3);
    }

    #[test]
    fn test_distance_for_overlap_custom_max_iter() {
        let r1 = 2.0;
        let r2 = 1.5;
        let target_overlap = 1.0;
        let max_iter = 50;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, Some(max_iter)).unwrap();

        // Should still converge within fewer iterations
        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-2);
    }

    #[test]
    fn test_distance_for_overlap_small_circles() {
        let r1 = 0.5;
        let r2 = 0.3;
        let target_overlap = 0.1;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-4);
    }

    #[test]
    fn test_distance_for_overlap_large_circles() {
        let r1 = 100.0;
        let r2 = 75.0;
        let target_overlap = 500.0;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1.0); // Larger tolerance for large circles
    }

    #[test]
    fn test_distance_for_overlap_bounds() {
        let r1 = 2.0;
        let r2 = 1.5;
        let target_overlap = 1.0;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        // Distance should be between min (one inside other) and max (circles touching)
        let min_distance = (r1 - r2).abs();
        let max_distance = r1 + r2;

        assert!(distance >= min_distance);
        assert!(distance <= max_distance);
    }

    #[test]
    fn test_intersection_points_no_intersection() {
        // Circles too far apart
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(5.0, 0.0), 1.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_intersection_points_one_inside_other() {
        // One circle completely inside the other
        let c1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let c2 = Circle::new(Point::new(0.0, 0.0), 1.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_intersection_points_touching_externally() {
        // Circles touch at exactly one point (externally)
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(4.0, 0.0), 2.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 1);

        // The touching point should be at (2.0, 0.0)
        assert!(approx_eq(points[0].x(), 2.0));
        assert!(approx_eq(points[0].y(), 0.0));
    }

    #[test]
    fn test_intersection_points_two_points() {
        // Circles intersect at two points
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(2.0, 0.0), 2.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // Both points should be on the circles
        for point in &points {
            let dist_to_c1 = c1.center.distance(point);
            let dist_to_c2 = c2.center.distance(point);
            assert!(approx_eq(dist_to_c1, c1.radius));
            assert!(approx_eq(dist_to_c2, c2.radius));
        }

        // Points should be at (1.0, sqrt(3)) and (1.0, -sqrt(3))
        let expected_x = 1.0;
        let expected_y = 3.0_f64.sqrt();

        // Check one point is at (1.0, sqrt(3))
        let found_positive = points
            .iter()
            .any(|p| approx_eq(p.x(), expected_x) && approx_eq(p.y(), expected_y));
        // Check other point is at (1.0, -sqrt(3))
        let found_negative = points
            .iter()
            .any(|p| approx_eq(p.x(), expected_x) && approx_eq(p.y(), -expected_y));

        assert!(found_positive);
        assert!(found_negative);
    }

    #[test]
    fn test_intersection_points_vertical_alignment() {
        // Test with circles aligned vertically
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.5);
        let c2 = Circle::new(Point::new(0.0, 2.0), 1.5);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // Both points should be equidistant from both centers
        for point in &points {
            let dist_to_c1 = c1.center.distance(point);
            let dist_to_c2 = c2.center.distance(point);
            assert!(approx_eq(dist_to_c1, c1.radius));
            assert!(approx_eq(dist_to_c2, c2.radius));
        }
    }

    #[test]
    fn test_intersection_points_equal_radii_partial_overlap() {
        // Two circles with equal radii, partially overlapping
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.0, 0.0), 1.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // The intersection points should be symmetric about the line connecting centers
        // They should both have x-coordinate = 0.5
        for point in &points {
            assert!(approx_eq(point.x(), 0.5));
        }
    }

    #[test]
    fn test_intersection_points_different_radii() {
        // Two circles with different radii
        let c1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let c2 = Circle::new(Point::new(2.0, 0.0), 1.5);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // Verify both points lie on both circles
        for point in &points {
            let dist_to_c1 = c1.center.distance(point);
            let dist_to_c2 = c2.center.distance(point);
            assert!(approx_eq(dist_to_c1, c1.radius));
            assert!(approx_eq(dist_to_c2, c2.radius));
        }
    }

    #[test]
    fn test_perimeter() {
        let c = Circle::new(Point::new(0.0, 0.0), 1.0);
        let perimeter = c.perimeter();
        assert!(approx_eq(perimeter, 2.0 * PI));
    }

    #[test]
    fn test_perimeter_various_radii() {
        let test_cases = vec![(0.5, PI), (2.0, 4.0 * PI), (10.0, 20.0 * PI)];

        for (radius, expected) in test_cases {
            let c = Circle::new(Point::new(0.0, 0.0), radius);
            assert!(approx_eq(c.perimeter(), expected));
        }
    }

    #[test]
    fn test_sector_area_full_circle() {
        let c = Circle::new(Point::new(0.0, 0.0), 2.0);
        let full_circle_angle = 2.0 * PI;
        let sector = c.sector_area(full_circle_angle);
        assert!(approx_eq(sector, c.area()));
    }

    #[test]
    fn test_sector_area_half_circle() {
        let c = Circle::new(Point::new(0.0, 0.0), 3.0);
        let half_circle_angle = PI;
        let sector = c.sector_area(half_circle_angle);
        assert!(approx_eq(sector, c.area() / 2.0));
    }

    #[test]
    fn test_sector_area_quarter_circle() {
        let c = Circle::new(Point::new(0.0, 0.0), 4.0);
        let quarter_circle_angle = PI / 2.0;
        let sector = c.sector_area(quarter_circle_angle);
        assert!(approx_eq(sector, c.area() / 4.0));
    }

    #[test]
    fn test_sector_area_zero() {
        let c = Circle::new(Point::new(0.0, 0.0), 5.0);
        let sector = c.sector_area(0.0);
        assert!(approx_eq(sector, 0.0));
    }

    #[test]
    fn test_segment_area_from_angle_semicircle() {
        let c = Circle::new(Point::new(0.0, 0.0), 2.0);
        let angle = PI;
        let segment = c.segment_area_from_angle(angle);
        // For a semicircle, segment area equals sector area (half circle)
        assert!(approx_eq(segment, c.area() / 2.0));
    }

    #[test]
    fn test_segment_area_from_angle_zero() {
        let c = Circle::new(Point::new(0.0, 0.0), 3.0);
        let segment = c.segment_area_from_angle(0.0);
        assert!(approx_eq(segment, 0.0));
    }

    #[test]
    fn test_segment_area_from_angle_small() {
        let c = Circle::new(Point::new(0.0, 0.0), 1.0);
        let angle = PI / 6.0; // 30 degrees
        let segment = c.segment_area_from_angle(angle);

        // Segment should be positive and less than sector area
        let sector = c.sector_area(angle);
        assert!(segment > 0.0);
        assert!(segment < sector);
    }

    #[test]
    fn test_segment_area_from_chord_diameter() {
        let radius = 2.0;
        let c = Circle::new(Point::new(0.0, 0.0), radius);
        let chord_length = 2.0 * radius; // Diameter
        let segment = c.segment_area_from_chord(chord_length);

        // A chord equal to diameter creates a semicircle
        assert!(approx_eq(segment, c.area() / 2.0));
    }

    #[test]
    fn test_segment_area_from_chord_small() {
        let c = Circle::new(Point::new(0.0, 0.0), 5.0);
        let chord_length = 2.0;
        let segment = c.segment_area_from_chord(chord_length);

        // Small chord should create small segment
        assert!(segment > 0.0);
        assert!(segment < c.area() / 4.0);
    }

    #[test]
    fn test_segment_area_from_chord_vs_angle_consistency() {
        let c = Circle::new(Point::new(0.0, 0.0), 3.0);
        let angle = PI / 3.0; // 60 degrees

        // Calculate chord length from angle
        let chord_length = 2.0 * c.radius() * (angle / 2.0).sin();

        let segment_from_angle = c.segment_area_from_angle(angle);
        let segment_from_chord = c.segment_area_from_chord(chord_length);

        // Both methods should give same result
        assert!(approx_eq(segment_from_angle, segment_from_chord));
    }

    #[test]
    fn test_segment_area_relationships() {
        let c = Circle::new(Point::new(0.0, 0.0), 1.0);
        let angle = PI / 4.0; // 45 degrees

        let sector = c.sector_area(angle);
        let segment = c.segment_area_from_angle(angle);

        // Segment area should be less than sector area (triangle is subtracted)
        assert!(segment < sector);

        // For small angles, segment should be much smaller than sector
        let small_angle = 0.1;
        let small_sector = c.sector_area(small_angle);
        let small_segment = c.segment_area_from_angle(small_angle);
        assert!(small_segment < small_sector / 2.0);
    }

    #[test]
    #[should_panic(expected = "Chord length")]
    fn test_segment_area_from_chord_invalid() {
        let c = Circle::new(Point::new(0.0, 0.0), 2.0);
        let chord_length = 5.0; // Impossible: longer than diameter
        c.segment_area_from_chord(chord_length);
    }

    #[test]
    fn test_three_circle_complete_overlap() {
        use crate::geometry::traits::DiagramShape;

        // Three identical unit circles. Without the index tiebreaker each
        // circle would emit a full-circle arc and A∩B∩C would return ~3π.
        let c = Circle::new(Point::new(0.0, 0.0), 1.0);
        let areas = Circle::compute_exclusive_regions(&[c, c, c]);

        let mask_all = 0b111;
        let all_three = areas.get(&mask_all).copied().unwrap_or(0.0);

        assert!(
            (all_three - PI).abs() < 1e-6,
            "Complete overlap should give area ~π, got {all_three}"
        );
    }

    #[test]
    fn test_three_circle_two_coincident_one_smaller() {
        use crate::geometry::traits::DiagramShape;

        // Two coincident unit circles plus a smaller circle of radius 0.5
        // sharing the same centre. Expected A∩B∩C = π·0.25.
        // Without the tiebreaker, both unit circles emit a full-circle arc
        // (each one sits on the other's boundary), so the inclusion-exclusion
        // pipeline overshoots.
        let big = Circle::new(Point::new(0.0, 0.0), 1.0);
        let small = Circle::new(Point::new(0.0, 0.0), 0.5);
        let areas = Circle::compute_exclusive_regions(&[big, big, small]);

        let mask_all = 0b111;
        let all_three = areas.get(&mask_all).copied().unwrap_or(0.0);
        let expected = PI * 0.25;

        assert!(
            (all_three - expected).abs() < 1e-6,
            "Expected area {expected}, got {all_three}"
        );
    }

    // ===== Container clipping (compute_exclusive_regions_clipped) =====

    /// A circle entirely inside the container should keep its full area; the
    /// complement region is the rest of the container.
    #[test]
    fn clipped_single_circle_inside_container() {
        let circles = vec![Circle::new(Point::new(0.0, 0.0), 1.0)];
        let container = Rectangle::new(Point::new(0.0, 0.0), 6.0, 6.0); // 36
        let areas = compute_exclusive_regions_clipped(&circles, &container);

        let disk = areas.get(&0b1).copied().unwrap_or(0.0);
        let complement = areas.get(&0).copied().unwrap_or(0.0);

        assert!((disk - PI).abs() < 1e-9, "disk area {disk} ≠ π");
        assert!(
            (complement - (36.0 - PI)).abs() < 1e-9,
            "complement {complement} ≠ 36 − π"
        );
    }

    /// A circle entirely outside the container should contribute zero; the
    /// complement equals the full container area.
    #[test]
    fn clipped_single_circle_outside_container() {
        let circles = vec![Circle::new(Point::new(100.0, 100.0), 1.0)];
        let container = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0); // 16
        let areas = compute_exclusive_regions_clipped(&circles, &container);

        let disk = areas.get(&0b1).copied().unwrap_or(0.0);
        let complement = areas.get(&0).copied().unwrap_or(0.0);

        assert!(disk.abs() < 1e-9, "disk area {disk} ≠ 0");
        assert!(
            (complement - 16.0).abs() < 1e-9,
            "complement {complement} ≠ 16"
        );
    }

    /// A circle clipped by exactly one edge produces a circular segment cut
    /// off; the kept area equals `π − segment(2·angle)` where the segment is
    /// the slice above the cut.
    #[test]
    fn clipped_single_circle_one_edge_matches_segment_formula() {
        // Unit circle at origin; container's top edge cuts through y = 0.5.
        let circles = vec![Circle::new(Point::new(0.0, 0.0), 1.0)];
        let container = Rectangle::new(Point::new(0.0, -2.25), 100.0, 5.5);
        // Container bounds: x ∈ [−50, 50], y ∈ [−5, 0.5]. Cuts the top of the
        // unit circle at y = 0.5.
        let areas = compute_exclusive_regions_clipped(&circles, &container);

        let disk_clipped = areas.get(&0b1).copied().unwrap_or(0.0);

        // Closed form: the slice above y = 0.5 is a circular segment with
        // central angle 2·acos(0.5) = 2π/3. Segment area = ½r²(θ − sin θ).
        let theta = 2.0 * (0.5_f64).acos(); // 2π/3
        let segment = 0.5 * (theta - theta.sin());
        let expected = PI - segment;

        assert!(
            (disk_clipped - expected).abs() < 1e-9,
            "clipped disk {disk_clipped} ≠ π − segment ({expected})"
        );
    }

    /// Container fully inside one circle: clipped area equals container area;
    /// complement is zero.
    #[test]
    fn clipped_container_fully_inside_circle() {
        let circles = vec![Circle::new(Point::new(0.0, 0.0), 100.0)];
        let container = Rectangle::new(Point::new(0.0, 0.0), 4.0, 6.0);
        let areas = compute_exclusive_regions_clipped(&circles, &container);

        let disk = areas.get(&0b1).copied().unwrap_or(0.0);
        let complement = areas.get(&0).copied().unwrap_or(0.0);

        assert!(
            (disk - 24.0).abs() < 1e-9,
            "disk-clipped {disk} ≠ container area"
        );
        assert!(complement.abs() < 1e-9, "complement {complement} ≠ 0");
    }

    /// Two overlapping disks both fully inside the container: the per-mask
    /// exclusive areas sum to the (2-disk) union area; complement = box − union.
    #[test]
    fn clipped_two_disks_inside_sum_to_union_area() {
        // Two unit disks centered at (±0.6, 0). Both inside a 6×6 box.
        let circles = vec![
            Circle::new(Point::new(-0.6, 0.0), 1.0),
            Circle::new(Point::new(0.6, 0.0), 1.0),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.0), 6.0, 6.0); // 36
        let areas = compute_exclusive_regions_clipped(&circles, &container);

        let a_only = areas.get(&0b01).copied().unwrap_or(0.0);
        let b_only = areas.get(&0b10).copied().unwrap_or(0.0);
        let a_and_b = areas.get(&0b11).copied().unwrap_or(0.0);
        let complement = areas.get(&0).copied().unwrap_or(0.0);

        // Union = a-only + b-only + a∩b. Compute it independently.
        let union = a_only + b_only + a_and_b;
        // Direct calc: 2π − (lens between two unit circles distance 1.2 apart).
        let lens = circles[0].intersection_area(&circles[1]);
        let expected_union = 2.0 * PI - lens;

        assert!(
            (union - expected_union).abs() < 1e-9,
            "union via per-mask sum {union} ≠ {expected_union}"
        );
        assert!(
            (complement - (36.0 - expected_union)).abs() < 1e-9,
            "complement {complement} ≠ box − union"
        );
    }

    /// Sanity: with no shapes, the complement equals the container area.
    #[test]
    fn clipped_no_circles_complement_equals_container_area() {
        let circles: Vec<Circle> = Vec::new();
        let container = Rectangle::new(Point::new(1.0, 2.0), 3.0, 4.0);
        let areas = compute_exclusive_regions_clipped(&circles, &container);

        let complement = areas.get(&0).copied().unwrap_or(0.0);
        assert!(
            (complement - 12.0).abs() < 1e-9,
            "complement {complement} ≠ 12"
        );
    }

    /// Container clipping a circle from a corner (top + right edges). The
    /// kept piece is the disk minus a "moon" wedge near the top-right; we
    /// validate against a Monte-Carlo reference.
    #[test]
    fn clipped_single_circle_corner_matches_monte_carlo() {
        // Unit circle at origin. Container's interior is the lower-left
        // quadrant from (-2, -2) to (0.5, 0.5), so two adjacent edges cut
        // the circle.
        let circles = vec![Circle::new(Point::new(0.0, 0.0), 1.0)];
        let container = Rectangle::new(Point::new(-0.75, -0.75), 2.5, 2.5);
        let areas = compute_exclusive_regions_clipped(&circles, &container);
        let kept = areas.get(&0b1).copied().unwrap_or(0.0);

        // Monte-Carlo reference: sample uniformly in the bounding box of the
        // unit disk (a 2x2 square). Count points inside both the disk and
        // container.
        let n = 400_000;
        let mut hits = 0;
        // Deterministic LCG so the test is reproducible without rand.
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let next_u01 = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for _ in 0..n {
            let x = -1.0 + 2.0 * next_u01(&mut state);
            let y = -1.0 + 2.0 * next_u01(&mut state);
            let in_disk = x * x + y * y <= 1.0;
            let in_box = (-2.0..=0.5).contains(&x) && (-2.0..=0.5).contains(&y);
            if in_disk && in_box {
                hits += 1;
            }
        }
        let mc_area = 4.0 * (hits as f64) / (n as f64); // 4 = box area
        // 400k samples → stderr ≈ √(p(1-p)/n) · 4 ≈ 0.005 typically.
        assert!(
            (kept - mc_area).abs() < 0.02,
            "clipped disk {kept} vs MC {mc_area}"
        );
    }

    // ===== Clipped exclusive areas: analytical gradient (S2) =====

    /// Pack `(circles, container)` into the same flat parameter layout the
    /// optimiser uses: per-circle `[x, y, r]` blocks, then container
    /// `[x_c, y_c, ln(area), ln(ratio)]`.
    fn pack_clipped_params(circles: &[Circle], container: &Rectangle) -> Vec<f64> {
        let mut p = Vec::with_capacity(circles.len() * 3 + 4);
        for c in circles {
            p.push(c.center().x());
            p.push(c.center().y());
            p.push(c.radius());
        }
        p.extend(container.to_optimizer_params());
        p
    }

    /// Inverse of `pack_clipped_params`: decode a flat parameter slice back
    /// into circles plus container.
    fn unpack_clipped_params(p: &[f64], n_sets: usize) -> (Vec<Circle>, Rectangle) {
        let circles: Vec<Circle> = (0..n_sets)
            .map(|i| {
                Circle::new(
                    Point::new(p[3 * i], p[3 * i + 1]),
                    p[3 * i + 2].max(f64::MIN_POSITIVE),
                )
            })
            .collect();
        let container = Rectangle::from_optimizer_params(&p[3 * n_sets..3 * n_sets + 4]);
        (circles, container)
    }

    /// FD vs analytical gradient comparison for the clipped per-mask area
    /// helper. `params` is laid out as in `pack_clipped_params`.
    fn assert_clipped_gradient_matches_fd(
        circles: &[Circle],
        container: &Rectangle,
        h: f64,
        tol: f64,
        label: &str,
    ) {
        let (areas, grads) = compute_exclusive_regions_clipped_with_gradient(circles, container);
        let n_sets = circles.len();
        let n_params = n_sets * 3 + 4;
        let p0 = pack_clipped_params(circles, container);

        // Compute FD gradient per mask.
        for (mask, analytic) in &grads {
            assert_eq!(
                analytic.len(),
                n_params,
                "{}: gradient length {} ≠ expected {}",
                label,
                analytic.len(),
                n_params
            );
            let mut fd = vec![0.0; n_params];
            for i in 0..n_params {
                let mut plus = p0.clone();
                let mut minus = p0.clone();
                plus[i] += h;
                minus[i] -= h;
                let (cp, kp) = unpack_clipped_params(&plus, n_sets);
                let (cm, km) = unpack_clipped_params(&minus, n_sets);
                let ap = compute_exclusive_regions_clipped(&cp, &kp)
                    .get(mask)
                    .copied()
                    .unwrap_or(0.0);
                let am = compute_exclusive_regions_clipped(&cm, &km)
                    .get(mask)
                    .copied()
                    .unwrap_or(0.0);
                fd[i] = (ap - am) / (2.0 * h);
            }
            // Check the analytical area matches the value-only path.
            let direct = compute_exclusive_regions_clipped(circles, container)
                .get(mask)
                .copied()
                .unwrap_or(0.0);
            let analytic_area = areas.get(mask).copied().unwrap_or(0.0);
            assert!(
                (analytic_area - direct).abs() < 1e-12,
                "{label}: mask {mask:b} area {analytic_area} vs direct {direct} mismatch"
            );
            // Compare gradients.
            let diff_norm: f64 = analytic
                .iter()
                .zip(fd.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
            let rel = if fd_norm > 1e-9 {
                diff_norm / fd_norm
            } else {
                diff_norm
            };
            assert!(
                rel < tol,
                "{label}: mask {mask:b} analytic vs FD mismatch (rel={rel:.3e}, |fd|={fd_norm:.3e})\n  analytic={analytic:?}\n  fd      ={fd:?}"
            );
        }
    }

    /// Two disks fully inside a wide container. No box edge clips any disk —
    /// only the complement region touches the box. Verifies that shape param
    /// gradients reproduce the unclipped gradient and that the container
    /// gradient is purely (0, 0, container_area, 0).
    #[test]
    fn clipped_grad_two_disks_inside_box_matches_fd() {
        let circles = vec![
            Circle::new(Point::new(-0.6, 0.05), 1.0),
            Circle::new(Point::new(0.62, -0.04), 0.95),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.0), 6.0, 5.0);
        assert_clipped_gradient_matches_fd(&circles, &container, 1e-6, 1e-5, "two_disks_inside");
    }

    /// Two disks each clipped by a different box edge. Exercises the
    /// box-edge boundary contribution to the container-param gradient and
    /// the trimmed-arc contribution to shape params.
    #[test]
    fn clipped_grad_two_disks_each_touching_an_edge_matches_fd() {
        // Disk 0 sits near the right edge of the box (x_max = 1.8),
        // clipped at x ≥ 1.8. Disk 1 sits left-of-centre, overlapping disk 0.
        let circles = vec![
            Circle::new(Point::new(1.4, 0.05), 0.9),
            Circle::new(Point::new(0.0, -0.07), 0.85),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.0), 3.6, 3.0);
        assert_clipped_gradient_matches_fd(
            &circles,
            &container,
            1e-6,
            1e-4,
            "two_disks_clipped_edge",
        );
    }

    /// Three overlapping disks, all inside the box but with the box snug
    /// enough that the complement and pairwise/triple regions all carry
    /// container-edge contributions.
    #[test]
    fn clipped_grad_three_disks_inside_box_matches_fd() {
        let circles = vec![
            Circle::new(Point::new(-0.5, -0.3), 1.0),
            Circle::new(Point::new(0.5, -0.3), 1.0),
            Circle::new(Point::new(0.0, 0.55), 1.0),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.05), 3.5, 3.5);
        assert_clipped_gradient_matches_fd(&circles, &container, 1e-6, 1e-4, "three_disks");
    }

    /// Three overlapping disks with the box clipping the right-most disk
    /// from the top. Mixes inside-disk regions with disks that have
    /// inside-container sub-arcs.
    #[test]
    fn clipped_grad_three_disks_one_clipped_matches_fd() {
        let circles = vec![
            Circle::new(Point::new(-0.6, -0.2), 0.9),
            Circle::new(Point::new(0.7, -0.2), 0.9),
            Circle::new(Point::new(0.05, 0.6), 0.9),
        ];
        // Top edge cuts through circle 2 (at y = 0.6 + 0.9 - clip = 1.0
        // means the top edge at y_max = 1.0 cuts a slice off the top of
        // disk 2).
        let container = Rectangle::new(Point::new(0.0, 0.0), 3.5, 2.0);
        assert_clipped_gradient_matches_fd(
            &circles,
            &container,
            1e-6,
            1e-4,
            "three_disks_clipped_top",
        );
    }

    fn assert_circle(circle: &Circle, x: f64, y: f64, radius: f64) {
        assert!(approx_eq(circle.center().x(), x), "center.x");
        assert!(approx_eq(circle.center().y(), y), "center.y");
        assert!(approx_eq(circle.radius(), radius), "radius");
    }

    #[test]
    fn test_canonical_venn_layout_n1() {
        use crate::geometry::traits::DiagramShape;
        let shapes = Circle::canonical_venn_layout(1).unwrap();
        assert_eq!(shapes.len(), 1);
        assert_circle(&shapes[0], 0.0, 0.0, 1.0);
    }

    #[test]
    fn test_canonical_venn_layout_n2() {
        use crate::geometry::traits::DiagramShape;
        let shapes = Circle::canonical_venn_layout(2).unwrap();
        assert_eq!(shapes.len(), 2);
        assert_circle(&shapes[0], -0.35, 0.0, 0.6);
        assert_circle(&shapes[1], 0.35, 0.0, 0.6);
    }

    #[test]
    fn test_canonical_venn_layout_n3() {
        use crate::geometry::traits::DiagramShape;
        let shapes = Circle::canonical_venn_layout(3).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_circle(&shapes[0], 0.0, 0.45, 0.6);
        assert_circle(&shapes[1], -0.389711, -0.225, 0.6);
        assert_circle(&shapes[2], 0.389711, -0.225, 0.6);
    }

    #[test]
    fn test_canonical_venn_layout_unsupported() {
        use crate::geometry::traits::DiagramShape;
        assert!(Circle::canonical_venn_layout(0).is_none());
        assert!(Circle::canonical_venn_layout(4).is_none());
        assert!(Circle::canonical_venn_layout(5).is_none());
    }
}
