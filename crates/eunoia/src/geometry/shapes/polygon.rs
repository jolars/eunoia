//! Polygon shape for visualization and export.
//!
//! Polygons are used to represent circles and ellipses as discrete vertices
//! for plotting and export to other formats. They are not used in the core
//! diagram computation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::geometry::primitives::Point;

/// A polygon defined by a sequence of vertices.
///
/// Polygons are primarily used for visualization - converting analytical shapes
/// (circles, ellipses) into discrete point representations for plotting libraries
/// like D3.js or other renderers.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::geometry::primitives::Point;
/// use eunoia::geometry::traits::Polygonize;
///
/// // Create from a circle
/// let circle = Circle::new(Point::new(0.0, 0.0), 5.0);
/// let polygon = circle.polygonize(32);
///
/// assert_eq!(polygon.vertices().len(), 32);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    vertices: Vec<Point>,
}

impl Polygon {
    /// Creates a new polygon from a sequence of vertices.
    ///
    /// # Validity
    ///
    /// Unlike [`Circle::new`](super::Circle::new),
    /// [`Ellipse::new`](super::Ellipse::new), and
    /// [`Square::new`](super::Square::new) — which validate their numeric
    /// parameters — `Polygon::new` accepts any vertex sequence, including
    /// empty (`[]`), degenerate (fewer than three vertices, collinear),
    /// self-intersecting, or non-simple rings. This is intentional: this
    /// type is the carrier for arbitrary `i_overlay` clip output, where
    /// intermediate empty/degenerate rings are normal and adding validation
    /// here would force every internal caller to guard against them.
    /// Methods such as [`Polygon::area`] return safe defaults (e.g. `0.0`)
    /// for degenerate input.
    pub fn new(vertices: Vec<Point>) -> Self {
        Self { vertices }
    }

    /// Returns the vertices of the polygon.
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }

    /// Computes the area of the polygon using the shoelace formula.
    pub fn area(&self) -> f64 {
        if self.vertices.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x() * self.vertices[j].y();
            area -= self.vertices[j].x() * self.vertices[i].y();
        }

        (area / 2.0).abs()
    }

    /// Computes the centroid of the polygon.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Polygon;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// // Unit square centered at (0.5, 0.5)
    /// let polygon = Polygon::new(vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(0.0, 1.0),
    /// ]);
    /// let centroid = polygon.centroid();
    /// assert!((centroid.x() - 0.5).abs() < 1e-10);
    /// assert!((centroid.y() - 0.5).abs() < 1e-10);
    /// ```
    pub fn centroid(&self) -> Point {
        if self.vertices.is_empty() {
            return Point::new(0.0, 0.0);
        }

        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut area = 0.0;
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let cross = self.vertices[i].x() * self.vertices[j].y()
                - self.vertices[j].x() * self.vertices[i].y();
            area += cross;
            cx += (self.vertices[i].x() + self.vertices[j].x()) * cross;
            cy += (self.vertices[i].y() + self.vertices[j].y()) * cross;
        }

        area *= 0.5;
        if area.abs() < 1e-10 {
            cx = self.vertices.iter().map(|p| p.x()).sum::<f64>() / n as f64;
            cy = self.vertices.iter().map(|p| p.y()).sum::<f64>() / n as f64;
        } else {
            cx /= 6.0 * area;
            cy /= 6.0 * area;
        }

        Point::new(cx, cy)
    }

    /// Finds the pole of inaccessibility (visual center) of the polygon.
    ///
    /// The pole of inaccessibility is the most distant internal point from the polygon outline.
    /// This is more visually pleasing than the centroid for label placement, especially for
    /// complex or concave polygons.
    ///
    /// # Arguments
    ///
    /// * `precision` - The tolerance for the algorithm (smaller = more accurate but
    ///   slower).
    ///
    /// # Returns
    ///
    /// The point representing the visual center of the polygon.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Polygon;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// // L-shaped polygon
    /// let polygon = Polygon::new(vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(4.0, 0.0),
    ///     Point::new(4.0, 1.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(1.0, 4.0),
    ///     Point::new(0.0, 4.0),
    /// ]);
    /// let pole = polygon.pole_of_inaccessibility(0.1);
    /// // pole will be near (0.5625, 0.5625) - much better than centroid for L-shapes
    /// ```
    pub fn pole_of_inaccessibility(&self, precision: f64) -> Point {
        self.pole_of_inaccessibility_with_distance(precision).0
    }

    /// Like [`pole_of_inaccessibility`], but also returns the distance from
    /// the pole to the polygon boundary.
    ///
    /// The distance is a measure of how much room the polygon has around its
    /// label anchor — useful when picking among several candidate polygons
    /// (e.g. the disconnected pieces of a single Euler region) for the one
    /// that best fits a label.
    ///
    /// [`pole_of_inaccessibility`]: Self::pole_of_inaccessibility
    pub fn pole_of_inaccessibility_with_distance(&self, precision: f64) -> (Point, f64) {
        if self.vertices.len() < 3 {
            return (self.centroid(), 0.0);
        }
        polylabel(&self.vertices, self.centroid(), precision)
    }
}

fn min_distance_to_boundary(point: &Point, vertices: &[Point]) -> f64 {
    if vertices.len() < 2 {
        return 0.0;
    }
    let n = vertices.len();
    let mut best = f64::INFINITY;
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        let dx = b.x() - a.x();
        let dy = b.y() - a.y();
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((point.x() - a.x()) * dx + (point.y() - a.y()) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let px = a.x() + t * dx;
        let py = a.y() + t * dy;
        let d = ((point.x() - px).powi(2) + (point.y() - py).powi(2)).sqrt();
        if d < best {
            best = d;
        }
    }
    best
}

fn point_in_polygon(x: f64, y: f64, vertices: &[Point]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (vertices[i].x(), vertices[i].y());
        let (xj, yj) = (vertices[j].x(), vertices[j].y());
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn signed_distance_to_polygon(x: f64, y: f64, vertices: &[Point]) -> f64 {
    let d = min_distance_to_boundary(&Point::new(x, y), vertices);
    if point_in_polygon(x, y, vertices) {
        d
    } else {
        -d
    }
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    x: f64,
    y: f64,
    h: f64,
    d: f64,
    max: f64,
}

impl Cell {
    fn new(x: f64, y: f64, h: f64, vertices: &[Point]) -> Self {
        let d = signed_distance_to_polygon(x, y, vertices);
        let max = d + h * std::f64::consts::SQRT_2;
        Self { x, y, h, d, max }
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.max == other.max
    }
}

impl Eq for Cell {}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.max.partial_cmp(&other.max).unwrap_or(Ordering::Equal)
    }
}

/// Mapbox's polylabel algorithm: priority-queue quadtree subdivision over the
/// polygon's bounding box, scored by signed distance to the boundary.
/// Winding-agnostic — relies on a ray-cast point-in-polygon test rather than
/// signed area.
fn polylabel(vertices: &[Point], centroid: Point, precision: f64) -> (Point, f64) {
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for v in vertices {
        if v.x() < min_x {
            min_x = v.x();
        }
        if v.y() < min_y {
            min_y = v.y();
        }
        if v.x() > max_x {
            max_x = v.x();
        }
        if v.y() > max_y {
            max_y = v.y();
        }
    }

    let width = max_x - min_x;
    let height = max_y - min_y;
    let cell_size = width.min(height);
    if cell_size <= 0.0 {
        return (Point::new(min_x, min_y), 0.0);
    }
    let half = cell_size / 2.0;

    let mut heap: BinaryHeap<Cell> = BinaryHeap::new();
    let mut x = min_x;
    while x < max_x {
        let mut y = min_y;
        while y < max_y {
            heap.push(Cell::new(x + half, y + half, half, vertices));
            y += cell_size;
        }
        x += cell_size;
    }

    let bbox_center = Cell::new(min_x + width / 2.0, min_y + height / 2.0, 0.0, vertices);
    let centroid_cell = Cell::new(centroid.x(), centroid.y(), 0.0, vertices);
    let mut best = if centroid_cell.d > bbox_center.d {
        centroid_cell
    } else {
        bbox_center
    };

    while let Some(cell) = heap.pop() {
        if cell.max - best.d <= precision {
            continue;
        }
        if cell.d > best.d {
            best = cell;
        }
        let h2 = cell.h / 2.0;
        heap.push(Cell::new(cell.x - h2, cell.y - h2, h2, vertices));
        heap.push(Cell::new(cell.x + h2, cell.y - h2, h2, vertices));
        heap.push(Cell::new(cell.x - h2, cell.y + h2, h2, vertices));
        heap.push(Cell::new(cell.x + h2, cell.y + h2, h2, vertices));
    }

    (Point::new(best.x, best.y), best.d.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::{Circle, Ellipse};
    use crate::geometry::traits::Polygonize;

    #[test]
    fn test_polygon_area_square() {
        let polygon = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]);
        assert!((polygon.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_circle() {
        let circle = Circle::new(Point::new(0.0, 0.0), 5.0);
        let polygon = circle.polygonize(32);

        assert_eq!(polygon.vertices().len(), 32);

        for vertex in polygon.vertices() {
            let dist = ((vertex.x() - 0.0).powi(2) + (vertex.y() - 0.0).powi(2)).sqrt();
            assert!((dist - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_from_ellipse() {
        let ellipse = Ellipse::new(Point::new(1.0, 2.0), 4.0, 2.0, 0.0);
        let polygon = ellipse.polygonize(64);

        assert_eq!(polygon.vertices().len(), 64);
    }

    #[test]
    fn test_pole_of_inaccessibility_square() {
        // For a square, pole should be near the center
        let polygon = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let pole = polygon.pole_of_inaccessibility(0.1);

        // Should be very close to (5, 5)
        assert!((pole.x() - 5.0).abs() < 1.0);
        assert!((pole.y() - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_pole_of_inaccessibility_l_shape() {
        // L-shaped polygon - pole should be better than centroid
        let polygon = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 4.0),
            Point::new(0.0, 4.0),
        ]);

        let pole = polygon.pole_of_inaccessibility(0.1);
        let centroid = polygon.centroid();

        // Pole should be in the bottom-left area (near 0.5, 0.5)
        // This is better than centroid which would be pulled toward center
        assert!(pole.x() < 1.5);
        assert!(pole.y() < 1.5);

        // Centroid should be more toward the center of the bounding box
        assert!(centroid.x() > pole.x());
        assert!(centroid.y() > pole.y());
    }

    #[test]
    fn test_pole_of_inaccessibility_circle() {
        // For a circle polygon, pole should be at center
        let circle = Circle::new(Point::new(3.0, 4.0), 5.0);
        let polygon = circle.polygonize(64);

        let pole = polygon.pole_of_inaccessibility(0.1);

        // Should be very close to circle center (3, 4)
        assert!((pole.x() - 3.0).abs() < 0.5);
        assert!((pole.y() - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_pole_degenerate_polygon() {
        // Triangle (should not panic)
        let polygon = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 1.0),
        ]);
        let pole = polygon.pole_of_inaccessibility(0.1);

        // Should be somewhere inside the triangle
        assert!(pole.x() >= 0.0 && pole.x() <= 1.0);
        assert!(pole.y() >= 0.0 && pole.y() <= 1.0);
    }

    #[test]
    fn test_pole_clockwise_matches_counterclockwise() {
        // Regression guard: polylabel-mini used to return (0, 0) for CW rings.
        // The in-house implementation is winding-agnostic.
        let ccw = vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 4.0),
            Point::new(0.0, 4.0),
        ];
        let cw: Vec<Point> = ccw.iter().rev().copied().collect();

        let (pole_ccw, dist_ccw) = Polygon::new(ccw).pole_of_inaccessibility_with_distance(0.01);
        let (pole_cw, dist_cw) = Polygon::new(cw).pole_of_inaccessibility_with_distance(0.01);

        assert!((pole_ccw.x() - pole_cw.x()).abs() < 0.02);
        assert!((pole_ccw.y() - pole_cw.y()).abs() < 0.02);
        assert!((dist_ccw - dist_cw).abs() < 0.02);
        assert!(dist_ccw > 0.0);
    }

    #[test]
    fn test_pole_closed_ring_matches_open_ring() {
        // The old code re-closed the ring before handing off; the rewrite
        // should be agnostic to whether the input already repeats vertex 0.
        let open = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let mut closed = open.clone();
        closed.push(Point::new(0.0, 0.0));

        let pole_open = Polygon::new(open).pole_of_inaccessibility(0.01);
        let pole_closed = Polygon::new(closed).pole_of_inaccessibility(0.01);

        assert!((pole_open.x() - pole_closed.x()).abs() < 0.02);
        assert!((pole_open.y() - pole_closed.y()).abs() < 0.02);
    }

    #[test]
    fn test_pole_distance_positive_and_bounded_by_inradius() {
        // 10x10 square: inradius is 5, so the pole's distance must be in (0, 5].
        let polygon = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let (_, dist) = polygon.pole_of_inaccessibility_with_distance(0.01);
        assert!(dist > 0.0);
        assert!(dist <= 5.0 + 1e-9);
        // Square's true inradius is 5; precision is 0.01, so dist should be close.
        assert!((dist - 5.0).abs() < 0.05);
    }

    #[test]
    fn test_pole_thin_sliver() {
        // Long thin rectangle: pole should sit on the long axis with
        // distance equal to half the short side.
        let polygon = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 0.1),
            Point::new(0.0, 0.1),
        ]);
        let (pole, dist) = polygon.pole_of_inaccessibility_with_distance(0.001);
        assert!((pole.y() - 0.05).abs() < 0.01);
        assert!(pole.x() > 0.05 && pole.x() < 9.95);
        assert!((dist - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_pole_concave_chevron() {
        // Concave chevron whose centroid lies outside the polygon — the pole
        // must end up inside and farther from the boundary than the centroid.
        let polygon = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
        ]);
        let (pole, dist) = polygon.pole_of_inaccessibility_with_distance(0.01);
        assert!(point_in_polygon(pole.x(), pole.y(), polygon.vertices()));
        let centroid = polygon.centroid();
        let centroid_dist = min_distance_to_boundary(&centroid, polygon.vertices());
        let centroid_dist_signed =
            if point_in_polygon(centroid.x(), centroid.y(), polygon.vertices()) {
                centroid_dist
            } else {
                -centroid_dist
            };
        assert!(dist > centroid_dist_signed);
    }
}
