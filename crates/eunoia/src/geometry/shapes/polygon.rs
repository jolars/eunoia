//! Polygon shape for visualization and export.
//!
//! Polygons are used to represent circles and ellipses as discrete vertices
//! for plotting and export to other formats. They are not used in the core
//! diagram computation.

use crate::geometry::primitives::Point;

#[cfg(feature = "plotting")]
use polylabel_mini::polylabel;

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
    #[cfg(feature = "plotting")]
    pub fn pole_of_inaccessibility(&self, precision: f64) -> Point {
        if self.vertices.len() < 3 {
            // Degenerate case - return centroid
            return self.centroid();
        }

        // Convert to polylabel-mini types
        let mut points: Vec<polylabel_mini::Point> = self
            .vertices
            .iter()
            .map(|p| polylabel_mini::Point { x: p.x(), y: p.y() })
            .collect();

        // Ensure the polygon is closed (first == last)
        if let (Some(first), Some(last)) = (self.vertices.first(), self.vertices.last()) {
            if (first.x() - last.x()).abs() > 1e-10 || (first.y() - last.y()).abs() > 1e-10 {
                points.push(polylabel_mini::Point {
                    x: first.x(),
                    y: first.y(),
                });
            }
        }

        let exterior = polylabel_mini::LineString { points };
        let poly = polylabel_mini::Polygon {
            exterior,
            interiors: vec![], // No holes
        };

        // Call polylabel
        let pole = polylabel(&poly, precision);

        Point::new(pole.x, pole.y)
    }
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
    #[cfg(feature = "plotting")]
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
    #[cfg(feature = "plotting")]
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
    #[cfg(feature = "plotting")]
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
    #[cfg(feature = "plotting")]
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
}
