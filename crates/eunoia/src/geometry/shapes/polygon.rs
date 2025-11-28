//! Polygon shape for visualization and export.
//!
//! Polygons are used to represent circles and ellipses as discrete vertices
//! for plotting and export to other formats. They are not used in the core
//! diagram computation.

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
}
