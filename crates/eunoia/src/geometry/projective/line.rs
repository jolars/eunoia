//! Homogeneous lines for projective geometry.

use nalgebra::Vector3;

use super::HomogeneousPoint;

/// A line in homogeneous coordinates [a, b, c].
///
/// In projective geometry, a line is represented by three coefficients where
/// scalar multiples represent the same line: [a, b, c] ~ [ka, kb, kc].
///
/// # Representation
///
/// The line [a, b, c] represents the set of points satisfying:
/// ```text
/// ax + by + cw = 0
/// ```
///
/// For Euclidean points where w = 1, this becomes the familiar ax + by + c = 0.
///
/// # Line at Infinity
///
/// The line [0, 0, 1] represents the "line at infinity" containing all ideal points.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::projective::{HomogeneousLine, HomogeneousPoint};
///
/// // Create a line: x - 2y + 3 = 0
/// let line = HomogeneousLine::new(1.0, -2.0, 3.0);
///
/// // Create line through two points
/// let p1 = HomogeneousPoint::new(1.0, 0.0, 1.0);
/// let p2 = HomogeneousPoint::new(0.0, 1.0, 1.0);
/// let line = HomogeneousLine::through_points(&p1, &p2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomogeneousLine {
    coeffs: Vector3<f64>,
}

impl HomogeneousLine {
    /// Creates a line from coefficients [a, b, c].
    ///
    /// The line represents ax + by + cw = 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousLine;
    ///
    /// // Line x - y + 1 = 0
    /// let line = HomogeneousLine::new(1.0, -1.0, 1.0);
    /// ```
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self {
            coeffs: Vector3::new(a, b, c),
        }
    }

    /// Creates a line passing through two homogeneous points.
    ///
    /// Uses the cross product: the line through points p₁ and p₂ is p₁ × p₂.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::{HomogeneousLine, HomogeneousPoint};
    ///
    /// let p1 = HomogeneousPoint::new(1.0, 0.0, 1.0);  // (1, 0)
    /// let p2 = HomogeneousPoint::new(0.0, 1.0, 1.0);  // (0, 1)
    /// let line = HomogeneousLine::through_points(&p1, &p2);
    ///
    /// // Both points should lie on the line
    /// assert!(line.contains(&p1));
    /// assert!(line.contains(&p2));
    /// ```
    pub fn through_points(p1: &HomogeneousPoint, p2: &HomogeneousPoint) -> Self {
        let cross = p1.coords().cross(p2.coords());
        Self { coeffs: cross }
    }

    /// Computes the intersection point of two lines.
    ///
    /// Uses the cross product: the intersection of lines ℓ₁ and ℓ₂ is ℓ₁ × ℓ₂.
    ///
    /// # Returns
    ///
    /// - Returns a finite point if the lines intersect at a regular point
    /// - Returns a point at infinity if the lines are parallel
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousLine;
    ///
    /// // Line 1: x + y = 1  →  x + y - 1 = 0
    /// let l1 = HomogeneousLine::new(1.0, 1.0, -1.0);
    ///
    /// // Line 2: x - y = 0
    /// let l2 = HomogeneousLine::new(1.0, -1.0, 0.0);
    ///
    /// let intersection = l1.intersect(&l2);
    /// let euclidean = intersection.to_euclidean().unwrap();
    ///
    /// // Should intersect at (0.5, 0.5)
    /// assert!((euclidean.x() - 0.5).abs() < 1e-10);
    /// assert!((euclidean.y() - 0.5).abs() < 1e-10);
    /// ```
    pub fn intersect(&self, other: &HomogeneousLine) -> HomogeneousPoint {
        let cross = self.coeffs.cross(&other.coeffs);
        HomogeneousPoint::new(cross[0], cross[1], cross[2])
    }

    /// Tests if a point lies on this line.
    ///
    /// A point [x, y, w] lies on the line [a, b, c] if ax + by + cw ≈ 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::{HomogeneousLine, HomogeneousPoint};
    ///
    /// let line = HomogeneousLine::new(1.0, 1.0, -2.0);  // x + y - 2 = 0
    /// let p1 = HomogeneousPoint::new(1.0, 1.0, 1.0);    // (1, 1)
    /// let p2 = HomogeneousPoint::new(0.0, 2.0, 1.0);    // (0, 2)
    /// let p3 = HomogeneousPoint::new(0.0, 0.0, 1.0);    // (0, 0)
    ///
    /// assert!(line.contains(&p1));
    /// assert!(line.contains(&p2));
    /// assert!(!line.contains(&p3));
    /// ```
    pub fn contains(&self, point: &HomogeneousPoint) -> bool {
        let dot = self.coeffs.dot(point.coords());
        dot.abs() < 1e-10
    }

    /// Tests if two lines are parallel.
    ///
    /// Lines are parallel if their intersection point is at infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousLine;
    ///
    /// // Both lines have slope 1 (parallel)
    /// let l1 = HomogeneousLine::new(1.0, -1.0, 0.0);  // x - y = 0
    /// let l2 = HomogeneousLine::new(1.0, -1.0, 1.0);  // x - y + 1 = 0
    ///
    /// assert!(l1.is_parallel(&l2));
    ///
    /// // Non-parallel lines
    /// let l3 = HomogeneousLine::new(1.0, 1.0, 0.0);   // x + y = 0
    /// assert!(!l1.is_parallel(&l3));
    /// ```
    pub fn is_parallel(&self, other: &HomogeneousLine) -> bool {
        self.intersect(other).is_at_infinity()
    }

    /// Normalizes the line coefficients.
    ///
    /// Scales the coefficients so that a² + b² = 1 (when possible).
    /// The line at infinity [0, 0, c] is returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousLine;
    ///
    /// let line = HomogeneousLine::new(3.0, 4.0, 5.0);
    /// let normalized = line.normalize();
    ///
    /// // a² + b² should equal 1
    /// let sum_squares = normalized.a().powi(2) + normalized.b().powi(2);
    /// assert!((sum_squares - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalize(&self) -> Self {
        let norm = (self.coeffs[0].powi(2) + self.coeffs[1].powi(2)).sqrt();
        if norm < 1e-10 {
            // Line at infinity
            *self
        } else {
            Self::new(
                self.coeffs[0] / norm,
                self.coeffs[1] / norm,
                self.coeffs[2] / norm,
            )
        }
    }

    /// Returns the a coefficient.
    pub fn a(&self) -> f64 {
        self.coeffs[0]
    }

    /// Returns the b coefficient.
    pub fn b(&self) -> f64 {
        self.coeffs[1]
    }

    /// Returns the c coefficient.
    pub fn c(&self) -> f64 {
        self.coeffs[2]
    }

    /// Returns the raw coefficients as a vector [a, b, c].
    pub fn coeffs(&self) -> &Vector3<f64> {
        &self.coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_new() {
        let line = HomogeneousLine::new(1.0, 2.0, 3.0);
        assert_eq!(line.a(), 1.0);
        assert_eq!(line.b(), 2.0);
        assert_eq!(line.c(), 3.0);
    }

    #[test]
    fn test_through_points() {
        let p1 = HomogeneousPoint::new(1.0, 0.0, 1.0); // (1, 0)
        let p2 = HomogeneousPoint::new(0.0, 1.0, 1.0); // (0, 1)

        let line = HomogeneousLine::through_points(&p1, &p2);

        // Both points should lie on the line
        assert!(line.contains(&p1));
        assert!(line.contains(&p2));
    }

    #[test]
    fn test_contains() {
        let line = HomogeneousLine::new(1.0, 1.0, -2.0); // x + y - 2 = 0

        let p1 = HomogeneousPoint::new(1.0, 1.0, 1.0); // (1, 1)
        let p2 = HomogeneousPoint::new(0.0, 2.0, 1.0); // (0, 2)
        let p3 = HomogeneousPoint::new(2.0, 0.0, 1.0); // (2, 0)
        let p4 = HomogeneousPoint::new(0.0, 0.0, 1.0); // (0, 0)

        assert!(line.contains(&p1));
        assert!(line.contains(&p2));
        assert!(line.contains(&p3));
        assert!(!line.contains(&p4));
    }

    #[test]
    fn test_intersect() {
        // Line 1: x + y = 1
        let l1 = HomogeneousLine::new(1.0, 1.0, -1.0);

        // Line 2: x - y = 0
        let l2 = HomogeneousLine::new(1.0, -1.0, 0.0);

        let intersection = l1.intersect(&l2);
        let euclidean = intersection.to_euclidean().unwrap();

        // Should intersect at (0.5, 0.5)
        assert!(approx_eq(euclidean.x(), 0.5));
        assert!(approx_eq(euclidean.y(), 0.5));
    }

    #[test]
    fn test_intersect_parallel() {
        // Parallel lines: x - y = 0 and x - y + 1 = 0
        let l1 = HomogeneousLine::new(1.0, -1.0, 0.0);
        let l2 = HomogeneousLine::new(1.0, -1.0, 1.0);

        let intersection = l1.intersect(&l2);
        assert!(intersection.is_at_infinity());
    }

    #[test]
    fn test_is_parallel() {
        let l1 = HomogeneousLine::new(1.0, -1.0, 0.0); // x - y = 0
        let l2 = HomogeneousLine::new(1.0, -1.0, 1.0); // x - y + 1 = 0
        let l3 = HomogeneousLine::new(1.0, 1.0, 0.0); // x + y = 0

        assert!(l1.is_parallel(&l2));
        assert!(!l1.is_parallel(&l3));
    }

    #[test]
    fn test_normalize() {
        let line = HomogeneousLine::new(3.0, 4.0, 5.0);
        let normalized = line.normalize();

        // a² + b² should equal 1
        let sum_squares = normalized.a().powi(2) + normalized.b().powi(2);
        assert!(approx_eq(sum_squares, 1.0));

        // c should be scaled by the same factor
        let scale = (3.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt();
        assert!(approx_eq(normalized.c(), 5.0 / scale));
    }

    #[test]
    fn test_through_points_contains_both() {
        let p1 = HomogeneousPoint::new(2.0, 3.0, 1.0);
        let p2 = HomogeneousPoint::new(-1.0, 5.0, 1.0);

        let line = HomogeneousLine::through_points(&p1, &p2);

        assert!(line.contains(&p1));
        assert!(line.contains(&p2));
    }

    #[test]
    fn test_line_point_duality() {
        // Create two lines
        let l1 = HomogeneousLine::new(1.0, 0.0, -1.0); // x = 1
        let l2 = HomogeneousLine::new(0.0, 1.0, -1.0); // y = 1

        // Their intersection
        let p = l1.intersect(&l2);

        // The line through this point and another point
        let p2 = HomogeneousPoint::new(2.0, 2.0, 1.0);
        let l3 = HomogeneousLine::through_points(&p, &p2);

        // Original point should lie on new line
        assert!(l3.contains(&p));
        assert!(l3.contains(&p2));
    }

    #[test]
    fn test_equivalence_under_scaling() {
        let l1 = HomogeneousLine::new(1.0, 2.0, 3.0);
        let l2 = HomogeneousLine::new(2.0, 4.0, 6.0);
        let l3 = HomogeneousLine::new(0.5, 1.0, 1.5);

        let p = HomogeneousPoint::new(1.0, -2.0, 1.0);

        // All scaled versions should have same containment
        assert_eq!(l1.contains(&p), l2.contains(&p));
        assert_eq!(l1.contains(&p), l3.contains(&p));
    }
}
