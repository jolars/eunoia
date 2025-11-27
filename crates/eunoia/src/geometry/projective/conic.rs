//! Conic sections in projective coordinates.

use nalgebra::Matrix3;

use super::HomogeneousPoint;

/// A conic section represented as a 3×3 symmetric matrix.
///
/// In projective geometry, a conic is the set of points [x, y, w] satisfying:
/// ```text
/// [x, y, w]ᵀ C [x, y, w] = 0
/// ```
/// where C is a 3×3 symmetric matrix.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::projective::Conic;
/// use nalgebra::Matrix3;
///
/// // Unit circle: x² + y² - 1 = 0
/// let matrix = Matrix3::new(
///     1.0, 0.0, 0.0,
///     0.0, 1.0, 0.0,
///     0.0, 0.0, -1.0,
/// );
/// let conic = Conic::new(matrix);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Conic {
    matrix: Matrix3<f64>,
}

impl Conic {
    /// Creates a conic from a 3×3 matrix.
    ///
    /// The matrix should be symmetric for a proper conic section.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::Conic;
    /// use nalgebra::Matrix3;
    ///
    /// let matrix = Matrix3::identity();
    /// let conic = Conic::new(matrix);
    /// ```
    pub fn new(matrix: Matrix3<f64>) -> Self {
        Self { matrix }
    }

    /// Returns the conic matrix.
    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.matrix
    }

    /// Tests if a point lies on the conic.
    ///
    /// A point p = [x, y, w] lies on the conic if pᵀCp ≈ 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::{Conic, HomogeneousPoint};
    /// use nalgebra::Matrix3;
    ///
    /// // Unit circle: x² + y² = 1
    /// let circle = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, -1.0,
    /// );
    /// let conic = Conic::new(circle);
    ///
    /// let p1 = HomogeneousPoint::new(1.0, 0.0, 1.0);  // (1, 0)
    /// let p2 = HomogeneousPoint::new(0.0, 1.0, 1.0);  // (0, 1)
    /// let p3 = HomogeneousPoint::new(0.0, 0.0, 1.0);  // (0, 0)
    ///
    /// assert!(conic.contains(&p1));
    /// assert!(conic.contains(&p2));
    /// assert!(!conic.contains(&p3));
    /// ```
    pub fn contains(&self, point: &HomogeneousPoint) -> bool {
        let p = point.coords();
        let value = p.transpose() * self.matrix * p;
        value[(0, 0)].abs() < 1e-10
    }

    /// Computes the dual conic (adjugate of the matrix).
    ///
    /// The dual conic C* satisfies the property that a line ℓ is tangent to
    /// the conic C if and only if ℓᵀC*ℓ = 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::Conic;
    /// use nalgebra::Matrix3;
    ///
    /// let matrix = Matrix3::identity();
    /// let conic = Conic::new(matrix);
    /// let dual = conic.dual();
    /// ```
    pub fn dual(&self) -> Self {
        use crate::math::linear_algebra::Matrix3Ext;
        Self::new(self.matrix.adjugate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let matrix = Matrix3::identity();
        let conic = Conic::new(matrix);
        assert_eq!(conic.matrix(), &matrix);
    }

    #[test]
    fn test_contains_unit_circle() {
        // Unit circle: x² + y² - 1 = 0
        let circle = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let conic = Conic::new(circle);

        // Points on the circle
        let p1 = HomogeneousPoint::new(1.0, 0.0, 1.0); // (1, 0)
        let p2 = HomogeneousPoint::new(0.0, 1.0, 1.0); // (0, 1)
        let p3 = HomogeneousPoint::new(-1.0, 0.0, 1.0); // (-1, 0)

        assert!(conic.contains(&p1));
        assert!(conic.contains(&p2));
        assert!(conic.contains(&p3));

        // Point not on the circle
        let p4 = HomogeneousPoint::new(0.0, 0.0, 1.0); // (0, 0)
        assert!(!conic.contains(&p4));
    }

    #[test]
    fn test_dual() {
        let matrix = Matrix3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -6.0);
        let conic = Conic::new(matrix);
        let dual = conic.dual();

        // Dual should exist
        assert!(dual.matrix()[(0, 0)].is_finite());
    }

    #[test]
    fn test_contains_scaled_point() {
        // Unit circle
        let circle = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let conic = Conic::new(circle);

        // Scaled versions of (1, 0) should all lie on circle
        let p1 = HomogeneousPoint::new(1.0, 0.0, 1.0);
        let p2 = HomogeneousPoint::new(2.0, 0.0, 2.0);
        let p3 = HomogeneousPoint::new(0.5, 0.0, 0.5);

        assert!(conic.contains(&p1));
        assert!(conic.contains(&p2));
        assert!(conic.contains(&p3));
    }
}
