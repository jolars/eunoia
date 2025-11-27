//! Conic sections in projective coordinates.

use crate::math::linear_algebra::Vector3Ext;
use std::cmp::Ordering;

use nalgebra::{ComplexField, Matrix3};
use num_complex::{Complex64, ComplexFloat};

use super::HomogeneousPoint;
use crate::{
    geometry::projective::HomogeneousLine,
    math::{
        linear_algebra::Matrix3Ext,
        polynomial::{self, extract_real_roots},
    },
};

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

    pub fn intersect_conic(&self, other: &Conic) -> Vec<HomogeneousPoint> {
        let a = &self.matrix;
        let m = &other.matrix;

        let alpha = a.determinant();
        let delta = m.determinant();
        let beta = Matrix3::from_columns(&[a.column(0), a.column(1), m.column(2)]).determinant()
            + Matrix3::from_columns(&[a.column(0), m.column(1), a.column(2)]).determinant()
            + Matrix3::from_columns(&[m.column(0), a.column(1), a.column(2)]).determinant();
        let gamma = Matrix3::from_columns(&[a.column(0), m.column(1), m.column(2)]).determinant()
            + Matrix3::from_columns(&[m.column(0), a.column(1), m.column(2)]).determinant()
            + Matrix3::from_columns(&[m.column(0), m.column(1), a.column(2)]).determinant();

        let roots = polynomial::solve_cubic(alpha, beta, gamma, delta);
        let real_roots = extract_real_roots(&roots, 1e-10);

        // Select the largest real root
        let lambda = real_roots.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Create the degenerate conic matrix
        let c = Conic::new((lambda * a + m).map(|x| if x.abs() < 1e-10 { 0.0 } else { x }));

        // Split the degenerate conic into two lines
        let (line1, line2) = c.split_degenerate();

        // Intersect each line with one of the conics to get intersection points
        let points1 = self.intersect_line(&line1);
        let points2 = self.intersect_line(&line2);

        points1.into_iter().chain(points2).collect()
    }

    /// Intersects this conic with a line to return 0 to 2 intersection points.
    ///
    /// # Arguments
    ///
    /// * `line` - The homogeneous line to intersect with
    ///
    /// # Returns
    ///
    /// A vector of 0, 1, or 2 intersection points
    pub fn intersect_line(&self, line: &HomogeneousLine) -> Vec<HomogeneousPoint> {
        const SMALL: f64 = 1e-10;

        let mut points = Vec::new();

        let m = line.coeffs().skew_symmetric_matrix();

        let b = m.transpose() * self.matrix * m;

        let l_abs = line.coeffs().abs();

        if l_abs.max() < SMALL {
            return points; // Line is degenerate
        }

        // Find index of maximum absolute value in line coefficients
        let i = l_abs.iamax();

        let b_sub = b.remove_row(i).remove_column(i);
        let det_sub = b_sub.determinant();
        let alpha = (-det_sub).sqrt() / line.coeffs()[i];

        let c = b + alpha * m;

        let (i0, i1) = c.iamax_full();

        if c[(i0, i1)].abs() <= SMALL {
            return points; // No intersection
        }

        // Extract row i0, normalize by element (i0, 2) to get first point
        if c[(i0, 2)].abs() > SMALL {
            let p0 = HomogeneousPoint::new(c[(i0, 0)] / c[(i0, 2)], c[(i0, 1)] / c[(i0, 2)], 1.0);
            points.push(p0);
        }

        // Extract column i1, normalize by element (2, i1) to get second point
        if c[(2, i1)].abs() > SMALL {
            let p1 = HomogeneousPoint::new(c[(0, i1)] / c[(2, i1)], c[(1, i1)] / c[(2, i1)], 1.0);
            points.push(p1);
        }

        points
    }

    fn split_degenerate(&self) -> (HomogeneousLine, HomogeneousLine) {
        let b = -self.matrix.adjugate();

        let b_diagonal = b.diagonal();
        let (i, max_val) = b_diagonal
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(Ordering::Equal))
            .unwrap(); // safe here because diagonal is non-empty

        let b_ii = Complex64::from(*max_val).sqrt();

        if b_ii.real() < 0.0 {
            panic!("Cannot split degenerate conic: Bii has negative real part.");
        }

        let b_i = b.column(i).map(Complex64::from).map(|x| x / b_ii);

        let skew = b_i.skew_symmetric_matrix();

        let c = self.matrix.map(Complex64::from) + skew;

        // Find the maximum absolute value element in C
        let (max_row, max_col) = c.map(|x| x.re()).iamax_full();

        let line1 =
            HomogeneousLine::new(c[(max_row, 0)].re, c[(max_row, 1)].re, c[(max_row, 2)].re);
        let line2 =
            HomogeneousLine::new(c[(0, max_col)].re, c[(1, max_col)].re, c[(2, max_col)].re);

        (line1, line2)
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
