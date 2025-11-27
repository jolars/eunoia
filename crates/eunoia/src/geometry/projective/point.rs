//! Homogeneous points for projective geometry.

use nalgebra::Vector3;

use crate::geometry::primitives::Point;

/// A point in homogeneous coordinates [x, y, w].
///
/// In projective geometry, homogeneous coordinates represent points using three
/// values where scalar multiples represent the same point: [x, y, w] ~ [kx, ky, kw].
///
/// # Representation
///
/// - **Finite points**: [x, y, w] with w ≠ 0 represents the Euclidean point (x/w, y/w)
/// - **Points at infinity**: [x, y, 0] represents a direction or ideal point
///
/// # Examples
///
/// ```
/// use eunoia::geometry::projective::HomogeneousPoint;
/// use eunoia::geometry::primitives::Point;
///
/// // Create from Euclidean coordinates
/// let euclidean = Point::new(2.0, 3.0);
/// let homogeneous = HomogeneousPoint::from_euclidean(euclidean);
///
/// // Convert back
/// assert_eq!(homogeneous.to_euclidean(), Some(euclidean));
///
/// // Point at infinity
/// let at_infinity = HomogeneousPoint::new(1.0, 0.0, 0.0);
/// assert!(at_infinity.is_at_infinity());
/// assert_eq!(at_infinity.to_euclidean(), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomogeneousPoint {
    coords: Vector3<f64>,
}

impl HomogeneousPoint {
    /// Creates a homogeneous point from coordinates [x, y, w].
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousPoint;
    ///
    /// // Finite point
    /// let p = HomogeneousPoint::new(4.0, 6.0, 2.0);
    /// assert_eq!(p.to_euclidean().unwrap().x(), 2.0);
    /// assert_eq!(p.to_euclidean().unwrap().y(), 3.0);
    ///
    /// // Point at infinity
    /// let inf = HomogeneousPoint::new(1.0, 1.0, 0.0);
    /// assert!(inf.is_at_infinity());
    /// ```
    pub fn new(x: f64, y: f64, w: f64) -> Self {
        Self {
            coords: Vector3::new(x, y, w),
        }
    }

    /// Creates a homogeneous point from a Euclidean point.
    ///
    /// The Euclidean point (x, y) is represented as [x, y, 1] in homogeneous coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousPoint;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let euclidean = Point::new(3.0, 4.0);
    /// let homogeneous = HomogeneousPoint::from_euclidean(euclidean);
    ///
    /// assert_eq!(homogeneous.x(), 3.0);
    /// assert_eq!(homogeneous.y(), 4.0);
    /// assert_eq!(homogeneous.w(), 1.0);
    /// ```
    pub fn from_euclidean(point: Point) -> Self {
        Self::new(point.x(), point.y(), 1.0)
    }

    /// Converts to a Euclidean point by dehomogenization.
    ///
    /// Returns `None` if the point is at infinity (w ≈ 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousPoint;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let h = HomogeneousPoint::new(6.0, 9.0, 3.0);
    /// assert_eq!(h.to_euclidean(), Some(Point::new(2.0, 3.0)));
    ///
    /// let inf = HomogeneousPoint::new(1.0, 2.0, 0.0);
    /// assert_eq!(inf.to_euclidean(), None);
    /// ```
    pub fn to_euclidean(&self) -> Option<Point> {
        if self.is_at_infinity() {
            None
        } else {
            Some(Point::new(
                self.coords[0] / self.coords[2],
                self.coords[1] / self.coords[2],
            ))
        }
    }

    /// Returns true if this is a point at infinity.
    ///
    /// A point is at infinity when its w coordinate is approximately zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousPoint;
    ///
    /// let finite = HomogeneousPoint::new(1.0, 2.0, 1.0);
    /// assert!(!finite.is_at_infinity());
    ///
    /// let infinite = HomogeneousPoint::new(1.0, 0.0, 0.0);
    /// assert!(infinite.is_at_infinity());
    /// ```
    pub fn is_at_infinity(&self) -> bool {
        self.coords[2].abs() < 1e-10
    }

    /// Normalizes the homogeneous coordinates so that w = 1.
    ///
    /// If the point is at infinity, returns the point unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::projective::HomogeneousPoint;
    ///
    /// let p = HomogeneousPoint::new(4.0, 6.0, 2.0);
    /// let normalized = p.normalize();
    ///
    /// assert_eq!(normalized.x(), 2.0);
    /// assert_eq!(normalized.y(), 3.0);
    /// assert_eq!(normalized.w(), 1.0);
    /// ```
    pub fn normalize(&self) -> Self {
        if self.is_at_infinity() {
            *self
        } else {
            Self::new(
                self.coords[0] / self.coords[2],
                self.coords[1] / self.coords[2],
                1.0,
            )
        }
    }

    /// Returns the x coordinate.
    pub fn x(&self) -> f64 {
        self.coords[0]
    }

    /// Returns the y coordinate.
    pub fn y(&self) -> f64 {
        self.coords[1]
    }

    /// Returns the w coordinate (homogeneous scaling factor).
    pub fn w(&self) -> f64 {
        self.coords[2]
    }

    /// Returns the raw homogeneous coordinates as a vector [x, y, w].
    pub fn coords(&self) -> &Vector3<f64> {
        &self.coords
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
        let p = HomogeneousPoint::new(2.0, 3.0, 1.0);
        assert_eq!(p.x(), 2.0);
        assert_eq!(p.y(), 3.0);
        assert_eq!(p.w(), 1.0);
    }

    #[test]
    fn test_from_euclidean() {
        let euclidean = Point::new(5.0, 7.0);
        let homogeneous = HomogeneousPoint::from_euclidean(euclidean);

        assert_eq!(homogeneous.x(), 5.0);
        assert_eq!(homogeneous.y(), 7.0);
        assert_eq!(homogeneous.w(), 1.0);
    }

    #[test]
    fn test_to_euclidean() {
        let h = HomogeneousPoint::new(6.0, 9.0, 3.0);
        let euclidean = h.to_euclidean().unwrap();

        assert!(approx_eq(euclidean.x(), 2.0));
        assert!(approx_eq(euclidean.y(), 3.0));
    }

    #[test]
    fn test_to_euclidean_normalized() {
        let h = HomogeneousPoint::new(2.0, 3.0, 1.0);
        let euclidean = h.to_euclidean().unwrap();

        assert_eq!(euclidean.x(), 2.0);
        assert_eq!(euclidean.y(), 3.0);
    }

    #[test]
    fn test_is_at_infinity() {
        let finite = HomogeneousPoint::new(1.0, 2.0, 1.0);
        assert!(!finite.is_at_infinity());

        let infinite = HomogeneousPoint::new(1.0, 0.0, 0.0);
        assert!(infinite.is_at_infinity());

        let nearly_infinite = HomogeneousPoint::new(1.0, 2.0, 1e-12);
        assert!(nearly_infinite.is_at_infinity());
    }

    #[test]
    fn test_to_euclidean_infinity() {
        let inf = HomogeneousPoint::new(1.0, 2.0, 0.0);
        assert_eq!(inf.to_euclidean(), None);
    }

    #[test]
    fn test_normalize() {
        let p = HomogeneousPoint::new(4.0, 6.0, 2.0);
        let normalized = p.normalize();

        assert!(approx_eq(normalized.x(), 2.0));
        assert!(approx_eq(normalized.y(), 3.0));
        assert!(approx_eq(normalized.w(), 1.0));
    }

    #[test]
    fn test_normalize_infinity() {
        let inf = HomogeneousPoint::new(1.0, 2.0, 0.0);
        let normalized = inf.normalize();

        assert_eq!(normalized.x(), 1.0);
        assert_eq!(normalized.y(), 2.0);
        assert_eq!(normalized.w(), 0.0);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = Point::new(3.5, -2.5);
        let homogeneous = HomogeneousPoint::from_euclidean(original);
        let recovered = homogeneous.to_euclidean().unwrap();

        assert!(approx_eq(recovered.x(), original.x()));
        assert!(approx_eq(recovered.y(), original.y()));
    }

    #[test]
    fn test_equivalence_under_scaling() {
        let p1 = HomogeneousPoint::new(2.0, 3.0, 1.0);
        let p2 = HomogeneousPoint::new(4.0, 6.0, 2.0);
        let p3 = HomogeneousPoint::new(6.0, 9.0, 3.0);

        let e1 = p1.to_euclidean().unwrap();
        let e2 = p2.to_euclidean().unwrap();
        let e3 = p3.to_euclidean().unwrap();

        assert!(approx_eq(e1.x(), e2.x()));
        assert!(approx_eq(e1.y(), e2.y()));
        assert!(approx_eq(e2.x(), e3.x()));
        assert!(approx_eq(e2.y(), e3.y()));
    }
}
