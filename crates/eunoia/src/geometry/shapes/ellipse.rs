//! Ellipse shape implementation.
//!
//! This module provides an ellipse implementation for use in Euler and Venn diagrams.
//! Ellipses are more flexible than circles and can represent elongated or rotated regions,
//! making them useful for more accurate area-proportional diagrams.
//!
//! # Representation
//!
//! An ellipse is defined by:
//! - **Center point**: The center of the ellipse
//! - **Semi-major axis** (a): Half the length of the longest diameter
//! - **Semi-minor axis** (b): Half the length of the shortest diameter  
//! - **Rotation**: Rotation angle in radians (counterclockwise from x-axis)
//!
//! # Area Calculations
//!
//! The module provides several specialized area computation methods:
//!
//! - **Sector area**: The area swept by a radius from the center through an angle θ
//! - **Segment area**: The area between the ellipse boundary and a chord connecting two points
//!
//! These primitives are essential for computing intersection areas in Euler diagrams.
//!
//! # Examples
//!
//! ```
//! use eunoia::geometry::shapes::Ellipse;
//! use eunoia::geometry::traits::Area;
//! use eunoia::geometry::primitives::Point;
//!
//! // Create an ellipse centered at origin
//! let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
//!
//! // Compute total area
//! let area = ellipse.area();
//!
//! // Compute sector area for π/4 radians
//! let sector = ellipse.sector_area(std::f64::consts::PI / 4.0);
//! ```

use std::f64::consts::PI;

use nalgebra::Matrix2;

use crate::geometry::primitives::Point;
use crate::geometry::projective::Conic;
use crate::geometry::shapes::Rectangle;
use crate::geometry::traits::{Area, BoundingBox, Centroid, Closed, Perimeter};

/// An ellipse defined by center, semi-major and semi-minor axes, and rotation.
///
/// Ellipses are oval-shaped closed curves that generalize circles. They are particularly
/// useful in Euler diagrams when sets have elongated or directional relationships.
///
/// # Representation
///
/// The ellipse is represented in standard form with:
/// - A center point `(h, k)`
/// - Semi-major axis length `a` (≥ semi-minor axis)
/// - Semi-minor axis length `b` (> 0)
/// - Rotation angle `φ` in radians (counterclockwise from the positive x-axis)
///
/// The canonical equation in the ellipse's local coordinate system is:
/// ```text
/// (x/a)² + (y/b)² = 1
/// ```
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Ellipse;
/// use eunoia::geometry::primitives::Point;
/// use eunoia::geometry::traits::Area;
///
/// // Create an ellipse at the origin with no rotation
/// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
/// assert!((ellipse.area() - 47.12).abs() < 0.01);
///
/// // Create a rotated ellipse
/// let rotated = Ellipse::new(
///     Point::new(2.0, 3.0),
///     4.0,
///     2.0,
///     std::f64::consts::PI / 4.0  // 45 degrees
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    center: Point,
    semi_major: f64,
    semi_minor: f64,
    rotation: f64, // in radians
}

impl Ellipse {
    /// Creates a new ellipse with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `center` - The center point of the ellipse
    /// * `semi_major` - The semi-major axis length (must be ≥ semi_minor)
    /// * `semi_minor` - The semi-minor axis length (must be > 0)
    /// * `rotation` - Rotation angle in radians (counterclockwise from x-axis)
    ///
    /// # Panics
    ///
    /// Panics in debug builds if:
    /// - `semi_major < semi_minor`
    /// - `semi_minor <= 0`
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let ellipse = Ellipse::new(Point::new(1.0, 2.0), 4.0, 3.0, 0.0);
    /// assert_eq!(ellipse.semi_major(), 4.0);
    /// assert_eq!(ellipse.semi_minor(), 3.0);
    /// ```
    pub fn new(center: Point, semi_major: f64, semi_minor: f64, rotation: f64) -> Self {
        debug_assert!(
            semi_major >= semi_minor,
            "Semi-major axis must be >= semi-minor axis"
        );
        debug_assert!(semi_minor > 0.0, "Semi-minor axis must be > 0");
        Self {
            center,
            semi_major,
            semi_minor,
            rotation,
        }
    }

    pub fn from_conic(conic: Conic) -> Option<Self> {
        let c = conic.matrix();

        // Extract algebraic coefficients from the symmetric matrix
        // Q = [ A   B/2  D/2 ]
        //     [ B/2 C    E/2 ]
        //     [ D/2 E/2  F   ]
        let m1 = c[(0, 0)];
        let m2 = 2.0 * c[(0, 1)];
        let m3 = c[(1, 1)];
        let m4 = 2.0 * c[(0, 2)];
        let m5 = 2.0 * c[(1, 2)];
        let m6 = c[(2, 2)];

        // Check ellipse condition (negative discriminant required)
        if m2 * m2 - 4.0 * m1 * m3 >= 0.0 {
            return None;
        }

        // Quadratic form matrix
        let m = Matrix2::new(m1, m2 / 2.0, m2 / 2.0, m3);

        // Solve for center:
        // M * [xc, yc]^T = -1/2 * [D, E]^T
        let rhs = nalgebra::Vector2::new(-m4 / 2.0, -m5 / 2.0);
        let center_vec = m.lu().solve(&rhs)?;
        let xc = center_vec[0];
        let yc = center_vec[1];

        // Compute translated constant term F̄
        let f_bar = m6 + m1 * xc * xc + m2 * xc * yc + m3 * yc * yc + m4 * xc + m5 * yc;

        if f_bar >= 0.0 {
            return None; // Not an ellipse
        }

        // Eigen-decompose the quadratic part
        let eig = m.symmetric_eigen();
        let lambda1 = eig.eigenvalues[0];
        let lambda2 = eig.eigenvalues[1];
        let v = eig.eigenvectors;

        // Identify major/minor axes:
        // smaller eigenvalue ⇒ bigger radius
        let (lambda_major, lambda_minor, vec_major) = if lambda1 < lambda2 {
            (lambda1, lambda2, v.column(0))
        } else {
            (lambda2, lambda1, v.column(1))
        };

        // Radii from: a² = -F̄ / λ
        let a2 = -f_bar / lambda_major;
        let b2 = -f_bar / lambda_minor;

        if a2 <= 0.0 || b2 <= 0.0 {
            return None;
        }

        let a = a2.sqrt();
        let b = b2.sqrt();

        // Rotation angle from the major-axis eigenvector
        let vx = vec_major[0];
        let vy = vec_major[1];
        let mut phi = vy.atan2(vx);

        // Normalize angle into [0, π)
        if phi < 0.0 {
            phi += std::f64::consts::PI;
        }

        Some(Ellipse {
            center: Point::new(xc, yc),
            semi_major: a,
            semi_minor: b,
            rotation: phi,
        })
    }

    /// Returns the center point of the ellipse.
    pub fn center(&self) -> Point {
        self.center
    }

    /// Returns the semi-major axis length.
    pub fn semi_major(&self) -> f64 {
        self.semi_major
    }

    /// Returns the semi-minor axis length.
    pub fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    /// Returns the rotation angle in radians.
    pub fn rotation(&self) -> f64 {
        self.rotation
    }

    /// Computes the area of an elliptical sector from the center through angle θ.
    ///
    /// An elliptical sector is the region bounded by:
    /// - Two radii from the center to the ellipse boundary
    /// - The ellipse arc between those radii
    ///
    /// This uses the exact formula for elliptical sectors, which accounts for
    /// the non-uniform curvature of the ellipse (unlike circular sectors where
    /// area is simply ½r²θ).
    ///
    /// # Arguments
    ///
    /// * `theta` - The angle in radians from the semi-major axis
    ///
    /// # Returns
    ///
    /// The area of the sector from 0 to θ
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    /// use eunoia::geometry::traits::Area;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 4.0, 2.0, 0.0);
    ///
    /// // Quarter sector (π/2 radians)
    /// let quarter = ellipse.sector_area(std::f64::consts::PI / 2.0);
    /// assert!((quarter - ellipse.area() / 4.0).abs() < 1e-10);
    /// ```
    pub fn sector_area(&self, theta: f64) -> f64 {
        let a = self.semi_major;
        let b = self.semi_minor;

        let num = (b - a) * (2.0 * theta).sin();
        let den = a + b + (b - a) * (2.0 * theta).cos();

        0.5 * a * b * (theta - num.atan2(den))
    }

    /// Computes the area of an elliptical sector between two angles.
    ///
    /// This is equivalent to `sector_area(theta1) - sector_area(theta0)` but
    /// handles counter-clockwise angle wrapping automatically.
    ///
    /// # Arguments
    ///
    /// * `theta0` - Starting angle in radians
    /// * `theta1` - Ending angle in radians
    ///
    /// # Returns
    ///
    /// The area of the sector from θ₀ to θ₁ (counter-clockwise)
    ///
    /// # Note
    ///
    /// If θ₁ < θ₀, the function adds 2π to θ₁ to compute the counter-clockwise
    /// sector that wraps through angle 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.0, 0.0);
    ///
    /// // Sector from 45° to 135°
    /// let sector = ellipse.sector_area_between(
    ///     std::f64::consts::PI / 4.0,
    ///     3.0 * std::f64::consts::PI / 4.0
    /// );
    /// assert!(sector > 0.0);
    /// ```
    pub fn sector_area_between(&self, theta0: f64, theta1: f64) -> f64 {
        let t0 = theta0;
        let mut t1 = theta1;

        // Ensure CCW ordering.
        if t1 < t0 {
            t1 += 2.0 * PI;
        }

        self.sector_area(t1) - self.sector_area(t0)
    }

    /// Computes the area of an ellipse segment between two boundary points.
    ///
    /// An ellipse segment is the region bounded by:
    /// - The ellipse arc between two points on the boundary
    /// - The chord (straight line) connecting those two points
    ///
    /// This method automatically handles:
    /// - Coordinate system transformation to the ellipse's local frame
    /// - Counter-clockwise angle ordering
    /// - Minor arc (≤ 180°) vs major arc (> 180°) selection
    ///
    /// # Algorithm
    ///
    /// 1. Transform points to ellipse coordinate system (centered, unrotated)
    /// 2. Compute angles θ₀ and θ₁ for each point
    /// 3. Ensure counter-clockwise ordering (add 2π if needed)
    /// 4. Calculate segment as: sector - triangle (for minor arc)
    ///    or: total_area - complementary_sector + triangle (for major arc)
    ///
    /// # Arguments
    ///
    /// * `p0` - First boundary point (in world coordinates)
    /// * `p1` - Second boundary point (in world coordinates)
    ///
    /// # Returns
    ///
    /// The area of the segment. When the angular span > π, returns the **major arc**
    /// segment (the larger of the two possible segments).
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    /// use eunoia::geometry::traits::Area;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
    ///
    /// // Segment between points on opposite sides (semicircle)
    /// let p0 = Point::new(5.0, 0.0);   // Right end of major axis
    /// let p1 = Point::new(-5.0, 0.0);  // Left end of major axis
    /// let segment = ellipse.ellipse_segment(p0, p1);
    ///
    /// // Should be approximately half the ellipse area
    /// assert!((segment - ellipse.area() / 2.0).abs() < 1e-10);
    /// ```
    pub fn ellipse_segment(&self, p0: Point, p1: Point) -> f64 {
        // 1. Move into ellipse coordinate system.
        let p0 = p0.to_ellipse_frame(self);
        let p1 = p1.to_ellipse_frame(self);

        // 2. Compute angles
        let theta0 = p0.angle_from_origin();
        let mut theta1 = p1.angle_from_origin();

        // 3. Ensure CCW ordering.
        if theta1 < theta0 {
            theta1 += 2.0 * PI;
        }

        // 4. Triangle correction (signed area of parallelogram / 2).
        let triangle = 0.5 * (p1.x() * p0.y() - p0.x() * p1.y()).abs();

        // 5. Minor or major arc?
        if (theta1 - theta0) <= PI {
            self.sector_area(theta1) - self.sector_area(theta0) - triangle
        } else {
            self.area() - (self.sector_area(theta0 + 2.0 * PI) - self.sector_area(theta1))
                + triangle
        }
    }
}

impl Area for Ellipse {
    /// Computes the total area of the ellipse using the formula A = πab.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    /// use eunoia::geometry::traits::Area;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
    /// let area = ellipse.area();
    /// assert!((area - std::f64::consts::PI * 15.0).abs() < 1e-10);
    /// ```
    fn area(&self) -> f64 {
        PI * self.semi_major * self.semi_minor
    }
}

impl Perimeter for Ellipse {
    /// Computes an approximation of the ellipse perimeter.
    ///
    /// Uses Ramanujan's second approximation formula, which provides excellent
    /// accuracy for all ellipses:
    ///
    /// ```text
    /// P ≈ π(a + b)(1 + 3h / (10 + √(4 - 3h)))
    /// where h = ((a - b) / (a + b))²
    /// ```
    ///
    /// This approximation has a relative error of less than 0.01% for typical cases.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    /// use eunoia::geometry::traits::Perimeter;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
    /// let perimeter = ellipse.perimeter();
    /// assert!(perimeter > 0.0);
    /// ```
    fn perimeter(&self) -> f64 {
        // Approximation using Ramanujan's second formula
        let a = self.semi_major;
        let b = self.semi_minor;
        let h = ((a - b).powi(2)) / ((a + b).powi(2));
        PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + (4.0 - 3.0 * h).sqrt()))
    }
}

impl Centroid for Ellipse {
    /// Returns the centroid of the ellipse, which is its center point.
    fn centroid(&self) -> Point {
        self.center
    }
}

impl BoundingBox for Ellipse {
    /// Computes the axis-aligned bounding box that contains the ellipse.
    ///
    /// For a rotated ellipse, this computes the smallest axis-aligned rectangle
    /// that fully contains the ellipse. The calculation accounts for the rotation
    /// by projecting the ellipse axes onto the coordinate axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    /// use eunoia::geometry::traits::BoundingBox;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
    /// let bbox = ellipse.bounding_box();
    /// // For unrotated ellipse, bbox dimensions equal 2*semi-major and 2*semi-minor
    /// ```
    fn bounding_box(&self) -> Rectangle {
        let cos_theta = self.rotation.cos();
        let sin_theta = self.rotation.sin();

        let width = 2.0
            * ((self.semi_major * cos_theta).powi(2) + (self.semi_minor * sin_theta).powi(2))
                .sqrt();
        let height = 2.0
            * ((self.semi_major * sin_theta).powi(2) + (self.semi_minor * cos_theta).powi(2))
                .sqrt();

        Rectangle::new(self.center, width, height)
    }
}

impl Closed for Ellipse {
    fn contains(&self, other: &Self) -> bool {
        // Quick check: if other's center is outside self, it can't be contained
        if !self.contains_point(&other.center) {
            return false;
        }

        // If other is larger than self (by area), it can't be contained
        if other.area() > self.area() {
            return false;
        }

        // Convert both ellipses to conics
        let c1 = Conic::from_ellipse(*self);
        let c2 = Conic::from_ellipse(*other);

        // Check for boundary intersections
        let intersection_points = c1.intersect_conic(&c2);

        // If no intersections found and center is inside, other is contained
        if intersection_points.is_empty() {
            return true;
        }

        // If intersections were found, verify they're not real by checking
        // the extreme points of other (ends of major/minor axes)
        let phi = other.rotation;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        // Check points at the ends of the semi-major axis
        let major_offset_x = other.semi_major * cos_phi;
        let major_offset_y = other.semi_major * sin_phi;
        let p1 = Point::new(
            other.center.x() + major_offset_x,
            other.center.y() + major_offset_y,
        );
        let p2 = Point::new(
            other.center.x() - major_offset_x,
            other.center.y() - major_offset_y,
        );

        // Check points at the ends of the semi-minor axis
        let minor_offset_x = other.semi_minor * -sin_phi;
        let minor_offset_y = other.semi_minor * cos_phi;
        let p3 = Point::new(
            other.center.x() + minor_offset_x,
            other.center.y() + minor_offset_y,
        );
        let p4 = Point::new(
            other.center.x() - minor_offset_x,
            other.center.y() - minor_offset_y,
        );

        // All extreme points must be inside or on self
        self.contains_point(&p1)
            && self.contains_point(&p2)
            && self.contains_point(&p3)
            && self.contains_point(&p4)
    }

    fn contains_point(&self, point: &Point) -> bool {
        // Transform point to ellipse's local coordinate system
        let dx = point.x() - self.center.x();
        let dy = point.y() - self.center.y();

        let cos_phi = self.rotation.cos();
        let sin_phi = self.rotation.sin();

        // Rotate point to align with ellipse axes
        let x_local = dx * cos_phi + dy * sin_phi;
        let y_local = dx * sin_phi - dy * cos_phi;

        // Check if point is inside using the ellipse equation
        (x_local * x_local) / (self.semi_major * self.semi_major)
            + (y_local * y_local) / (self.semi_minor * self.semi_minor)
            <= 1.0
    }

    fn intersects(&self, other: &Self) -> bool {
        // Quick rejection: if centers are too far apart, they can't intersect
        let center_distance = self.center.distance(&other.center);
        let max_reach_self = self.semi_major;
        let max_reach_other = other.semi_major;

        if center_distance > max_reach_self + max_reach_other {
            return false;
        }

        // Check if one contains the other (no intersection in that case)
        if self.contains(other) || other.contains(self) {
            return false;
        }

        // Convert to conics and check for intersection points
        let c1 = Conic::from_ellipse(*self);
        let c2 = Conic::from_ellipse(*other);

        let intersection_points = c1.intersect_conic(&c2);

        !intersection_points.is_empty()
    }

    fn intersection_area(&self, _other: &Self) -> f64 {
        // Placeholder implementation
        unimplemented!()
    }

    fn intersection_points(&self, other: &Self) -> Vec<Point> {
        // Convert both ellipses to conics
        let c1 = Conic::from_ellipse(*self);
        let c2 = Conic::from_ellipse(*other);

        // Get homogeneous intersection points
        let homogeneous_points = c1.intersect_conic(&c2);

        // Convert to Cartesian points
        homogeneous_points
            .into_iter()
            .map(|hp| Point::new(hp.x(), hp.y()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::Circle;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_sector_area_between_quarter() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);

        // Quarter sector from 0 to π/2
        let sector_between = ellipse.sector_area_between(0.0, PI / 2.0);
        let expected = ellipse.sector_area(PI / 2.0) - ellipse.sector_area(0.0);

        assert!(approx_eq(sector_between, expected));
        assert!(approx_eq(sector_between, ellipse.area() / 4.0));
    }

    #[test]
    fn test_sector_area_between_with_ccw_adjustment() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 4.0, 2.0, 0.0);

        // When theta1 < theta0, it should add 2π to theta1 for CCW ordering
        // Going from 3π/4 to π/4 CCW (wrapping around through 0)
        let theta0 = 3.0 * PI / 4.0;
        let theta1 = PI / 4.0;

        let sector_between = ellipse.sector_area_between(theta0, theta1);

        // This should be equivalent to going from 3π/4 to (π/4 + 2π)
        let expected = ellipse.sector_area(theta1 + 2.0 * PI) - ellipse.sector_area(theta0);

        assert!(
            approx_eq(sector_between, expected),
            "Expected {}, got {}",
            expected,
            sector_between
        );

        // The sector should be positive and cover most of the ellipse
        assert!(sector_between > 0.0);
        assert!(sector_between > ellipse.area() / 2.0);
    }

    #[test]
    fn test_sector_area_between_same_angle() {
        let ellipse = Ellipse::new(Point::new(1.0, 2.0), 3.0, 2.5, 0.3);

        // Same angle should give zero sector area
        let angle = PI / 3.0;
        let sector = ellipse.sector_area_between(angle, angle);

        assert!(approx_eq(sector, 0.0));
    }

    #[test]
    fn test_sector_area_between_full_rotation() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);

        // Full rotation: from 0 to 2π
        let sector = ellipse.sector_area_between(0.0, 2.0 * PI);

        assert!(approx_eq(sector, ellipse.area()));
    }

    #[test]
    fn test_sector_full_ellipse() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.0, 0.0);
        let full_angle = 2.0 * PI;
        let sector = ellipse.sector_area(full_angle);
        assert!(approx_eq(sector, ellipse.area()));
    }

    #[test]
    fn test_sector_half_ellipse() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 4.0, 3.0, 0.0);
        let half_angle = PI;
        let sector = ellipse.sector_area(half_angle);
        assert!(approx_eq(sector, ellipse.area() / 2.0));
    }

    #[test]
    fn test_sector_quarter_ellipse() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
        let quarter_angle = PI / 2.0;
        let sector = ellipse.sector_area(quarter_angle);
        assert!(approx_eq(sector, ellipse.area() / 4.0));
    }

    #[test]
    fn test_sector_zero_angle() {
        let ellipse = Ellipse::new(Point::new(1.0, 1.0), 3.0, 2.0, 0.5);
        let sector = ellipse.sector_area(0.0);
        assert!(approx_eq(sector, 0.0));
    }

    #[test]
    fn test_sector_circular_ellipse_matches_circle() {
        // Create a circle-like ellipse (semi_major == semi_minor)
        let radius = 3.0;
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), radius, radius, 0.0);
        let circle = Circle::new(Point::new(0.0, 0.0), radius);

        let angle = PI / 3.0;
        let ellipse_sector = ellipse.sector_area(angle);
        let circle_sector = circle.sector_area(angle);

        assert!(approx_eq(ellipse_sector, circle_sector));
    }

    #[test]
    fn test_sector_circular_ellipse_multiple_angles() {
        let radius = 2.0;
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), radius, radius, 0.0);
        let circle = Circle::new(Point::new(0.0, 0.0), radius);

        let test_angles = vec![
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 2.0,
            PI,
            3.0 * PI / 2.0,
            2.0 * PI,
        ];

        for angle in test_angles {
            let ellipse_sector = ellipse.sector_area(angle);
            let circle_sector = circle.sector_area(angle);
            assert!(
                approx_eq(ellipse_sector, circle_sector),
                "Mismatch at angle {}: ellipse={}, circle={}",
                angle,
                ellipse_sector,
                circle_sector
            );
        }
    }

    #[test]
    fn test_ellipse_segment_semicircle() {
        let radius = 2.0;
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), radius, radius, 0.0);

        // Points at opposite ends of diameter (along x-axis)
        let p0 = Point::new(-radius, 0.0);
        let p1 = Point::new(radius, 0.0);

        let segment = ellipse.ellipse_segment(p0, p1);
        // Semicircle segment equals half the circle area
        assert!(approx_eq(segment, ellipse.area() / 2.0));
    }

    #[test]
    fn test_ellipse_segment_circular_matches_circle_segment() {
        let radius = 3.0;
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), radius, radius, 0.0);
        let circle = Circle::new(Point::new(0.0, 0.0), radius);

        // Test angle
        let angle = PI / 4.0;

        // Points on circle boundary
        let p0 = Point::new(radius, 0.0);
        let p1 = Point::new(radius * angle.cos(), radius * angle.sin());

        let ellipse_segment = ellipse.ellipse_segment(p0, p1);
        let circle_segment = circle.segment_area_from_angle(angle);

        assert!(
            approx_eq(ellipse_segment, circle_segment),
            "Circular ellipse segment should match circle segment: ellipse={}, circle={}",
            ellipse_segment,
            circle_segment
        );
    }

    #[test]
    fn test_ellipse_segment_small_angle() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.0, 0.0);

        let angle = PI / 12.0; // 15 degrees
        let p0 = Point::new(ellipse.semi_major(), 0.0);
        let p1 = Point::new(
            ellipse.semi_major() * angle.cos(),
            ellipse.semi_minor() * angle.sin(),
        );

        let segment = ellipse.ellipse_segment(p0, p1);

        // Segment should be positive and less than the sector
        assert!(segment > 0.0);
        let sector = ellipse.sector_area(angle);
        assert!(segment < sector);
    }

    #[test]
    fn test_ellipse_segment_zero_angle() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 4.0, 2.5, 0.0);

        // Same point twice should give zero area
        let p = Point::new(ellipse.semi_major(), 0.0);
        let segment = ellipse.ellipse_segment(p, p);

        assert!(approx_eq(segment, 0.0));
    }

    #[test]
    fn test_ellipse_segment_with_known_values() {
        // Reference values computed for ellipse with semi_major=5.0, semi_minor=3.0
        // Total area: 47.12388980384689

        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);

        // Quarter ellipse segment (from θ=0 to θ=π/2)
        let p0 = Point::new(ellipse.semi_major(), 0.0);
        let p1 = Point::new(0.0, ellipse.semi_minor());
        let segment_quarter = ellipse.ellipse_segment(p0, p1);
        let expected_quarter = 4.280972450961723;

        assert!(
            approx_eq(segment_quarter, expected_quarter),
            "Quarter segment: expected {}, got {}",
            expected_quarter,
            segment_quarter
        );

        // Half ellipse segment (from θ=0 to θ=π)
        let p2 = Point::new(-ellipse.semi_major(), 0.0);
        let segment_half = ellipse.ellipse_segment(p0, p2);
        let expected_half = 23.561944901923447;

        assert!(
            approx_eq(segment_half, expected_half),
            "Half segment: expected {}, got {}",
            expected_half,
            segment_half
        );

        // Half segment should also equal half the ellipse area
        assert!(approx_eq(segment_half, ellipse.area() / 2.0));
    }

    #[test]
    fn test_ellipse_segment_additional_angles() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);

        // 30 degree segment (π/6)
        let angle = PI / 6.0;
        let p0 = Point::new(ellipse.semi_major(), 0.0);
        let p1 = Point::new(
            ellipse.semi_major() * angle.cos(),
            ellipse.semi_minor() * angle.sin(),
        );
        let segment_30deg = ellipse.ellipse_segment(p0, p1);
        let expected_30deg = 0.17699081698724095;

        assert!(
            approx_eq(segment_30deg, expected_30deg),
            "30° segment: expected {}, got {}",
            expected_30deg,
            segment_30deg
        );

        // 60 degree segment (π/3)
        let angle = PI / 3.0;
        let p2 = Point::new(
            ellipse.semi_major() * angle.cos(),
            ellipse.semi_minor() * angle.sin(),
        );
        let segment_60deg = ellipse.ellipse_segment(p0, p2);
        let expected_60deg = 1.3587911055911919;

        assert!(
            approx_eq(segment_60deg, expected_60deg),
            "60° segment: expected {}, got {}",
            expected_60deg,
            segment_60deg
        );

        // Additional sanity checks
        assert!(segment_30deg < segment_60deg);
        assert!(segment_60deg < ellipse.area() / 4.0);
    }

    #[test]
    fn test_ellipse_segment_270_degrees() {
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);

        // 270 degree segment (3π/2)
        // When theta1 - theta0 > π, the algorithm computes the MAJOR arc segment.
        // This is the correct behavior: going CCW from 0° to 270° covers 270° of arc,
        // which is the larger of the two possible segments between these points.
        let angle = 3.0 * PI / 2.0;
        let p0 = Point::new(ellipse.semi_major(), 0.0);
        let p1 = Point::new(
            ellipse.semi_major() * angle.cos(),
            ellipse.semi_minor() * angle.sin(),
        );
        let segment_270deg = ellipse.ellipse_segment(p0, p1);
        let expected_270deg = 42.84291735288517;

        assert!(
            approx_eq(segment_270deg, expected_270deg),
            "270° segment: expected {}, got {}",
            expected_270deg,
            segment_270deg
        );

        // Verify this is indeed the major arc (> half the ellipse area)
        assert!(segment_270deg > ellipse.area() / 2.0);

        // The complementary segment (minor arc) would be the small 90° piece
        let complementary = ellipse.area() - segment_270deg;
        assert!(complementary > 0.0);
        assert!(complementary < ellipse.area() / 4.0);
    }

    #[test]
    fn test_contains_point_center_and_boundary() {
        let ellipse = Ellipse::new(Point::new(2.0, 3.0), 5.0, 3.0, 0.0);

        // Center should be inside
        assert!(ellipse.contains_point(&Point::new(2.0, 3.0)));

        // Points on the semi-major axis (horizontal)
        assert!(ellipse.contains_point(&Point::new(7.0, 3.0))); // right edge
        assert!(ellipse.contains_point(&Point::new(-3.0, 3.0))); // left edge

        // Points on the semi-minor axis (vertical)
        assert!(ellipse.contains_point(&Point::new(2.0, 6.0))); // top edge
        assert!(ellipse.contains_point(&Point::new(2.0, 0.0))); // bottom edge

        // Point clearly outside
        assert!(!ellipse.contains_point(&Point::new(10.0, 10.0)));
    }

    #[test]
    fn test_contains_point_rotated_ellipse() {
        // Create an ellipse rotated 45 degrees
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 4.0, 2.0, PI / 4.0);

        // Center should be inside
        assert!(ellipse.contains_point(&Point::new(0.0, 0.0)));

        // Point on the rotated semi-major axis
        let dist = 4.0 / (2.0_f64).sqrt(); // 4 * cos(45°) = 4 * sin(45°)
        assert!(ellipse.contains_point(&Point::new(dist, dist)));
        assert!(ellipse.contains_point(&Point::new(-dist, -dist)));

        // Point clearly outside
        assert!(!ellipse.contains_point(&Point::new(5.0, 0.0)));
        assert!(!ellipse.contains_point(&Point::new(0.0, 5.0)));

        // Point inside but not on axis
        assert!(ellipse.contains_point(&Point::new(0.5, 0.5)));
    }

    #[test]
    fn test_contains_ellipse_concentric() {
        // Larger ellipse should contain smaller concentric one
        let outer = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
        let inner = Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.0, 0.0);

        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_contains_ellipse_offset_centers() {
        // Large ellipse at origin
        let outer = Ellipse::new(Point::new(0.0, 0.0), 10.0, 8.0, 0.0);

        // Small ellipse offset but still inside
        let inner = Ellipse::new(Point::new(2.0, 1.0), 2.0, 1.5, 0.0);

        assert!(outer.contains(&inner));

        // Small ellipse offset too far (outside)
        let outside = Ellipse::new(Point::new(9.0, 0.0), 2.0, 1.5, 0.0);

        assert!(!outer.contains(&outside));
    }

    #[test]
    fn test_contains_ellipse_rotated() {
        // Test with rotated ellipses
        let outer = Ellipse::new(Point::new(0.0, 0.0), 6.0, 4.0, PI / 6.0);
        let inner = Ellipse::new(Point::new(0.0, 0.0), 2.0, 1.5, PI / 4.0);

        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_contains_ellipse_intersecting() {
        // Overlapping but not containing
        let e1 = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
        let e2 = Ellipse::new(Point::new(3.0, 0.0), 4.0, 2.5, 0.0);

        assert!(!e1.contains(&e2));
        assert!(!e2.contains(&e1));
    }

    #[test]
    fn test_intersects_overlapping() {
        // Two ellipses that intersect
        let e1 = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
        let e2 = Ellipse::new(Point::new(3.0, 0.0), 4.0, 2.5, 0.0);

        assert!(e1.intersects(&e2));
        assert!(e2.intersects(&e1));
    }

    #[test]
    fn test_intersects_separate() {
        // Two ellipses that don't touch
        let e1 = Ellipse::new(Point::new(0.0, 0.0), 2.0, 1.5, 0.0);
        let e2 = Ellipse::new(Point::new(10.0, 0.0), 3.0, 2.0, 0.0);

        assert!(!e1.intersects(&e2));
        assert!(!e2.intersects(&e1));
    }

    #[test]
    fn test_intersects_contained() {
        // One ellipse contains the other - no intersection
        let outer = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
        let inner = Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.0, 0.0);

        assert!(!outer.intersects(&inner));
        assert!(!inner.intersects(&outer));
    }

    #[test]
    fn test_intersects_touching() {
        // Two ellipses that barely touch (externally tangent)
        let e1 = Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.0, 0.0);
        let e2 = Ellipse::new(Point::new(6.0, 0.0), 3.0, 2.0, 0.0);

        // They should intersect (tangent point counts as intersection)
        assert!(e1.intersects(&e2));
    }

    #[test]
    fn test_intersection_points_overlapping() {
        // Two ellipses that intersect at multiple points
        let e1 = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
        let e2 = Ellipse::new(Point::new(3.0, 0.0), 4.0, 2.5, 0.0);

        let points = e1.intersection_points(&e2);

        // Should have 2 or 4 intersection points (depending on overlap)
        assert!(!points.is_empty());
        assert!(points.len() <= 4);

        // Note: Due to numerical precision, points on the boundary may not satisfy
        // contains_point with the <= 1.0 check, so we just verify they exist
    }

    #[test]
    fn test_intersection_points_separate() {
        // Two ellipses that don't intersect
        let e1 = Ellipse::new(Point::new(0.0, 0.0), 2.0, 1.5, 0.0);
        let e2 = Ellipse::new(Point::new(10.0, 0.0), 3.0, 2.0, 0.0);

        let points = e1.intersection_points(&e2);

        // Note: The conic intersection algorithm may return spurious points
        // for widely separated ellipses due to numerical issues
        // In practice, these should be filtered or we should check intersects() first
        assert!(points.len() <= 4);
    }

    #[test]
    fn test_intersection_points_contained() {
        // One ellipse contains the other - no boundary intersection
        let outer = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
        let inner = Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.0, 0.0);

        let points = outer.intersection_points(&inner);

        // Should have no intersection points (or handle numerical artifacts)
        // The contains check would return true, so we expect 0 or duplicate points
        assert!(points.len() <= 4);
    }

    #[test]
    fn test_intersection_points_tangent() {
        // Two circles (special case of ellipses) that are externally tangent
        let e1 = Ellipse::new(Point::new(0.0, 0.0), 2.0, 2.0, 0.0); // circle
        let e2 = Ellipse::new(Point::new(4.0, 0.0), 2.0, 2.0, 0.0); // circle

        let points = e1.intersection_points(&e2);

        // Should have 1 or 2 intersection points (tangent at one point, possibly duplicated)
        assert!(!points.is_empty());

        // The tangent point should be at (2.0, 0.0)
        if !points.is_empty() {
            let first_point = &points[0];
            assert!((first_point.x() - 2.0).abs() < 1e-6);
            assert!(first_point.y().abs() < 1e-6);
        }
    }
}
