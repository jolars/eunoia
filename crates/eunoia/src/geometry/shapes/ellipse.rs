//! Ellipse shape implementation.

use std::f64::consts::PI;

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::geometry::traits::{Area, BoundingBox, Centroid, Closed, Distance, Perimeter};

/// An ellipse defined by a center point, semi-major and semi-minor axes, and rotation.
///
/// Ellipses provide more flexibility than circles for Euler and Venn diagrams,
/// allowing better fits for asymmetric set relationships. The ellipse is defined
/// by its center, semi-major axis (longest radius), semi-minor axis (shortest radius),
/// and a rotation angle in radians.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Ellipse;
/// use eunoia::geometry::traits::Area;
/// use eunoia::geometry::primitives::Point;
///
/// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
/// let area = ellipse.area();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    center: Point,
    semi_major: f64,
    semi_minor: f64,
    rotation: f64, // in radians
}

impl Ellipse {
    /// Creates a new ellipse with the specified center, semi-axes, and rotation.
    ///
    /// # Arguments
    ///
    /// * `center` - The center point of the ellipse
    /// * `semi_major` - The semi-major axis length (longest radius)
    /// * `semi_minor` - The semi-minor axis length (shortest radius)
    /// * `rotation` - The rotation angle in radians (counter-clockwise from positive x-axis)
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let ellipse = Ellipse::new(Point::new(1.0, 2.0), 5.0, 3.0, 0.0);
    /// ```
    pub fn new(center: Point, semi_major: f64, semi_minor: f64, rotation: f64) -> Self {
        Self {
            center,
            semi_major,
            semi_minor,
            rotation,
        }
    }

    /// Returns the center point of the ellipse.
    pub fn center(&self) -> Point {
        self.center
    }

    /// Returns the semi-major axis length (longest radius).
    pub fn semi_major(&self) -> f64 {
        self.semi_major
    }

    /// Returns the semi-minor axis length (shortest radius).
    pub fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    /// Returns the rotation angle in radians.
    pub fn rotation(&self) -> f64 {
        self.rotation
    }

    /// Computes the area of an elliptical sector from angle 0 to the given angle.
    ///
    /// Uses the exact formula for elliptical sector area. This is a primitive
    /// operation used by other segment and area computations.
    ///
    /// # Arguments
    ///
    /// * `theta` - The angle in radians (measured from the positive x-axis in the
    ///   ellipse's local coordinate system)
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    /// use std::f64::consts::PI;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
    /// let quarter_sector = ellipse.sector_area(PI / 2.0);
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
    /// The sector is computed in counter-clockwise (CCW) order from `theta0` to `theta1`.
    /// If `theta1 < theta0`, the function automatically adjusts by adding 2π to `theta1`
    /// to maintain CCW ordering.
    ///
    /// # Arguments
    ///
    /// * `theta0` - The starting angle in radians
    /// * `theta1` - The ending angle in radians
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    /// use std::f64::consts::PI;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
    /// let half_sector = ellipse.sector_area_between(0.0, PI);
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
    /// An ellipse segment is the region bounded by an arc and the chord connecting
    /// its endpoints. This method transforms the points into the ellipse's local
    /// coordinate system and computes the segment area using sector and triangle
    /// corrections.
    ///
    /// The segment is computed in CCW order from `p0` to `p1`. For arcs spanning
    /// more than π radians (major arc), the computation accounts for the larger
    /// segment.
    ///
    /// # Arguments
    ///
    /// * `p0` - The first boundary point on the ellipse
    /// * `p1` - The second boundary point on the ellipse
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Ellipse;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let ellipse = Ellipse::new(Point::new(0.0, 0.0), 5.0, 3.0, 0.0);
    /// let p0 = Point::new(5.0, 0.0);  // Point on ellipse at angle 0
    /// let p1 = Point::new(0.0, 3.0);  // Point on ellipse at angle π/2
    /// let segment_area = ellipse.ellipse_segment(p0, p1);
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
    fn area(&self) -> f64 {
        PI * self.semi_major * self.semi_minor
    }
}

impl Perimeter for Ellipse {
    fn perimeter(&self) -> f64 {
        // Approximation using Ramanujan's second formula
        let a = self.semi_major;
        let b = self.semi_minor;
        let h = ((a - b).powi(2)) / ((a + b).powi(2));
        PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + (4.0 - 3.0 * h).sqrt()))
    }
}

impl Centroid for Ellipse {
    fn centroid(&self) -> Point {
        self.center
    }
}

impl BoundingBox for Ellipse {
    fn bounding_box(&self) -> Rectangle {
        let cos_theta = self.rotation.cos();
        let sin_theta = self.rotation.sin();

        let width = 2.0
            * ((self.semi_major * cos_theta).powi(2) + (self.semi_minor * sin_theta).powi(2))
                .sqrt();
        let height = 2.0
            * ((self.semi_major * sin_theta).powi(2) + (self.semi_minor * cos_theta).powi(2))
                .sqrt();

        Rectangle::new(
            self.center,
            width,
            height,
        )
    }
}

impl Distance for Ellipse {
    fn distance(&self, _other: &Self) -> f64 {
        // Placeholder implementation
        unimplemented!()
    }
}

impl Closed for Ellipse {
    fn contains(&self, _other: &Self) -> bool {
        // Placeholder implementation
        unimplemented!()
    }

    fn contains_point(&self, _point: &Point) -> bool {
        // Placeholder implementation
        unimplemented!()
    }

    fn intersects(&self, _other: &Self) -> bool {
        // Placeholder implementation
        unimplemented!()
    }

    fn intersection_area(&self, _other: &Self) -> f64 {
        // Placeholder implementation
        unimplemented!()
    }

    fn intersection_points(&self, _other: &Self) -> Vec<Point> {
        // Placeholder implementation
        unimplemented!()
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
}
