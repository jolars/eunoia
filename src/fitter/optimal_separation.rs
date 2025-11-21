//! Optimal separation solver for computing distances between shapes.
//!
//! This module uses the existing `IntersectionArea` trait from the geometry module.

use crate::geometry::coord::Coord;
use crate::geometry::operations::{Area, IntersectionArea};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::brent::BrentOpt;

/// Cost function for finding optimal separation distance.
///
/// Minimizes |overlap(d) - target_overlap|² by adjusting the distance between
/// shape centers while keeping their sizes fixed.
struct SeparationCost<'a, S>
where
    S: IntersectionArea + Clone,
{
    shape1: S,
    shape2: S,
    axis: Coord, // Unit vector along which to separate
    target_overlap: f64,
    original_center1: Coord,
    original_center2: Coord,
    set_center: &'a dyn Fn(&mut S, Coord),
}

impl<'a, S> CostFunction for SeparationCost<'a, S>
where
    S: IntersectionArea + Clone,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, distance: &Self::Param) -> Result<Self::Output, Error> {
        // Position shapes along the axis at the given distance
        let mut s1 = self.shape1.clone();
        let mut s2 = self.shape2.clone();

        (self.set_center)(&mut s1, self.original_center1);
        (self.set_center)(
            &mut s2,
            Coord::new(
                self.original_center2.x() + self.axis.x() * distance,
                self.original_center2.y() + self.axis.y() * distance,
            ),
        );

        let overlap = s1.intersection_area(&s2);
        let diff = overlap - self.target_overlap;
        Ok(diff * diff)
    }
}

/// Solve for the optimal distance between two shapes to achieve target overlap.
///
/// This function uses Brent's method to find the center-to-center distance that
/// produces the desired overlap area. It works with any shape that implements
/// `IntersectionArea` and `Area`.
///
/// # Arguments
/// * `shape1` - First shape (used to determine size bounds)
/// * `shape2` - Second shape (used to determine size bounds)
/// * `target_overlap` - Desired overlap area
/// * `get_center` - Function to extract center from a shape
/// * `set_center` - Function to set center of a shape
/// * `get_size` - Function to get size parameter of a shape
/// * `tolerance` - Convergence tolerance (default: 1e-6)
///
/// # Returns
/// The distance between centers that achieves the target overlap
///
/// # Type Parameters
/// * `S` - Shape type that implements `IntersectionArea`, `Area`, and `Clone`
pub fn solve_optimal_separation<S>(
    shape1: &S,
    shape2: &S,
    target_overlap: f64,
    get_center: &dyn Fn(&S) -> Coord,
    set_center: &dyn Fn(&mut S, Coord),
    get_size: &dyn Fn(&S) -> f64,
    tolerance: Option<f64>,
) -> Result<f64, Error>
where
    S: IntersectionArea + Area + Clone,
{
    let tol = tolerance.unwrap_or(1e-6);

    // Determine search bounds based on shape sizes
    let size1 = get_size(shape1);
    let size2 = get_size(shape2);
    let min_distance = (size1 - size2).abs(); // One inside other
    let max_distance = size1 + size2; // Touching externally

    // Handle edge cases
    if target_overlap < 1e-10 {
        // No overlap desired - shapes should touch or be separated
        return Ok(max_distance);
    }

    let center1 = get_center(shape1);
    let center2 = get_center(shape2);

    // Compute axis of separation (unit vector from center1 to center2)
    let dx = center2.x() - center1.x();
    let dy = center2.y() - center1.y();
    let axis_length = (dx * dx + dy * dy).sqrt();
    let axis = if axis_length > 1e-10 {
        Coord::new(dx / axis_length, dy / axis_length)
    } else {
        Coord::new(1.0, 0.0) // Default to x-axis if centers coincide
    };

    // Set up Brent's method optimizer
    let cost = SeparationCost {
        shape1: shape1.clone(),
        shape2: shape2.clone(),
        axis,
        target_overlap,
        original_center1: center1,
        original_center2: center1, // Start from same point
        set_center,
    };

    let solver = BrentOpt::new(min_distance, max_distance);

    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(100).target_cost(tol))
        .run()?;

    Ok(*result.state.get_best_param().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::operations::Distance;

    // Simple test shape: a circle
    #[derive(Clone)]
    struct TestCircle {
        center: Coord,
        radius: f64,
    }

    impl Area for TestCircle {
        fn area(&self) -> f64 {
            std::f64::consts::PI * self.radius * self.radius
        }
    }

    impl IntersectionArea for TestCircle {
        fn intersection_area(&self, other: &Self) -> f64 {
            let d = Distance::distance(&self.center, &other.center);

            // Simplified: if distance >= r1 + r2, no overlap
            if d >= self.radius + other.radius {
                return 0.0;
            }
            // If distance <= |r1 - r2|, full overlap of smaller circle
            if d <= (self.radius - other.radius).abs() {
                return std::f64::consts::PI * self.radius.min(other.radius).powi(2);
            }
            // Otherwise, partial overlap (simplified approximation)
            let overlap_fraction = 1.0
                - (d - (self.radius - other.radius).abs())
                    / (self.radius + other.radius - (self.radius - other.radius).abs());
            overlap_fraction * std::f64::consts::PI * self.radius.min(other.radius).powi(2)
        }
    }

    #[test]
    fn test_separation_no_overlap() {
        let c1 = TestCircle {
            center: Coord::new(0.0, 0.0),
            radius: 1.0,
        };
        let c2 = TestCircle {
            center: Coord::new(0.0, 0.0),
            radius: 1.0,
        };

        let get_center = |s: &TestCircle| s.center;
        let set_center = |s: &mut TestCircle, c: Coord| s.center = c;
        let get_size = |s: &TestCircle| s.radius;

        let distance =
            solve_optimal_separation(&c1, &c2, 0.0, &get_center, &set_center, &get_size, None)
                .unwrap();

        // Should be approximately r1 + r2
        assert!((distance - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_separation_full_overlap() {
        let c1 = TestCircle {
            center: Coord::new(0.0, 0.0),
            radius: 2.0,
        };
        let c2 = TestCircle {
            center: Coord::new(0.0, 0.0),
            radius: 1.0,
        };

        let get_center = |s: &TestCircle| s.center;
        let set_center = |s: &mut TestCircle, c: Coord| s.center = c;
        let get_size = |s: &TestCircle| s.radius;

        let full_overlap = std::f64::consts::PI * 1.0_f64.powi(2);
        let distance = solve_optimal_separation(
            &c1,
            &c2,
            full_overlap,
            &get_center,
            &set_center,
            &get_size,
            None,
        )
        .unwrap();

        // Should be approximately |r1 - r2|
        assert!((distance - 1.0).abs() < 0.1);
    }
}
