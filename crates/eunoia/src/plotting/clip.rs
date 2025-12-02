//! Polygon clipping operations.
//!
//! This module provides a clean wrapper around the i_overlay library for
//! performing boolean operations on polygons.

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Polygon;
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;

/// Polygon clipping operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipOperation {
    /// Intersection: returns the common area of both polygons.
    Intersection,
    /// Union: returns the combined area of both polygons.
    Union,
    /// Difference: returns area in first polygon but not in second.
    Difference,
    /// Symmetric difference: returns areas in either polygon but not in both.
    Xor,
}

impl ClipOperation {
    fn to_overlay_rule(&self) -> OverlayRule {
        match self {
            ClipOperation::Intersection => OverlayRule::Intersect,
            ClipOperation::Union => OverlayRule::Union,
            ClipOperation::Difference => OverlayRule::Difference,
            ClipOperation::Xor => OverlayRule::Xor,
        }
    }
}

/// Performs a clipping operation between two polygons.
///
/// Returns a list of polygons representing the result. Multiple polygons
/// can result when the operation creates disconnected regions.
///
/// # Arguments
///
/// * `subject` - The first (subject) polygon
/// * `clip` - The second (clip) polygon
/// * `operation` - The clipping operation to perform
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Polygon;
/// use eunoia::geometry::primitives::Point;
/// use eunoia::plotting::{polygon_clip, ClipOperation};
///
/// let square1 = Polygon::new(vec![
///     Point::new(0.0, 0.0),
///     Point::new(2.0, 0.0),
///     Point::new(2.0, 2.0),
///     Point::new(0.0, 2.0),
/// ]);
///
/// let square2 = Polygon::new(vec![
///     Point::new(1.0, 1.0),
///     Point::new(3.0, 1.0),
///     Point::new(3.0, 3.0),
///     Point::new(1.0, 3.0),
/// ]);
///
/// let intersection = polygon_clip(&square1, &square2, ClipOperation::Intersection);
/// assert_eq!(intersection.len(), 1);
/// ```
pub fn polygon_clip(subject: &Polygon, clip: &Polygon, operation: ClipOperation) -> Vec<Polygon> {
    // Handle edge cases
    if subject.vertices().is_empty() {
        match operation {
            ClipOperation::Union => {
                if clip.vertices().is_empty() {
                    return vec![];
                }
                return vec![clip.clone()];
            }
            ClipOperation::Intersection | ClipOperation::Difference => return vec![],
            ClipOperation::Xor => {
                if clip.vertices().is_empty() {
                    return vec![];
                }
                return vec![clip.clone()];
            }
        }
    }

    if clip.vertices().is_empty() {
        match operation {
            ClipOperation::Union | ClipOperation::Difference | ClipOperation::Xor => {
                return vec![subject.clone()]
            }
            ClipOperation::Intersection => return vec![],
        }
    }

    // Convert to i_overlay format
    let subject_points: Vec<Point> = subject.vertices().to_vec();
    let clip_points: Vec<Point> = clip.vertices().to_vec();

    // Perform overlay operation
    let result =
        subject_points.overlay(&clip_points, operation.to_overlay_rule(), FillRule::NonZero);

    // Convert back to our Polygon type
    result
        .into_iter()
        .flat_map(|shape| shape.into_iter())
        .map(|path| Polygon::new(path))
        .collect()
}

/// Clips a list of polygons against a single clip polygon.
///
/// This is a convenience function for clipping multiple subject polygons
/// against a single clip polygon.
pub fn polygon_clip_many(
    subjects: &[Polygon],
    clip: &Polygon,
    operation: ClipOperation,
) -> Vec<Polygon> {
    subjects
        .iter()
        .flat_map(|subject| polygon_clip(subject, clip, operation))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection_overlapping_squares() {
        let square1 = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ]);

        let square2 = Polygon::new(vec![
            Point::new(1.0, 1.0),
            Point::new(3.0, 1.0),
            Point::new(3.0, 3.0),
            Point::new(1.0, 3.0),
        ]);

        let result = polygon_clip(&square1, &square2, ClipOperation::Intersection);

        assert_eq!(result.len(), 1);
        let intersection_area = result[0].area();
        assert!((intersection_area - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_union_overlapping_squares() {
        let square1 = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ]);

        let square2 = Polygon::new(vec![
            Point::new(1.0, 1.0),
            Point::new(3.0, 1.0),
            Point::new(3.0, 3.0),
            Point::new(1.0, 3.0),
        ]);

        let result = polygon_clip(&square1, &square2, ClipOperation::Union);

        assert_eq!(result.len(), 1);
        let union_area = result[0].area();
        // Two 2x2 squares with 1x1 overlap = 4 + 4 - 1 = 7
        assert!((union_area - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_difference_overlapping_squares() {
        let square1 = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ]);

        let square2 = Polygon::new(vec![
            Point::new(1.0, 1.0),
            Point::new(3.0, 1.0),
            Point::new(3.0, 3.0),
            Point::new(1.0, 3.0),
        ]);

        let result = polygon_clip(&square1, &square2, ClipOperation::Difference);

        assert!(result.len() >= 1);
        let diff_area: f64 = result.iter().map(|p| p.area()).sum();
        // 2x2 square minus 1x1 overlap = 4 - 1 = 3
        assert!((diff_area - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_subject() {
        let empty = Polygon::new(vec![]);
        let square = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]);

        assert_eq!(
            polygon_clip(&empty, &square, ClipOperation::Intersection).len(),
            0
        );
        assert_eq!(polygon_clip(&empty, &square, ClipOperation::Union).len(), 1);
        assert_eq!(
            polygon_clip(&empty, &square, ClipOperation::Difference).len(),
            0
        );
    }

    #[test]
    fn test_empty_clip() {
        let square = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]);
        let empty = Polygon::new(vec![]);

        assert_eq!(
            polygon_clip(&square, &empty, ClipOperation::Intersection).len(),
            0
        );
        assert_eq!(polygon_clip(&square, &empty, ClipOperation::Union).len(), 1);
        assert_eq!(
            polygon_clip(&square, &empty, ClipOperation::Difference).len(),
            1
        );
    }
}
