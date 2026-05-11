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
    fn to_overlay_rule(self) -> OverlayRule {
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
        .map(Polygon::new)
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

/// Subtracts the union of `clips` from `subject` in a single overlay pass.
///
/// Returns a flat list of rings — outer rings and holes are intermixed,
/// which matches the output shape of [`polygon_clip`]. Use
/// [`crate::plotting::classify_into_pieces`] to recover the
/// outer-with-holes piece structure.
///
/// # Why a dedicated multi-shape helper
///
/// [`polygon_clip`] wraps i_overlay's single-resource overlay path: each
/// call takes one subject and one clip. Iterating
/// `polygon_clip(running, &clip, Difference)` over a slice of clips
/// works as long as the running result is a single-ring polygon, but as
/// soon as the running result becomes multi-ring (one outer + one hole,
/// say) the next pairwise call has no way to know which output rings
/// belong to which input piece — outer/hole pairing is lost.
/// `polygon_difference` avoids that by handing the full clip set to
/// i_overlay as one composite resource so a non-zero-fill difference is
/// computed end-to-end in one pass.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::primitives::Point;
/// use eunoia::geometry::shapes::Polygon;
/// use eunoia::plotting::polygon_difference;
///
/// // Outer 6×6 square, with two overlapping 2×2 squares to subtract.
/// let outer = Polygon::new(vec![
///     Point::new(0.0, 0.0),
///     Point::new(6.0, 0.0),
///     Point::new(6.0, 6.0),
///     Point::new(0.0, 6.0),
/// ]);
/// let a = Polygon::new(vec![
///     Point::new(2.0, 2.0),
///     Point::new(4.0, 2.0),
///     Point::new(4.0, 4.0),
///     Point::new(2.0, 4.0),
/// ]);
/// let b = Polygon::new(vec![
///     Point::new(3.0, 3.0),
///     Point::new(5.0, 3.0),
///     Point::new(5.0, 5.0),
///     Point::new(3.0, 5.0),
/// ]);
///
/// let rings = polygon_difference(&outer, &[a, b]);
/// // Net region area: outer (36) − union of a∪b (7) = 29. Signed shoelace
/// // sum gives the net independent of i_overlay's per-ring orientation;
/// // its absolute value equals the region's net area.
/// let signed_sum: f64 = rings.iter().map(|r| {
///     let v = r.vertices();
///     let n = v.len();
///     let mut s = 0.0;
///     for i in 0..n {
///         let j = (i + 1) % n;
///         s += v[i].x() * v[j].y() - v[j].x() * v[i].y();
///     }
///     0.5 * s
/// }).sum();
/// assert!((signed_sum.abs() - 29.0).abs() < 1e-6);
/// ```
/// Union of a flat list of polygons in a single multi-resource overlay
/// pass.
///
/// Returns a flat list of rings — outer rings and holes are intermixed,
/// matching the output shape of [`polygon_clip`]. Use
/// [`crate::plotting::classify_into_pieces`] to recover the outer-with-holes
/// piece structure.
///
/// # Why a dedicated helper
///
/// Iterating `polygon_clip_many(running, &next, Union)` works for two
/// inputs but loses pairing as soon as `running` becomes multi-polygon and
/// `next` would bridge two existing pieces — the iterative result is
/// `[A ∪ next, B ∪ next]`, two overlapping polygons instead of the single
/// connected piece. Handing all rings to i_overlay as a single Contours
/// resource lets it compute the union end-to-end without that loss.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::primitives::Point;
/// use eunoia::geometry::shapes::Polygon;
/// use eunoia::plotting::polygon_union_many;
///
/// // Three rectangles arranged in an L. Adjacent and overlapping inputs
/// // are merged into a single connected piece.
/// let a = Polygon::new(vec![
///     Point::new(0.0, 0.0),
///     Point::new(2.0, 0.0),
///     Point::new(2.0, 1.0),
///     Point::new(0.0, 1.0),
/// ]);
/// let b = Polygon::new(vec![
///     Point::new(0.0, 1.0),
///     Point::new(1.0, 1.0),
///     Point::new(1.0, 2.0),
///     Point::new(0.0, 2.0),
/// ]);
/// let c = Polygon::new(vec![
///     Point::new(1.5, 0.5),
///     Point::new(3.0, 0.5),
///     Point::new(3.0, 1.5),
///     Point::new(1.5, 1.5),
/// ]);
///
/// let rings = polygon_union_many(&[a, b, c]);
/// // Net area: a (2) + b (1) + c (1.5) − overlap(a ∩ c) (0.25) = 4.25.
/// // (a ∩ b shares only an edge — zero area; b and c are disjoint.)
/// let signed_sum: f64 = rings.iter().map(|r| {
///     let v = r.vertices();
///     let n = v.len();
///     let mut s = 0.0;
///     for i in 0..n {
///         let j = (i + 1) % n;
///         s += v[i].x() * v[j].y() - v[j].x() * v[i].y();
///     }
///     0.5 * s
/// }).sum();
/// assert!((signed_sum.abs() - 4.25).abs() < 1e-6);
/// ```
pub fn polygon_union_many(polygons: &[Polygon]) -> Vec<Polygon> {
    let rings: Vec<Vec<Point>> = polygons
        .iter()
        .filter(|p| !p.vertices().is_empty())
        .map(|p| p.vertices().to_vec())
        .collect();
    if rings.is_empty() {
        return vec![];
    }
    let empty: Vec<Vec<Point>> = Vec::new();
    let result = rings.overlay(&empty, OverlayRule::Union, FillRule::NonZero);
    result
        .into_iter()
        .flat_map(|shape| shape.into_iter())
        .map(Polygon::new)
        .collect()
}

pub fn polygon_difference(subject: &Polygon, clips: &[Polygon]) -> Vec<Polygon> {
    if subject.vertices().is_empty() {
        return vec![];
    }

    // Drop empty clip polygons; bail out early if none of them carry geometry,
    // since `subject ∖ ∅ = subject`.
    let resource: Vec<Vec<Point>> = clips
        .iter()
        .filter(|p| !p.vertices().is_empty())
        .map(|p| p.vertices().to_vec())
        .collect();
    if resource.is_empty() {
        return vec![subject.clone()];
    }

    let subject_points: Vec<Point> = subject.vertices().to_vec();
    let result = subject_points.overlay(&resource, OverlayRule::Difference, FillRule::NonZero);

    result
        .into_iter()
        .flat_map(|shape| shape.into_iter())
        .map(Polygon::new)
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

        assert!(!result.is_empty());
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

    #[test]
    fn test_polygon_difference_no_clips_returns_subject() {
        let square = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]);
        let result = polygon_difference(&square, &[]);
        assert_eq!(result.len(), 1);
        assert!((result[0].area() - 1.0).abs() < 1e-9);
    }

    /// Sum the signed shoelace areas of every ring in `rings`. This is the
    /// net area of the corresponding outer/hole region — independent of
    /// i_overlay's (unstable) per-ring orientation, since outer rings and
    /// hole rings always appear with opposite signs. The absolute value
    /// equals the net region area.
    fn signed_area_sum(rings: &[Polygon]) -> f64 {
        rings
            .iter()
            .map(|r| {
                let v = r.vertices();
                let n = v.len();
                let mut s = 0.0;
                for i in 0..n {
                    let j = (i + 1) % n;
                    s += v[i].x() * v[j].y() - v[j].x() * v[i].y();
                }
                0.5 * s
            })
            .sum()
    }

    #[test]
    fn test_polygon_difference_overlapping_clips_preserve_pairing() {
        // Subject is a 6×6 square. Two clips overlap each other (a∪b = 7 area
        // because they share a 1×1 corner). Pairwise difference would lose
        // outer/hole pairing as soon as the running result became multi-ring;
        // `polygon_difference` should give an exact net area.
        let outer = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(6.0, 0.0),
            Point::new(6.0, 6.0),
            Point::new(0.0, 6.0),
        ]);
        let a = Polygon::new(vec![
            Point::new(2.0, 2.0),
            Point::new(4.0, 2.0),
            Point::new(4.0, 4.0),
            Point::new(2.0, 4.0),
        ]);
        let b = Polygon::new(vec![
            Point::new(3.0, 3.0),
            Point::new(5.0, 3.0),
            Point::new(5.0, 5.0),
            Point::new(3.0, 5.0),
        ]);
        let result = polygon_difference(&outer, &[a, b]);
        // Net = 36 - (4 + 4 - 1) = 29. Absolute value because output ring
        // orientation is not guaranteed to be CCW-outer.
        assert!((signed_area_sum(&result).abs() - 29.0).abs() < 1e-6);
    }

    #[test]
    fn test_polygon_difference_with_nested_clip_yields_outer_and_hole() {
        // A clip strictly inside the subject must produce an outer + a hole,
        // i.e. two rings. This is the canonical multi-ring case that
        // pairwise difference can't compose.
        let outer = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let inner = Polygon::new(vec![
            Point::new(3.0, 3.0),
            Point::new(7.0, 3.0),
            Point::new(7.0, 7.0),
            Point::new(3.0, 7.0),
        ]);
        let result = polygon_difference(&outer, &[inner]);
        assert_eq!(result.len(), 2, "expected one outer + one hole ring");
        // Net = outer (100) - hole (16) = 84.
        assert!((signed_area_sum(&result).abs() - 84.0).abs() < 1e-6);
    }

    #[test]
    fn test_polygon_difference_skips_empty_clips() {
        let outer = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ]);
        let empty = Polygon::new(vec![]);
        let real = Polygon::new(vec![
            Point::new(0.5, 0.5),
            Point::new(1.5, 0.5),
            Point::new(1.5, 1.5),
            Point::new(0.5, 1.5),
        ]);
        // `[empty]` alone should be a no-op (treated as no clips).
        let r = polygon_difference(&outer, &[empty.clone()]);
        assert_eq!(r.len(), 1);
        assert!((r[0].area() - 4.0).abs() < 1e-9);

        // Empty interleaved with real clip subtracts only the real clip.
        let r = polygon_difference(&outer, &[empty, real]);
        // Net = subject (4) - real_clip (1) = 3.
        assert!((signed_area_sum(&r).abs() - 3.0).abs() < 1e-6);
    }
}
