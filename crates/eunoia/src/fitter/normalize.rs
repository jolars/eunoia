//! Diagram layout normalization.
//!
//! This module provides functionality to normalize diagram layouts by:
//! 1. Rotating clusters to a canonical orientation
//! 2. Centering the overall layout
//! 3. Packing disjoint clusters compactly

use crate::fitter::clustering::find_clusters;
use crate::fitter::packing::skyline_pack;
use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::geometry::traits::DiagramShape;
use std::f64::consts::PI;

/// Normalize a collection of diagram shapes.
///
/// This function:
/// 1. Identifies disjoint clusters of shapes
/// 2. Rotates each cluster to a canonical orientation
/// 3. Centers each cluster
/// 4. Packs multiple clusters together compactly
/// 5. Centers the final layout
///
/// # Arguments
///
/// * `shapes` - Mutable slice of shapes to normalize
/// * `padding_factor` - Padding between clusters as a fraction of total width (default: 0.015)
pub fn normalize_layout<S>(shapes: &mut [S], padding_factor: f64)
where
    S: DiagramShape + Clone,
{
    if shapes.is_empty() {
        return;
    }

    // Step 1: Find disjoint clusters
    let clusters = find_clusters(shapes);

    if clusters.len() == 1 {
        // Single cluster - just rotate and center
        let cluster = &clusters[0];
        if cluster.len() > 1 {
            rotate_cluster(shapes, cluster);
        }
        center_layout(shapes);
    } else {
        // Multiple clusters - rotate, pack, and center
        for cluster in &clusters {
            if cluster.len() > 1 {
                rotate_cluster(shapes, cluster);
            }
        }

        pack_clusters(shapes, &clusters, padding_factor);
        center_layout(shapes);
    }
}

/// Rotate a cluster to a canonical orientation.
///
/// For clusters with 2+ shapes:
/// 1. Rotate so the line between first two shapes is horizontal
/// 2. Mirror if needed so first shape is bottom-left
fn rotate_cluster<S>(shapes: &mut [S], cluster: &[usize])
where
    S: DiagramShape + Clone,
{
    if cluster.len() < 2 {
        return;
    }

    let idx0 = cluster[0];
    let idx1 = cluster[1];

    let c0 = shapes[idx0].centroid();
    let c1 = shapes[idx1].centroid();

    // Compute rotation angle to make the line horizontal
    let dx = c1.x() - c0.x();
    let dy = c1.y() - c0.y();
    let theta = -dy.atan2(dx); // Negative because we rotate the other way

    // Rotate all shapes in cluster around first shape's centroid
    if theta.abs() > 1e-10 {
        let pivot = c0;
        for &idx in cluster {
            shapes[idx] = rotate_shape(&shapes[idx], theta, &pivot);
        }
    }

    // Recompute centroids after rotation
    let c0 = shapes[idx0].centroid();
    let _c1 = shapes[idx1].centroid();

    // Compute cluster bounding box for mirroring decisions
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for &idx in cluster {
        let c = shapes[idx].centroid();
        x_min = x_min.min(c.x());
        x_max = x_max.max(c.x());
        y_min = y_min.min(c.y());
        y_max = y_max.max(c.y());
    }

    let x_center = (x_min + x_max) / 2.0;
    let y_center = (y_min + y_max) / 2.0;

    // Mirror across y-axis if first shape is right of center
    if c0.x() > x_center {
        for &idx in cluster {
            shapes[idx] = mirror_x_shape(&shapes[idx], x_center);
        }
    }

    // Recompute after potential x-mirror
    let c0 = shapes[idx0].centroid();

    // Mirror across x-axis if first shape is above center
    if c0.y() > y_center {
        for &idx in cluster {
            shapes[idx] = mirror_y_shape(&shapes[idx], y_center);
        }
    }
}

/// Rotate a shape around a pivot point.
fn rotate_shape<S>(shape: &S, theta: f64, pivot: &Point) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    // For most shapes, first two params are (x, y) and last is rotation
    if n >= 2 {
        let x = params[0];
        let y = params[1];

        // Rotate point around pivot
        let dx = x - pivot.x();
        let dy = y - pivot.y();
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let new_x = pivot.x() + dx * cos_t - dy * sin_t;
        let new_y = pivot.y() + dx * sin_t + dy * cos_t;

        let mut new_params = params.clone();
        new_params[0] = new_x;
        new_params[1] = new_y;

        // Update rotation parameter if it exists (last parameter)
        if n >= 5 {
            // Assuming last param is rotation (for ellipses)
            new_params[n - 1] = params[n - 1] - theta;
        }

        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Mirror a shape across a vertical line x = x_center.
fn mirror_x_shape<S>(shape: &S, x_center: f64) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    if n >= 1 {
        let mut new_params = params.clone();
        new_params[0] = 2.0 * x_center - params[0]; // Mirror x coordinate

        // For shapes with rotation, also mirror the rotation
        if n >= 5 {
            new_params[n - 1] = PI - params[n - 1]; // Mirror rotation
        }

        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Mirror a shape across a horizontal line y = y_center.
fn mirror_y_shape<S>(shape: &S, y_center: f64) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    if n >= 2 {
        let mut new_params = params.clone();
        new_params[1] = 2.0 * y_center - params[1]; // Mirror y coordinate

        // For shapes with rotation, also mirror the rotation
        if n >= 5 {
            new_params[n - 1] = PI - params[n - 1]; // Mirror rotation
        }

        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Pack disjoint clusters using the skyline algorithm.
fn pack_clusters<S>(shapes: &mut [S], clusters: &[Vec<usize>], padding_factor: f64)
where
    S: DiagramShape + Clone,
{
    if clusters.len() <= 1 {
        return;
    }

    // Compute bounding boxes for each cluster
    let mut cluster_boxes = Vec::new();

    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }

        // Compute overall bounding box for this cluster
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for &idx in cluster {
            let bbox = shapes[idx].bounding_box();
            let (bx_min, bx_max, by_min, by_max) = bbox.bounds();
            x_min = x_min.min(bx_min);
            x_max = x_max.max(bx_max);
            y_min = y_min.min(by_min);
            y_max = y_max.max(by_max);
        }

        let width = x_max - x_min;
        let height = y_max - y_min;
        let center = Point::new((x_min + x_max) / 2.0, (y_min + y_max) / 2.0);

        cluster_boxes.push((Rectangle::new(center, width, height), x_min, y_min));
    }

    // Compute padding based on total area (to match new packing algorithm)
    let total_area: f64 = cluster_boxes
        .iter()
        .map(|(r, _, _)| r.width() * r.height())
        .sum();
    let padding = total_area.sqrt() * padding_factor;

    // Pack the bounding boxes
    let rectangles: Vec<Rectangle> = cluster_boxes.iter().map(|(r, _, _)| *r).collect();
    let packed = skyline_pack(&rectangles, padding);

    // Update shape positions based on new bounding box positions
    for (i, cluster) in clusters.iter().enumerate() {
        if cluster.is_empty() {
            continue;
        }

        let (old_box, _old_x_min, _old_y_min) = cluster_boxes[i];
        let new_box = packed[i];

        // Compute translation
        let old_center = old_box.center();
        let new_center = new_box.center();
        let dx = new_center.x() - old_center.x();
        let dy = new_center.y() - old_center.y();

        // Translate all shapes in cluster
        for &idx in cluster {
            shapes[idx] = translate_shape(&shapes[idx], dx, dy);
        }
    }
}

/// Translate a shape by (dx, dy).
fn translate_shape<S>(shape: &S, dx: f64, dy: f64) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    if n >= 2 {
        let mut new_params = params.clone();
        new_params[0] += dx;
        new_params[1] += dy;
        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Center the entire layout around the origin.
fn center_layout<S>(shapes: &mut [S])
where
    S: DiagramShape + Clone,
{
    if shapes.is_empty() {
        return;
    }

    // Find bounding box of all shapes
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for shape in shapes.iter() {
        let bbox = shape.bounding_box();
        let (bx_min, bx_max, by_min, by_max) = bbox.bounds();
        x_min = x_min.min(bx_min);
        x_max = x_max.max(bx_max);
        y_min = y_min.min(by_min);
        y_max = y_max.max(by_max);
    }

    // Compute center
    let center_x = (x_min + x_max) / 2.0;
    let center_y = (y_min + y_max) / 2.0;

    // Translate all shapes to center the layout
    for shape in shapes.iter_mut() {
        *shape = translate_shape(shape, -center_x, -center_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::Circle;
    use crate::geometry::traits::Centroid;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_center_single_shape() {
        let mut shapes = vec![Circle::new(Point::new(5.0, 3.0), 2.0)];
        center_layout(&mut shapes);

        let centroid = shapes[0].centroid();
        assert!(approx_eq(centroid.x(), 0.0));
        assert!(approx_eq(centroid.y(), 0.0));
    }

    #[test]
    fn test_center_two_shapes() {
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(10.0, 0.0), 1.0),
        ];
        center_layout(&mut shapes);

        // Center should be at x=5, so after centering, shapes should be at x=-5 and x=5
        assert!(approx_eq(shapes[0].centroid().x(), -5.0));
        assert!(approx_eq(shapes[1].centroid().x(), 5.0));
        assert!(approx_eq(shapes[0].centroid().y(), 0.0));
        assert!(approx_eq(shapes[1].centroid().y(), 0.0));
    }

    #[test]
    fn test_translate_shape() {
        let shape = Circle::new(Point::new(1.0, 2.0), 3.0);
        let translated = translate_shape(&shape, 4.0, 5.0);

        assert!(approx_eq(translated.centroid().x(), 5.0));
        assert!(approx_eq(translated.centroid().y(), 7.0));
        assert_eq!(translated.radius(), 3.0);
    }

    #[test]
    fn test_normalize_single_shape() {
        let mut shapes = vec![Circle::new(Point::new(5.0, 3.0), 2.0)];
        normalize_layout(&mut shapes, 0.015);

        // Should just be centered
        let centroid = shapes[0].centroid();
        assert!(approx_eq(centroid.x(), 0.0));
        assert!(approx_eq(centroid.y(), 0.0));
    }

    #[test]
    fn test_normalize_two_overlapping() {
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 2.0), 2.0),
            Circle::new(Point::new(3.0, 2.0), 2.0),
        ];
        normalize_layout(&mut shapes, 0.015);

        // Should be rotated to horizontal and centered
        let c0 = shapes[0].centroid();
        let c1 = shapes[1].centroid();

        // Y coordinates should be equal (horizontal alignment)
        assert!(
            approx_eq(c0.y(), c1.y()),
            "Expected y coords to be equal, got {} and {}",
            c0.y(),
            c1.y()
        );

        // Layout should be centered
        let center_x = (c0.x() + c1.x()) / 2.0;
        let center_y = (c0.y() + c1.y()) / 2.0;
        assert!(approx_eq(center_x, 0.0));
        assert!(approx_eq(center_y, 0.0));

        // First shape should be to the left
        assert!(c0.x() < c1.x());
    }

    #[test]
    fn test_normalize_disjoint_clusters() {
        // Two disjoint pairs
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 1.5),
            Circle::new(Point::new(2.0, 0.0), 1.5),
            Circle::new(Point::new(20.0, 0.0), 1.0),
            Circle::new(Point::new(22.0, 0.0), 1.0),
        ];

        normalize_layout(&mut shapes, 0.015);

        // Both clusters should be rotated and packed together
        let c0 = shapes[0].centroid();
        let c1 = shapes[1].centroid();
        let c2 = shapes[2].centroid();
        let c3 = shapes[3].centroid();

        // First cluster should be horizontally aligned (was already horizontal)
        assert!(
            (c0.y() - c1.y()).abs() < 1e-3,
            "Cluster 1 should be horizontal, got y diff: {}",
            (c0.y() - c1.y()).abs()
        );

        // Clusters should be closer together than original (packed)
        // Original separation was 20 units between cluster centers
        let max_x = c0.x().max(c1.x()).max(c2.x()).max(c3.x());
        let min_x = c0.x().min(c1.x()).min(c2.x()).min(c3.x());
        let max_y = c0.y().max(c1.y()).max(c2.y()).max(c3.y());
        let min_y = c0.y().min(c1.y()).min(c2.y()).min(c3.y());

        let packed_width = max_x - min_x;
        let _packed_height = max_y - min_y;

        // The bounding box should be smaller than if they were 20 units apart horizontally
        let original_bbox_width = 22.0 + 1.0 - (0.0 - 1.5); // rightmost + radius - leftmost + radius

        assert!(
            packed_width < original_bbox_width,
            "Packed width {} should be less than original width {}",
            packed_width,
            original_bbox_width
        );

        // Verify layout is centered around origin using bounding boxes
        use crate::geometry::traits::BoundingBox;

        let mut bb_x_min = f64::INFINITY;
        let mut bb_x_max = f64::NEG_INFINITY;
        let mut bb_y_min = f64::INFINITY;
        let mut bb_y_max = f64::NEG_INFINITY;

        for shape in &shapes {
            let bbox = shape.bounding_box();
            let (bx_min, bx_max, by_min, by_max) = bbox.bounds();
            bb_x_min = bb_x_min.min(bx_min);
            bb_x_max = bb_x_max.max(bx_max);
            bb_y_min = bb_y_min.min(by_min);
            bb_y_max = bb_y_max.max(by_max);
        }

        let bb_center_x = (bb_x_max + bb_x_min) / 2.0;
        let bb_center_y = (bb_y_max + bb_y_min) / 2.0;

        assert!(
            (bb_center_x).abs() < 1e-6,
            "Should be centered in x, got {}",
            bb_center_x
        );
        assert!(
            (bb_center_y).abs() < 1e-6,
            "Should be centered in y, got {}",
            bb_center_y
        );
    }
}
