//! 2D rectangle packing for compact layouts.
//!
//! This module implements a grid-based packing algorithm that arranges
//! rectangular bounding boxes in a compact layout targeting the golden ratio.

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;

const GOLDEN_RATIO: f64 = 1.618;

/// Pack rectangles using a grid-based layout targeting the golden ratio.
///
/// Arranges rectangles in a grid pattern aiming for an aspect ratio close
/// to the golden ratio (≈1.618), which is aesthetically pleasing.
///
/// # Arguments
///
/// * `rectangles` - Rectangles to pack (center and dimensions are used)
/// * `padding` - Extra spacing between rectangles
///
/// # Returns
///
/// New rectangles with updated positions. The dimensions remain unchanged.
pub fn skyline_pack(rectangles: &[Rectangle], padding: f64) -> Vec<Rectangle> {
    if rectangles.is_empty() {
        return vec![];
    }

    let n = rectangles.len();

    // Sort by area (largest first) for better visual balance
    let mut indexed_rects: Vec<(usize, &Rectangle)> = rectangles.iter().enumerate().collect();
    indexed_rects.sort_by(|a, b| {
        let area_a = a.1.width() * a.1.height();
        let area_b = b.1.width() * b.1.height();
        area_b.partial_cmp(&area_a).unwrap()
    });

    // Determine grid dimensions targeting golden ratio
    let cols = ((n as f64).sqrt() * GOLDEN_RATIO.sqrt()).ceil() as usize;
    let cols = cols.max(1);
    let rows = (n as f64 / cols as f64).ceil() as usize;

    // Calculate cell sizes (max width/height in each row/column)
    let mut col_widths = vec![0.0; cols];
    let mut row_heights = vec![0.0; rows];

    for (i, (_idx, rect)) in indexed_rects.iter().enumerate() {
        let row = i / cols;
        let col = i % cols;
        if row < rows {
            col_widths[col] = f64::max(col_widths[col], rect.width() + padding);
            row_heights[row] = f64::max(row_heights[row], rect.height() + padding);
        }
    }

    // Calculate cumulative positions
    let mut col_positions = vec![0.0];
    for &width in &col_widths {
        let last = col_positions.last().unwrap();
        col_positions.push(last + width);
    }

    let mut row_positions = vec![0.0];
    for &height in &row_heights {
        let last = row_positions.last().unwrap();
        row_positions.push(last + height);
    }

    // Place rectangles in grid
    let mut result = vec![Rectangle::new(Point::new(0.0, 0.0), 0.0, 0.0); n];

    for (i, (orig_idx, rect)) in indexed_rects.iter().enumerate() {
        let row = i / cols;
        let col = i % cols;

        if row < rows {
            // Center rectangle in its grid cell
            let cell_x = col_positions[col];
            let cell_y = row_positions[row];
            let cell_w = col_widths[col];
            let cell_h = row_heights[row];

            let x = cell_x + (cell_w - rect.width()) / 2.0;
            let y = cell_y + (cell_h - rect.height()) / 2.0;

            let center_x = x + rect.width() / 2.0;
            let center_y = y + rect.height() / 2.0;

            result[*orig_idx] =
                Rectangle::new(Point::new(center_x, center_y), rect.width(), rect.height());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_empty() {
        let rectangles = vec![];
        let packed = skyline_pack(&rectangles, 0.0);
        assert_eq!(packed.len(), 0);
    }

    #[test]
    fn test_pack_single() {
        let rectangles = vec![Rectangle::new(Point::new(0.0, 0.0), 2.0, 3.0)];
        let packed = skyline_pack(&rectangles, 0.0);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].width(), 2.0);
        assert_eq!(packed[0].height(), 3.0);
    }

    #[test]
    fn test_pack_two_rectangles() {
        let rectangles = vec![
            Rectangle::new(Point::new(0.0, 0.0), 3.0, 2.0),
            Rectangle::new(Point::new(0.0, 0.0), 2.0, 1.0),
        ];
        let packed = skyline_pack(&rectangles, 0.0);
        assert_eq!(packed.len(), 2);

        // Verify dimensions are preserved
        assert_eq!(packed[0].width(), 3.0);
        assert_eq!(packed[0].height(), 2.0);
        assert_eq!(packed[1].width(), 2.0);
        assert_eq!(packed[1].height(), 1.0);

        // Verify they don't overlap
        for i in 0..packed.len() {
            for j in (i + 1)..packed.len() {
                let (x1_min, x1_max, y1_min, y1_max) = packed[i].bounds();
                let (x2_min, x2_max, y2_min, y2_max) = packed[j].bounds();

                let overlaps =
                    !(x1_max <= x2_min || x2_max <= x1_min || y1_max <= y2_min || y2_max <= y1_min);
                assert!(!overlaps, "Rectangles {} and {} overlap", i, j);
            }
        }
    }

    #[test]
    fn test_pack_with_padding() {
        let rectangles = vec![
            Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0),
            Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0),
        ];
        let padding = 0.5;
        let packed = skyline_pack(&rectangles, padding);

        // Dimensions should be unchanged
        assert_eq!(packed[0].width(), 2.0);
        assert_eq!(packed[1].width(), 2.0);

        // They should not overlap
        let (x1_min, x1_max, y1_min, y1_max) = packed[0].bounds();
        let (x2_min, x2_max, y2_min, y2_max) = packed[1].bounds();

        let overlaps =
            !(x1_max <= x2_min || x2_max <= x1_min || y1_max <= y2_min || y2_max <= y1_min);
        assert!(!overlaps, "Rectangles should not overlap");
    }

    #[test]
    fn test_pack_multiple_sizes() {
        let rectangles = vec![
            Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0),
            Rectangle::new(Point::new(0.0, 0.0), 2.0, 3.0),
            Rectangle::new(Point::new(0.0, 0.0), 3.0, 1.0),
            Rectangle::new(Point::new(0.0, 0.0), 1.0, 4.0),
        ];
        let packed = skyline_pack(&rectangles, 0.1);

        assert_eq!(packed.len(), 4);

        // Verify no overlaps
        for i in 0..packed.len() {
            for j in (i + 1)..packed.len() {
                let (x1_min, x1_max, y1_min, y1_max) = packed[i].bounds();
                let (x2_min, x2_max, y2_min, y2_max) = packed[j].bounds();

                let overlaps =
                    !(x1_max <= x2_min || x2_max <= x1_min || y1_max <= y2_min || y2_max <= y1_min);
                assert!(!overlaps, "Rectangles {} and {} overlap", i, j);
            }
        }
    }

    #[test]
    fn test_golden_ratio_aspect() {
        // Test that we get reasonable aspect ratios
        let rectangles = vec![
            Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0),
            Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0),
            Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0),
        ];
        let packed = skyline_pack(&rectangles, 0.1);

        // Calculate bounding box
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for rect in &packed {
            let (bx_min, bx_max, by_min, by_max) = rect.bounds();
            x_min = x_min.min(bx_min);
            x_max = x_max.max(bx_max);
            y_min = y_min.min(by_min);
            y_max = y_max.max(by_max);
        }

        let width = x_max - x_min;
        let height = y_max - y_min;
        let aspect = width / height;

        // Should be somewhere between 0.5 and 4.0 (reasonable range for small n)
        assert!(
            aspect > 0.4 && aspect < 4.5,
            "Aspect ratio {} is too extreme",
            aspect
        );
    }

    #[test]
    fn test_many_rectangles_aspect_ratio() {
        // Test with many rectangles to see if we get close to golden ratio
        let rectangles: Vec<Rectangle> = (0..9)
            .map(|_| Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0))
            .collect();
        let packed = skyline_pack(&rectangles, 0.1);

        // Calculate bounding box
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for rect in &packed {
            let (bx_min, bx_max, by_min, by_max) = rect.bounds();
            x_min = x_min.min(bx_min);
            x_max = x_max.max(bx_max);
            y_min = y_min.min(by_min);
            y_max = y_max.max(by_max);
        }

        let width = x_max - x_min;
        let height = y_max - y_min;
        let aspect = width / height;

        println!(
            "9 rectangles: aspect ratio = {:.2} (golden = 1.618)",
            aspect
        );
        // Should be reasonable - not too tall or too wide
        assert!(
            aspect > 0.8 && aspect < 2.5,
            "Aspect ratio {} is too extreme",
            aspect
        );
    }
}
