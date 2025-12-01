//! Layout representation - the result of fitting a diagram specification.

use crate::geometry::diagram;
use crate::geometry::shapes::Circle;
use crate::geometry::traits::DiagramShape;
use crate::spec::{Combination, DiagramSpec};
use std::collections::HashMap;

/// Result of fitting a diagram specification to shapes.
///
/// The type parameter `S` determines which shape type was used (e.g., Circle, Ellipse).
/// Defaults to `Circle` for backward compatibility.
#[derive(Debug, Clone)]
pub struct Layout<S: DiagramShape = Circle> {
    /// The fitted shapes (one per set).
    pub(crate) shapes: Vec<S>,

    /// Mapping from set names to shape indices.
    set_to_shape: HashMap<String, usize>,

    /// Original requested combination areas.
    requested: HashMap<Combination, f64>,

    /// Actual fitted combination areas (computed from shapes).
    fitted: HashMap<Combination, f64>,

    /// Final loss value.
    loss: f64,

    /// Number of iterations performed.
    iterations: usize,
}

impl<S: DiagramShape + Copy + 'static> Layout<S> {
    /// Creates a new layout from shapes and specification.
    ///
    /// This computes the fitted areas and loss automatically.
    pub(crate) fn new(
        shapes: Vec<S>,
        set_to_shape: HashMap<String, usize>,
        spec: &DiagramSpec,
        iterations: usize,
    ) -> Self {
        let requested = spec.exclusive_areas().clone();
        let fitted = Self::compute_fitted_areas(&shapes, spec);
        let loss = Self::compute_loss(&requested, &fitted);

        Layout {
            shapes,
            set_to_shape,
            requested,
            fitted,
            loss,
            iterations,
        }
    }

    /// Get the fitted shapes.
    pub fn shapes(&self) -> &[S] {
        &self.shapes
    }

    /// Get the requested areas.
    pub fn requested(&self) -> &HashMap<Combination, f64> {
        &self.requested
    }

    /// Get the actual fitted areas.
    pub fn fitted(&self) -> &HashMap<Combination, f64> {
        &self.fitted
    }

    /// Get the final loss value.
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Get the number of iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Get the shape for a specific set.
    pub fn shape_for_set(&self, set_name: &str) -> Option<&S> {
        self.set_to_shape
            .get(set_name)
            .map(|&idx| &self.shapes[idx])
    }

    /// Normalize the layout by rotating, centering, and packing disjoint clusters.
    ///
    /// This modifies the layout in-place to:
    /// 1. Rotate each cluster to a canonical orientation (first two shapes horizontal)
    /// 2. Mirror clusters so the first shape is in the bottom-left
    /// 3. Pack disjoint clusters together compactly
    /// 4. Center the entire layout around the origin
    ///
    /// # Arguments
    ///
    /// * `padding_factor` - Padding between clusters as a fraction of total width
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .intersection(&["A", "B"], 2.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let mut layout = Fitter::<Circle>::new(&spec).fit().unwrap();
    /// layout.normalize(0.015);
    /// ```
    pub fn normalize(&mut self, padding_factor: f64)
    where
        S: Clone,
    {
        crate::fitter::normalize::normalize_layout(&mut self.shapes, padding_factor);
    }

    /// Compute all combination areas from current shapes.
    fn compute_fitted_areas(shapes: &[S], spec: &DiagramSpec) -> HashMap<Combination, f64> {
        let set_names = spec.set_names();

        // Use the shape-specific exact computation method
        let exclusive_areas_by_mask = S::compute_exclusive_regions(shapes);

        // Convert RegionMask to Combination
        let mut exclusive_combos = HashMap::new();
        for (mask, area) in exclusive_areas_by_mask {
            if area > 1e-10 {
                // Only include non-negligible areas
                let indices = diagram::mask_to_indices(mask, shapes.len());
                let combo_sets: Vec<&str> =
                    indices.iter().map(|&i| set_names[i].as_str()).collect();

                if !combo_sets.is_empty() {
                    let combo = Combination::new(&combo_sets);
                    exclusive_combos.insert(combo, area);
                }
            }
        }

        exclusive_combos
    }

    /// Compute the loss between requested and fitted areas.
    fn compute_loss(
        requested: &HashMap<Combination, f64>,
        fitted: &HashMap<Combination, f64>,
    ) -> f64 {
        requested
            .iter()
            .map(|(combo, &req)| {
                let fit = fitted.get(combo).copied().unwrap_or(0.0);
                (req - fit).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::DiagramSpecBuilder;

    #[test]
    fn test_layout_creation() {
        use crate::geometry::primitives::Point;

        let spec = DiagramSpecBuilder::new()
            .set("A", std::f64::consts::PI)
            .build()
            .unwrap();

        let shapes = vec![Circle::new(Point::new(0.0, 0.0), 1.0)];
        let mut set_to_shape = HashMap::new();
        set_to_shape.insert("A".to_string(), 0);

        let layout = Layout::new(shapes, set_to_shape, &spec, 0);

        assert_eq!(layout.shapes().len(), 1);
        assert!(layout.loss() < 0.001); // Should be very close to π
    }

    #[test]
    fn test_shape_for_set() {
        use crate::geometry::primitives::Point;

        let spec = DiagramSpecBuilder::new().set("A", 10.0).build().unwrap();

        let shapes = vec![Circle::new(Point::new(1.0, 2.0), 3.0)];
        let mut set_to_shape = HashMap::new();
        set_to_shape.insert("A".to_string(), 0);

        let layout = Layout::new(shapes, set_to_shape, &spec, 0);

        let circle = layout.shape_for_set("A").unwrap();
        assert_eq!(circle.radius(), 3.0);
        assert_eq!(circle.center().x(), 1.0);
        assert_eq!(circle.center().y(), 2.0);
    }

    #[test]
    fn test_ellipse_area_computation_uses_exact_method() {
        use crate::geometry::primitives::Point;
        use crate::geometry::shapes::Ellipse;
        use crate::spec::InputType;

        // Test case: Three disjoint ellipses should NOT report intersection areas
        // This was the bug: Monte Carlo sampling was reporting spurious intersections
        let spec = DiagramSpecBuilder::new()
            .set("A", 2.9)
            .set("B", 4.9)
            .set("C", 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // Create three disjoint ellipses (circles for simplicity)
        let shapes = vec![
            Ellipse::new(Point::new(-5.0, 0.0), 1.0, 1.0, 0.0), // A: left
            Ellipse::new(Point::new(5.0, 0.0), 1.3, 1.3, 0.0),  // B: right
            Ellipse::new(Point::new(0.0, 5.0), 0.6, 0.6, 0.0),  // C: top
        ];

        let mut set_to_shape = HashMap::new();
        set_to_shape.insert("A".to_string(), 0);
        set_to_shape.insert("B".to_string(), 1);
        set_to_shape.insert("C".to_string(), 2);

        let layout = Layout::new(shapes, set_to_shape, &spec, 0);

        // Check fitted areas - there should be NO intersection areas
        let ab_combo = Combination::new(&["A", "B"]);
        let ac_combo = Combination::new(&["A", "C"]);
        let bc_combo = Combination::new(&["B", "C"]);
        let abc_combo = Combination::new(&["A", "B", "C"]);

        let ab_area = layout.fitted().get(&ab_combo).copied().unwrap_or(0.0);
        let ac_area = layout.fitted().get(&ac_combo).copied().unwrap_or(0.0);
        let bc_area = layout.fitted().get(&bc_combo).copied().unwrap_or(0.0);
        let abc_area = layout.fitted().get(&abc_combo).copied().unwrap_or(0.0);

        // All intersection areas should be zero (or negligible) since shapes are disjoint
        assert!(
            ab_area < 1e-6,
            "A&B should be ~0 for disjoint shapes, got {}",
            ab_area
        );
        assert!(
            ac_area < 1e-6,
            "A&C should be ~0 for disjoint shapes, got {}",
            ac_area
        );
        assert!(
            bc_area < 1e-6,
            "B&C should be ~0 for disjoint shapes, got {}",
            bc_area
        );
        assert!(
            abc_area < 1e-6,
            "A&B&C should be ~0 for disjoint shapes, got {}",
            abc_area
        );

        // Individual areas should match shape areas
        let a_only = layout
            .fitted()
            .get(&Combination::new(&["A"]))
            .copied()
            .unwrap_or(0.0);
        let expected_a = std::f64::consts::PI * 1.0 * 1.0;
        assert!(
            (a_only - expected_a).abs() < 0.01,
            "A area should be ~π, got {}",
            a_only
        );
    }
}
