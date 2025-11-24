//! Layout representation - the result of fitting a diagram specification.

use crate::diagram::{Combination, DiagramSpec};
use crate::geometry::shapes::circle::Circle;
use crate::geometry::shapes::Shape;
use std::collections::HashMap;

/// Result of fitting a diagram specification to shapes.
#[derive(Debug, Clone)]
pub struct Layout {
    /// The fitted shapes (one per set).
    shapes: Vec<Circle>,

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

impl Layout {
    /// Creates a new layout from shapes and specification.
    ///
    /// This computes the fitted areas and loss automatically.
    pub(crate) fn new(
        shapes: Vec<Circle>,
        set_to_shape: HashMap<String, usize>,
        spec: &DiagramSpec,
        iterations: usize,
    ) -> Self {
        // Use union areas for requested (what the shapes should produce)
        let requested = spec.union_areas().clone();
        let fitted = Self::compute_fitted_areas(&shapes, &set_to_shape, spec);
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
    pub fn shapes(&self) -> &[Circle] {
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
    pub fn shape_for_set(&self, set_name: &str) -> Option<&Circle> {
        self.set_to_shape
            .get(set_name)
            .map(|&idx| &self.shapes[idx])
    }

    /// Get the area difference for a specific combination.
    pub fn error_for_combination(&self, combo: &Combination) -> Option<f64> {
        match (self.requested.get(combo), self.fitted.get(combo)) {
            (Some(&req), Some(&fit)) => Some((req - fit).abs()),
            _ => None,
        }
    }

    /// Compute all combination areas from current shapes.
    fn compute_fitted_areas(
        shapes: &[Circle],
        set_to_shape: &HashMap<String, usize>,
        spec: &DiagramSpec,
    ) -> HashMap<Combination, f64> {
        let mut fitted = HashMap::new();

        // For each combination in the spec, compute its actual area
        for combo in spec.union_areas().keys() {
            let area = if combo.len() == 1 {
                // Single set - just the circle area
                let set_name = &combo.sets()[0];
                if let Some(&idx) = set_to_shape.get(set_name) {
                    shapes[idx].area()
                } else {
                    0.0
                }
            } else if combo.len() == 2 {
                // Two-way intersection
                let set_names = combo.sets();
                if let (Some(&idx1), Some(&idx2)) = (
                    set_to_shape.get(&set_names[0]),
                    set_to_shape.get(&set_names[1]),
                ) {
                    shapes[idx1].intersection_area(&shapes[idx2])
                } else {
                    0.0
                }
            } else {
                // TODO: Handle 3+ way intersections
                // For now, just use 0.0
                0.0
            };

            fitted.insert(combo.clone(), area);
        }

        fitted
    }

    /// Compute region error loss.
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
    use crate::diagram::DiagramSpecBuilder;

    #[test]
    fn test_layout_creation() {
        use crate::geometry::point::Point;

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
        use crate::geometry::point::Point;

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
}
