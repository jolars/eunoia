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

    /// Get the area difference for a specific combination.
    pub fn error_for_combination(&self, combo: &Combination) -> Option<f64> {
        match (self.requested.get(combo), self.fitted.get(combo)) {
            (Some(&req), Some(&fit)) => Some((req - fit).abs()),
            _ => None,
        }
    }

    /// Compute all combination areas from current shapes.
    fn compute_fitted_areas(shapes: &[S], spec: &DiagramSpec) -> HashMap<Combination, f64> {
        let set_names = spec.set_names();

        // Check if S is Circle at runtime using type introspection
        if std::any::TypeId::of::<S>() == std::any::TypeId::of::<Circle>() {
            // SAFETY: We just checked that S == Circle
            let circles: &[Circle] = unsafe { std::mem::transmute(shapes) };
            diagram::compute_exclusive_areas_from_layout(circles, set_names)
        } else {
            // For other shapes, use Monte Carlo
            diagram::compute_exclusive_areas_from_layout_generic(shapes, set_names)
        }
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
}
