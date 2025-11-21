//! Fitter for creating diagram layouts from specifications.

mod layout;

pub use layout::Layout;

use crate::diagram::{Combination, DiagramSpec};
use crate::geometry::coord::Coord;
use crate::geometry::shapes::circle::Circle;
use std::collections::HashMap;

/// Fitter for creating diagram layouts from specifications.
pub struct Fitter<'a> {
    spec: &'a DiagramSpec,
    max_iterations: usize,
}

impl<'a> Fitter<'a> {
    /// Create a new fitter for the given specification.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::new(&spec);
    /// ```
    pub fn new(spec: &'a DiagramSpec) -> Self {
        Fitter {
            spec,
            max_iterations: 100,
        }
    }

    /// Set maximum iterations for optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::new(&spec).max_iterations(500);
    /// ```
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Fit the diagram using circles.
    ///
    /// This creates a layout with circles positioned to match the specification.
    /// Currently uses a simple grid initialization; optimization will be added later.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .intersection(&["A", "B"], 2.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::new(&spec).fit();
    /// println!("Loss: {}", layout.loss());
    /// ```
    pub fn fit(self) -> Layout {
        let mut shapes = Vec::new();
        let mut set_to_shape = HashMap::new();

        // Create initial circles (simple grid placement)
        for (i, set_name) in self.spec.set_names().iter().enumerate() {
            let combo = Combination::new(&[set_name]);
            if let Some(area) = self.spec.get_combination(&combo) {
                let radius = (area / std::f64::consts::PI).sqrt();
                // Simple grid placement for now
                let x = (i as f64) * 3.0 * radius;
                let y = 0.0;

                shapes.push(Circle::new(Coord::new(x, y), radius));
                set_to_shape.insert(set_name.clone(), i);
            }
        }

        // Create and return the layout
        Layout::new(shapes, set_to_shape, self.spec, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagram::DiagramSpecBuilder;

    #[test]
    fn test_fitter_basic() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .build()
            .unwrap();

        let layout = Fitter::new(&spec).fit();

        assert_eq!(layout.shapes().len(), 2);
        assert!(layout.loss() >= 0.0);
    }

    #[test]
    fn test_fitter_with_intersection() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let layout = Fitter::new(&spec).fit();

        assert_eq!(layout.shapes().len(), 2);
        assert_eq!(layout.requested().len(), 3); // A, B, A&B
    }
}
