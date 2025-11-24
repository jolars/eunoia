//! Fitter for creating diagram layouts from specifications.

mod initial_layout;
mod layout;

pub use layout::Layout;

use crate::diagram::{Combination, DiagramSpec};
use crate::error::DiagramError;
use crate::geometry::coord::Coord;
use crate::geometry::shapes::circle::distance_for_overlap;
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
    /// let layout = Fitter::new(&spec).fit().unwrap();
    /// println!("Loss: {}", layout.loss());
    /// ```
    pub fn fit(self) -> Result<Layout, DiagramError> {
        let spec = self.spec.preprocess()?;
        let n_sets = spec.n_sets;

        // Compute optimal distances for circle centers based on desired overlaps
        let optimal_distances = Self::compute_optimal_distances(&spec)?;

        let initial_params =
            initial_layout::compute_initial_layout(&optimal_distances, &spec.relationships)
                .unwrap();

        let (x, y) = initial_params.split_at(n_sets);

        let mut shapes = Vec::new();
        let mut set_to_shape = HashMap::new();

        for (i, set_name) in self.spec.set_names().iter().enumerate() {
            let combo = Combination::new(&[set_name]);
            if let Some(area) = self.spec.get_union(&combo) {
                let radius = (area / std::f64::consts::PI).sqrt();
                let x = x[i];
                let y = y[i];

                shapes.push(Circle::new(Coord::new(x, y), radius));

                set_to_shape.insert(set_name.clone(), i);
            }
        }

        // Create and return the layout
        let layout = Layout::new(shapes, set_to_shape, self.spec, 0);

        Ok(layout)
    }

    /// Compute optimal distances between circle centers based on desired overlaps.
    ///
    /// For each pair of sets, this calculates the distance between circle centers
    /// that would produce the desired overlap area given their radii.
    #[allow(clippy::needless_range_loop)]
    fn compute_optimal_distances(
        spec: &crate::diagram::PreprocessedSpec,
    ) -> Result<Vec<Vec<f64>>, DiagramError> {
        let n_sets = spec.n_sets;
        let mut optimal_distances = vec![vec![0.0; n_sets]; n_sets];

        for i in 0..n_sets {
            for j in (i + 1)..n_sets {
                let overlap = spec.relationships.overlap_area(i, j);
                let r1 = (spec.set_areas[i] / std::f64::consts::PI).sqrt();
                let r2 = (spec.set_areas[j] / std::f64::consts::PI).sqrt();

                let desired_distance =
                    distance_for_overlap(r1, r2, overlap, None, None).map_err(|_| {
                        DiagramError::InvalidCombination(format!(
                            "Could not compute distance for sets {} and {}",
                            i, j
                        ))
                    })?;

                optimal_distances[i][j] = desired_distance;
                optimal_distances[j][i] = desired_distance;
            }
        }

        Ok(optimal_distances)
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

        let layout = Fitter::new(&spec).fit().unwrap();

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

        let layout = Fitter::new(&spec).fit().unwrap();

        assert_eq!(layout.shapes().len(), 2);
        assert_eq!(layout.requested().len(), 3); // A, B, A&B
    }
}
