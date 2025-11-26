//! Fitter for creating diagram layouts from specifications.

pub mod final_layout;
mod initial_layout;
mod layout;

pub use layout::Layout;

use crate::error::DiagramError;
use crate::geometry::point::Point;
use crate::geometry::shapes::circle::distance_for_overlap;
use crate::geometry::shapes::circle::Circle;
use crate::loss::LossType;
use crate::spec::DiagramSpec;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Fitter for creating diagram layouts from specifications.
pub struct Fitter<'a> {
    spec: &'a DiagramSpec,
    max_iterations: usize,
    seed: Option<u64>,
    loss_type: LossType,
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
            seed: None,
            loss_type: LossType::region_error(),
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

    /// Set random seed for reproducible layouts.
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
    /// let layout = Fitter::new(&spec).seed(42).fit().unwrap();
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the loss function type for optimization.
    pub fn loss_type(mut self, loss_type: LossType) -> Self {
        self.loss_type = loss_type;
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
        self.fit_with_optimization(true)
    }

    /// Fit the diagram, optionally skipping final optimization.
    ///
    /// When `optimize` is false, returns only the initial MDS-based layout.
    /// This is useful for debugging or comparing initial vs optimized layouts.
    pub fn fit_initial_only(self) -> Result<Layout, DiagramError> {
        self.fit_with_optimization(false)
    }

    fn fit_with_optimization(self, optimize: bool) -> Result<Layout, DiagramError> {
        let spec = self.spec.preprocess()?;
        let n_sets = spec.n_sets;

        // Create RNG based on seed
        let mut rng: Box<dyn rand::RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Step 1: Compute initial layout using MDS
        let optimal_distances = Self::compute_optimal_distances(&spec)?;

        let initial_params = initial_layout::compute_initial_layout(
            &optimal_distances,
            &spec.relationships,
            &mut *rng,
        )
        .unwrap();

        let (x, y) = initial_params.split_at(n_sets);

        // Step 2: Optimize layout to minimize region error
        let initial_positions: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .flat_map(|(xi, yi)| vec![*xi, *yi])
            .collect();

        let initial_radii: Vec<f64> = spec
            .set_areas
            .iter()
            .map(|area| (area / std::f64::consts::PI).sqrt())
            .collect();

        let (final_positions, final_radii, _loss) = if optimize {
            let config = final_layout::FinalLayoutConfig {
                max_iterations: self.max_iterations,
                loss_type: self.loss_type,
                ..Default::default()
            };

            final_layout::optimize_layout(&spec, &initial_positions, &initial_radii, config)
                .map_err(|e| {
                    DiagramError::InvalidCombination(format!("Optimization failed: {}", e))
                })?
        } else {
            // Skip optimization, just use initial layout
            (initial_positions, initial_radii, 0.0)
        };

        // Step 3: Create final shapes
        let mut shapes = Vec::new();
        let mut set_to_shape = HashMap::new();

        for (i, set_name) in self.spec.set_names().iter().enumerate() {
            let x = final_positions[i * 2];
            let y = final_positions[i * 2 + 1];
            let radius = final_radii[i];

            shapes.push(Circle::new(Point::new(x, y), radius));
            set_to_shape.insert(set_name.clone(), i);
        }

        // Create and return the layout
        let layout = Layout::new(shapes, set_to_shape, self.spec, self.max_iterations);

        Ok(layout)
    }

    /// Compute optimal distances between circle centers based on desired overlaps.
    ///
    /// For each pair of sets, this calculates the distance between circle centers
    /// that would produce the desired overlap area given their radii.
    #[allow(clippy::needless_range_loop)]
    fn compute_optimal_distances(
        spec: &crate::spec::PreprocessedSpec,
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
    use crate::spec::DiagramSpecBuilder;

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

    #[test]
    fn test_russian_doll_initial_fit() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 1.0)
            .intersection(&["A", "B"], 1.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::new(&spec).seed(42).fit_initial_only().unwrap();

        println!("Initial layout loss: {}", layout.loss());

        // With seeded RNG, initial layout quality varies with seed
        assert!(layout.loss() <= 1e-3);
    }

    #[test]
    fn test_seed_reproducibility() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        // Same seed should produce identical results
        let layout1 = Fitter::new(&spec).seed(42).fit().unwrap();
        let layout2 = Fitter::new(&spec).seed(42).fit().unwrap();

        assert_eq!(layout1.loss(), layout2.loss());

        // Verify shapes are identical
        for (s1, s2) in layout1.shapes().iter().zip(layout2.shapes().iter()) {
            assert_eq!(s1.center(), s2.center());
            assert_eq!(s1.radius(), s2.radius());
        }
    }
}
