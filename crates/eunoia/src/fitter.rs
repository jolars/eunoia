//! Fitter for creating diagram layouts from specifications.

pub mod final_layout;
mod initial_layout;
mod layout;

pub use final_layout::Optimizer;
pub use layout::Layout;

use crate::error::DiagramError;
use crate::geometry::shapes::circle::distance_for_overlap;
use crate::geometry::shapes::Circle;
use crate::geometry::traits::DiagramShape;
use crate::loss::LossType;
use crate::spec::DiagramSpec;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Fitter for creating diagram layouts from specifications.
///
/// The type parameter `S` determines which shape type will be used (e.g., Circle, Ellipse).
/// The specification itself is shape-agnostic - the shape type is chosen here.
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
///     .build()
///     .unwrap();
///
/// // Choose shape type when fitting
/// let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
/// ```
pub struct Fitter<'a, S: DiagramShape = Circle> {
    spec: &'a DiagramSpec,
    max_iterations: usize,
    seed: Option<u64>,
    loss_type: LossType,
    optimizer: Optimizer,
    _shape: std::marker::PhantomData<S>,
}

impl<'a, S: DiagramShape + Copy + 'static> Fitter<'a, S> {
    /// Create a new fitter for the given specification.
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
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::<Circle>::new(&spec);
    /// ```
    pub fn new(spec: &'a DiagramSpec) -> Self {
        Fitter {
            spec,
            max_iterations: 100,
            seed: None,
            loss_type: LossType::region_error(),
            optimizer: Optimizer::default(),
            _shape: std::marker::PhantomData,
        }
    }

    /// Set maximum iterations for optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::<Circle>::new(&spec).max_iterations(500);
    /// ```
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the optimizer to use for final layout optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, Optimizer};
    /// use eunoia::geometry::shapes::Ellipse;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Try L-BFGS optimizer
    /// let fitter = Fitter::<Ellipse>::new(&spec).optimizer(Optimizer::Lbfgs);
    /// ```
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set random seed for reproducible layouts.
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
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
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
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .intersection(&["A", "B"], 2.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
    /// println!("Loss: {}", layout.loss());
    /// ```
    pub fn fit(self) -> Result<Layout<S>, DiagramError> {
        self.fit_with_optimization(true)
    }

    /// Fit the diagram, optionally skipping final optimization.
    ///
    /// When `optimize` is false, returns only the initial MDS-based layout.
    /// This is useful for debugging or comparing initial vs optimized layouts.
    pub fn fit_initial_only(self) -> Result<Layout<S>, DiagramError> {
        self.fit_with_optimization(false)
    }

    fn fit_with_optimization(self, optimize: bool) -> Result<Layout<S>, DiagramError> {
        let spec = self.spec.preprocess()?;
        let n_sets = spec.n_sets;

        // Create RNG based on seed
        let mut rng: Box<dyn rand::RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Step 1: Compute initial layout using MDS (always use circles for initial layout)
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

        let (final_params, _loss) = if optimize {
            let config = final_layout::FinalLayoutConfig {
                max_iterations: self.max_iterations,
                loss_type: self.loss_type,
                optimizer: self.optimizer,
                ..Default::default()
            };

            final_layout::optimize_layout::<S>(&spec, &initial_positions, &initial_radii, config)
                .map_err(|e| {
                    DiagramError::InvalidCombination(format!("Optimization failed: {}", e))
                })?
        } else {
            // Skip optimization, use initial circle parameters converted to shape params
            let mut params = Vec::new();
            for i in 0..n_sets {
                let x = initial_positions[i * 2];
                let y = initial_positions[i * 2 + 1];
                let r = initial_radii[i];
                params.extend(S::params_from_circle(x, y, r));
            }
            (params, 0.0)
        };

        // Step 3: Create final shapes from optimized parameters
        let params_per_shape = S::n_params();
        let mut shapes: Vec<S> = Vec::new();
        let mut set_to_shape = HashMap::new();

        for (i, set_name) in self.spec.set_names().iter().enumerate() {
            let start = i * params_per_shape;
            let end = start + params_per_shape;
            let shape = S::from_params(&final_params[start..end]);
            shapes.push(shape);
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
    ///
    /// This always uses circles for initial layout, regardless of final shape type.
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

        let layout = Fitter::<Circle>::new(&spec).fit().unwrap();

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

        let layout = Fitter::<Circle>::new(&spec).fit().unwrap();

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

        let layout = Fitter::<Circle>::new(&spec)
            .seed(42)
            .fit_initial_only()
            .unwrap();

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
        let layout1 = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let layout2 = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        assert_eq!(layout1.loss(), layout2.loss());

        // Verify shapes are identical
        for (s1, s2) in layout1.shapes().iter().zip(layout2.shapes().iter()) {
            assert_eq!(s1.center(), s2.center());
            assert_eq!(s1.radius(), s2.radius());
        }
    }

    #[test]
    fn test_fitter_with_ellipses_basic() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).fit().unwrap();

        assert_eq!(layout.shapes().len(), 2);
        assert!(layout.loss() >= 0.0);
    }

    #[test]
    fn test_fitter_with_ellipses_intersection() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();

        assert_eq!(layout.shapes().len(), 2);
        assert_eq!(layout.requested().len(), 3); // A, B, A&B
        println!("Ellipse layout loss: {}", layout.loss());
        assert!(layout.loss() < 10.0); // Should converge to reasonable solution
    }

    #[test]
    fn test_fitter_with_ellipses_three_sets() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 15.0)
            .set("B", 12.0)
            .set("C", 10.0)
            .intersection(&["A", "B"], 3.0)
            .intersection(&["B", "C"], 2.5)
            .intersection(&["A", "C"], 2.0)
            .intersection(&["A", "B", "C"], 1.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(123).fit().unwrap();

        assert_eq!(layout.shapes().len(), 3);
        println!("Three-ellipse layout loss: {}", layout.loss());
        assert!(layout.loss() < 20.0); // Should converge
    }

    #[test]
    fn test_ellipse_to_polygon_workflow() {
        use crate::geometry::shapes::{Ellipse, Polygon};
        use crate::geometry::traits::Polygonize;

        // Fit a diagram with ellipses
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();

        // Convert to polygons for plotting
        let polygons: Vec<Polygon> = layout
            .shapes()
            .iter()
            .map(|ellipse| ellipse.polygonize(64))
            .collect();

        assert_eq!(polygons.len(), 2);
        assert_eq!(polygons[0].vertices().len(), 64);
        assert_eq!(polygons[1].vertices().len(), 64);

        // Polygons should have areas close to ellipse areas
        use crate::geometry::traits::Area;
        for (ellipse, polygon) in layout.shapes().iter().zip(polygons.iter()) {
            let error = (ellipse.area() - polygon.area()).abs() / ellipse.area();
            assert!(
                error < 0.01,
                "Polygon area error too large: {:.2}%",
                error * 100.0
            ); // < 1% error with 64 vertices
        }
    }

    #[test]
    fn test_spurious_ac_intersection() {
        use crate::geometry::shapes::Ellipse;
        use crate::geometry::traits::{Area, Closed};
        use crate::spec::{Combination, DiagramSpecBuilder, InputType};

        // User reported: A=2.2, B=2, C=2, A&B&C=1 (exclusive)
        // Result shows A&C=0.059 but A&C don't visually intersect
        let spec = DiagramSpecBuilder::new()
            .set("A", 2.2)
            .set("B", 2.0)
            .set("C", 2.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        println!("\nTarget exclusive areas:");
        for (combo, &area) in spec.exclusive_areas() {
            println!("  {}: {:.3}", combo, area);
        }

        // Try multiple seeds to see if we can find a good solution
        let seeds = vec![42, 123, 456, 789, 1000];
        let mut best_loss = f64::INFINITY;
        let mut best_seed = 0;

        for &seed in &seeds {
            let fitter = Fitter::<Ellipse>::new(&spec).seed(seed);
            let layout = fitter.fit().unwrap();
            if layout.loss() < best_loss {
                best_loss = layout.loss();
                best_seed = seed;
            }
        }

        println!("\nBest seed: {} with loss: {:.6}", best_seed, best_loss);

        // Use best seed for detailed analysis
        let fitter = Fitter::<Ellipse>::new(&spec).seed(best_seed);
        let layout = fitter.fit().unwrap();

        println!("\nFitted exclusive areas:");
        for (combo, &area) in layout.fitted() {
            println!("  {}: {:.3}", combo, area);
        }

        println!("\nLoss: {:.6}", layout.loss());

        // Get shapes
        let shape_a = layout.shape_for_set("A").unwrap();
        let shape_b = layout.shape_for_set("B").unwrap();
        let shape_c = layout.shape_for_set("C").unwrap();

        println!("\nShape details:");
        println!(
            "A: center=({:.3}, {:.3}), semi_major={:.3}, semi_minor={:.3}, rotation={:.3}°, area={:.3}",
            shape_a.center().x(),
            shape_a.center().y(),
            shape_a.semi_major(),
            shape_a.semi_minor(),
            shape_a.rotation().to_degrees(),
            shape_a.area()
        );
        println!(
            "B: center=({:.3}, {:.3}), semi_major={:.3}, semi_minor={:.3}, rotation={:.3}°, area={:.3}",
            shape_b.center().x(),
            shape_b.center().y(),
            shape_b.semi_major(),
            shape_b.semi_minor(),
            shape_b.rotation().to_degrees(),
            shape_b.area()
        );
        println!(
            "C: center=({:.3}, {:.3}), semi_major={:.3}, semi_minor={:.3}, rotation={:.3}°, area={:.3}",
            shape_c.center().x(),
            shape_c.center().y(),
            shape_c.semi_major(),
            shape_c.semi_minor(),
            shape_c.rotation().to_degrees(),
            shape_c.area()
        );

        // Check actual intersections
        println!("\nActual geometry intersections:");
        println!("  A ∩ B: {}", shape_a.intersects(shape_b));
        println!("  A ∩ C: {}", shape_a.intersects(shape_c));
        println!("  B ∩ C: {}", shape_b.intersects(shape_c));

        let ab_points = shape_a.intersection_points(shape_b);
        let ac_points = shape_a.intersection_points(shape_c);
        let bc_points = shape_b.intersection_points(shape_c);

        println!("\nIntersection point counts:");
        println!("  A ∩ B: {} points", ab_points.len());
        println!("  A ∩ C: {} points", ac_points.len());
        println!("  B ∩ C: {} points", bc_points.len());

        // Check fitted areas
        let ac_combo = Combination::new(&["A", "C"]);
        let ac_fitted = layout.fitted().get(&ac_combo).copied().unwrap_or(0.0);

        println!("\nA&C fitted area: {:.6}", ac_fitted);

        // This is challenging - with ellipses, the optimizer may struggle
        // to avoid pairwise intersections when a 3-way intersection is required
        println!("\n⚠️  Note: This configuration is challenging for the optimizer.");
        println!("   Target has A&B&C intersection but no pairwise intersections.");
        println!("   Current loss: {:.3}", layout.loss());
    }
}
#[test]
fn test_circles_ac_issue_seed42() {
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Circle;
    use crate::geometry::traits::Closed;
    use crate::spec::{Combination, DiagramSpecBuilder, InputType};

    // User test case: A=2.2, B=2, C=3, A&B&C=1 (exclusive), seed=42
    // Shows A&C=0.339 but no visual intersection
    let spec = DiagramSpecBuilder::new()
        .set("A", 2.2)
        .set("B", 2.0)
        .set("C", 3.0)
        .intersection(&["A", "B", "C"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    println!("\nTarget exclusive areas:");
    for (combo, &area) in spec.exclusive_areas() {
        println!("  {}: {:.3}", combo, area);
    }

    let fitter = Fitter::<Circle>::new(&spec).seed(42);
    let layout = fitter.fit().unwrap();

    println!("\nFitted exclusive areas:");
    for (combo, &area) in layout.fitted() {
        println!("  {}: {:.3}", combo, area);
    }

    println!("\nLoss: {:.6}", layout.loss());

    // Get shapes
    let shape_a = layout.shape_for_set("A").unwrap();
    let shape_b = layout.shape_for_set("B").unwrap();
    let shape_c = layout.shape_for_set("C").unwrap();

    println!("\nCircle details:");
    println!(
        "A: center=({:.3}, {:.3}), radius={:.3}, area={:.3}",
        shape_a.center().x(),
        shape_a.center().y(),
        shape_a.radius(),
        std::f64::consts::PI * shape_a.radius().powi(2)
    );
    println!(
        "B: center=({:.3}, {:.3}), radius={:.3}, area={:.3}",
        shape_b.center().x(),
        shape_b.center().y(),
        shape_b.radius(),
        std::f64::consts::PI * shape_b.radius().powi(2)
    );
    println!(
        "C: center=({:.3}, {:.3}), radius={:.3}, area={:.3}",
        shape_c.center().x(),
        shape_c.center().y(),
        shape_c.radius(),
        std::f64::consts::PI * shape_c.radius().powi(2)
    );

    // Calculate center distances
    let dist_ac = ((shape_a.center().x() - shape_c.center().x()).powi(2)
        + (shape_a.center().y() - shape_c.center().y()).powi(2))
    .sqrt();
    let dist_ab = ((shape_a.center().x() - shape_b.center().x()).powi(2)
        + (shape_a.center().y() - shape_b.center().y()).powi(2))
    .sqrt();
    let dist_bc = ((shape_b.center().x() - shape_c.center().x()).powi(2)
        + (shape_b.center().y() - shape_c.center().y()).powi(2))
    .sqrt();

    println!("\nCenter-to-center distances:");
    println!(
        "  A-C: {:.3} (sum of radii: {:.3})",
        dist_ac,
        shape_a.radius() + shape_c.radius()
    );
    println!(
        "  A-B: {:.3} (sum of radii: {:.3})",
        dist_ab,
        shape_a.radius() + shape_b.radius()
    );
    println!(
        "  B-C: {:.3} (sum of radii: {:.3})",
        dist_bc,
        shape_b.radius() + shape_c.radius()
    );

    // Check actual intersections
    println!("\nActual geometry intersections:");
    println!("  A ∩ B: {}", shape_a.intersects(shape_b));
    println!("  A ∩ C: {}", shape_a.intersects(shape_c));
    println!("  B ∩ C: {}", shape_b.intersects(shape_c));

    let ab_points = shape_a.intersection_points(shape_b);
    let ac_points = shape_a.intersection_points(shape_c);
    let bc_points = shape_b.intersection_points(shape_c);

    println!("\nIntersection point counts:");
    println!("  A ∩ B: {} points", ab_points.len());
    println!("  A ∩ C: {} points", ac_points.len());
    println!("  B ∩ C: {} points", bc_points.len());

    // Check fitted areas
    let ac_combo = Combination::new(&["A", "C"]);
    let ac_fitted = layout.fitted().get(&ac_combo).copied().unwrap_or(0.0);

    println!("\nA&C fitted area: {:.6}", ac_fitted);

    // If distance > sum of radii, they can't intersect
    if dist_ac > shape_a.radius() + shape_c.radius() {
        println!("⚠️  Circles are SEPARATED (distance > sum of radii)");
        if ac_fitted > 0.001 {
            panic!(
                "❌ BUG: A&C fitted area is {:.3} but circles are separated!",
                ac_fitted
            );
        }
    }
}
#[test]
fn test_compare_optimizers() {
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Ellipse;
    use crate::spec::{DiagramSpecBuilder, InputType};

    let spec = DiagramSpecBuilder::new()
        .set("A", 2.2)
        .set("B", 2.0)
        .set("C", 2.0)
        .intersection(&["A", "B", "C"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    println!("\nComparing optimizers with seed=42:");

    // Try with default (L-BFGS)
    let fitter_default = Fitter::<Ellipse>::new(&spec).seed(42);
    let start = std::time::Instant::now();
    let layout_default = fitter_default.fit().unwrap();
    let time_default = start.elapsed();
    println!(
        "Default (L-BFGS): Loss={:.6}, Time={:.3}s",
        layout_default.loss(),
        time_default.as_secs_f64()
    );

    // Try with Nelder-Mead
    // TODO: Need a way to override optimizer in Fitter
}
