//! Layout representation - the result of fitting a diagram specification.

use crate::geometry::diagram;
use crate::geometry::shapes::Circle;
use crate::geometry::traits::DiagramShape;
use crate::loss::LossType;
use crate::spec::{Combination, DiagramSpec};
use std::collections::{HashMap, HashSet};

/// Result of fitting a diagram specification to shapes.
///
/// The type parameter `S` determines which shape type was used (e.g., Circle, Ellipse).
/// Defaults to `Circle` for backward compatibility.
#[derive(Debug, Clone)]
pub struct Layout<S: DiagramShape = Circle> {
    /// The fitted shapes (one per set, in the order of the original spec's set names).
    ///
    /// Sets that were pruned during preprocessing (e.g. empty sets) are represented
    /// by zero-parameter shapes at their original index.
    pub(crate) shapes: Vec<S>,

    /// Mapping from set names to shape indices.
    set_to_shape: HashMap<String, usize>,

    /// Original requested combination areas.
    requested: HashMap<Combination, f64>,

    /// Actual fitted combination areas (computed from shapes).
    fitted: HashMap<Combination, f64>,

    /// The loss function used during optimization (same objective is reported by `loss()`).
    loss_type: LossType,

    /// Final loss value (computed using `loss_type`).
    loss: f64,

    /// Number of iterations performed.
    iterations: usize,
}

impl<S: DiagramShape + Copy + 'static> Layout<S> {
    /// Creates a new layout from shapes and specification.
    ///
    /// This computes the fitted areas and loss automatically. The `loss_type` determines
    /// the objective reported by [`Layout::loss`] — it should match the objective the
    /// fitter minimized so callers see a self-consistent value.
    pub(crate) fn new(
        shapes: Vec<S>,
        set_to_shape: HashMap<String, usize>,
        spec: &DiagramSpec,
        iterations: usize,
        loss_type: LossType,
    ) -> Self {
        let requested = spec.exclusive_areas().clone();
        let fitted = Self::compute_fitted_areas(&shapes, spec);
        let loss = Self::compute_loss(&requested, &fitted, loss_type);

        Layout {
            shapes,
            set_to_shape,
            requested,
            fitted,
            loss_type,
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

    /// Get the final loss value, computed using the optimizer's objective ([`Layout::loss_type`]).
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Get the loss function used by the optimizer.
    pub fn loss_type(&self) -> LossType {
        self.loss_type
    }

    /// Residuals per region: `requested - fitted`.
    ///
    /// Includes every combination that appears in either requested or fitted areas.
    pub fn residuals(&self) -> HashMap<Combination, f64> {
        self.all_combinations()
            .into_iter()
            .map(|combo| {
                let t = self.requested.get(&combo).copied().unwrap_or(0.0);
                let f = self.fitted.get(&combo).copied().unwrap_or(0.0);
                (combo, t - f)
            })
            .collect()
    }

    /// Per-region error: `|f_i / Σf - t_i / Σt|` for each combination.
    ///
    /// Matches eulerr's `regionError` definition. Returns an empty map if either the
    /// sum of fitted or the sum of requested areas is effectively zero.
    pub fn region_error(&self) -> HashMap<Combination, f64> {
        let sum_f: f64 = self.fitted.values().sum();
        let sum_t: f64 = self.requested.values().sum();

        if sum_f.abs() < 1e-10 || sum_t.abs() < 1e-10 {
            return HashMap::new();
        }

        self.all_combinations()
            .into_iter()
            .map(|combo| {
                let f = self.fitted.get(&combo).copied().unwrap_or(0.0);
                let t = self.requested.get(&combo).copied().unwrap_or(0.0);
                (combo, (f / sum_f - t / sum_t).abs())
            })
            .collect()
    }

    /// Diagnostic error: the maximum of [`Layout::region_error`] values.
    ///
    /// Matches eulerr's `diagError` scalar (EulerAPE style).
    pub fn diag_error(&self) -> f64 {
        self.region_error()
            .values()
            .copied()
            .fold(0.0_f64, f64::max)
    }

    /// venneuler-style stress metric (matches eulerr's `stress`):
    /// `Σ(f - β·t)² / Σf²` where `β = Σ(f·t) / Σt²`.
    ///
    /// Returns 0.0 if `Σt² < ε` or `Σf² < ε`.
    pub fn stress(&self) -> f64 {
        let combos = self.all_combinations();
        let sum_ft: f64 = combos
            .iter()
            .map(|c| {
                let f = self.fitted.get(c).copied().unwrap_or(0.0);
                let t = self.requested.get(c).copied().unwrap_or(0.0);
                f * t
            })
            .sum();
        let sum_t2: f64 = self.requested.values().map(|&v| v * v).sum();
        let sum_f2: f64 = self.fitted.values().map(|&v| v * v).sum();

        if sum_t2 < 1e-20 || sum_f2 < 1e-20 {
            return 0.0;
        }

        let beta = sum_ft / sum_t2;
        let numerator: f64 = combos
            .into_iter()
            .map(|c| {
                let f = self.fitted.get(&c).copied().unwrap_or(0.0);
                let t = self.requested.get(&c).copied().unwrap_or(0.0);
                (f - beta * t).powi(2)
            })
            .sum();
        numerator / sum_f2
    }

    fn all_combinations(&self) -> Vec<Combination> {
        let set: HashSet<Combination> = self
            .requested
            .keys()
            .chain(self.fitted.keys())
            .cloned()
            .collect();
        set.into_iter().collect()
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

    /// Decomposes the fitted shapes into polygons for each exclusive region.
    ///
    /// This is useful for visualization where you want to fill each region
    /// with a different color or pattern.
    ///
    /// **Requires the `plotting` feature to be enabled.**
    ///
    /// # Arguments
    ///
    /// * `spec` - The diagram specification
    /// * `n_vertices` - Number of vertices to use when converting shapes to polygons (e.g., 64)
    ///
    /// # Examples
    ///
    /// ```ignore
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
    /// let regions = layout.region_polygons(&spec, 64);
    ///
    /// // Iterate over regions
    /// for (combination, polygons) in regions.iter() {
    ///     println!("{}: {} polygons", combination, polygons.len());
    /// }
    /// ```
    #[cfg(feature = "plotting")]
    pub fn region_polygons(
        &self,
        spec: &DiagramSpec,
        n_vertices: usize,
    ) -> crate::plotting::RegionPolygons
    where
        S: crate::geometry::traits::Polygonize,
    {
        let set_names = spec.set_names();
        crate::plotting::decompose_regions(&self.shapes, set_names, spec, n_vertices)
    }

    /// Builds a [`PlotData`] bundle — region polygons, per-region and per-set
    /// label anchors, and per-set outlines — in one call.
    ///
    /// This is the recommended entry point for renderers and language
    /// bindings: it computes everything a typical Euler-diagram drawing
    /// routine needs from the fitted layout, with options for
    /// polygonization resolution and label-anchor precision.
    ///
    /// **Requires the `plotting` feature to be enabled.**
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
    /// use eunoia::geometry::shapes::Circle;
    /// use eunoia::plotting::PlotOptions;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 5.0)
    ///     .set("B", 3.0)
    ///     .intersection(&["A", "B"], 1.0)
    ///     .input_type(InputType::Exclusive)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
    /// let plot = layout.plot_data(&spec, PlotOptions::default());
    ///
    /// // Hand `plot.regions`, `plot.region_anchors`, `plot.set_anchors`,
    /// // `plot.shape_outlines` to the renderer of your choice.
    /// ```
    ///
    /// [`PlotData`]: crate::plotting::PlotData
    #[cfg(feature = "plotting")]
    pub fn plot_data(
        &self,
        spec: &DiagramSpec,
        options: crate::plotting::PlotOptions,
    ) -> crate::plotting::PlotData
    where
        S: crate::geometry::traits::Polygonize,
    {
        crate::plotting::build_plot_data(&self.shapes, spec, options)
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

    /// Compute the loss between requested and fitted areas using the optimizer's loss type.
    fn compute_loss(
        requested: &HashMap<Combination, f64>,
        fitted: &HashMap<Combination, f64>,
        loss_type: LossType,
    ) -> f64 {
        // LossType operates on RegionMask-keyed maps, but the mask encoding is only
        // relevant within a single call. Assign each distinct combination to a unique
        // mask so LossType sees the same pairing of fitted/requested values.
        let mut combo_to_mask: HashMap<&Combination, diagram::RegionMask> = HashMap::new();
        for combo in requested.keys().chain(fitted.keys()) {
            let next = combo_to_mask.len();
            combo_to_mask.entry(combo).or_insert(next);
        }

        let fitted_mask: HashMap<diagram::RegionMask, f64> =
            fitted.iter().map(|(c, &v)| (combo_to_mask[c], v)).collect();
        let requested_mask: HashMap<diagram::RegionMask, f64> = requested
            .iter()
            .map(|(c, &v)| (combo_to_mask[c], v))
            .collect();

        loss_type.compute(&fitted_mask, &requested_mask)
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

        let layout = Layout::new(shapes, set_to_shape, &spec, 0, LossType::sse());

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

        let layout = Layout::new(shapes, set_to_shape, &spec, 0, LossType::sse());

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

        let layout = Layout::new(shapes, set_to_shape, &spec, 0, LossType::sse());

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

    #[test]
    fn test_empty_set_reinsertion() {
        // A spec with one empty set (C = 0) should still produce a layout with an
        // entry (zero shape) for C in its original position.
        use crate::fitter::Fitter;
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .set("C", 0.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        // All 3 original sets should be represented.
        assert_eq!(layout.shapes().len(), 3);

        // C must be accessible by name and have zero area.
        let c_shape = layout.shape_for_set("C").expect("C should be present");
        assert_eq!(c_shape.radius(), 0.0);

        // Surviving sets should have positive-area shapes.
        let a_shape = layout.shape_for_set("A").expect("A should be present");
        let b_shape = layout.shape_for_set("B").expect("B should be present");
        assert!(a_shape.radius() > 0.0);
        assert!(b_shape.radius() > 0.0);
    }

    #[test]
    fn test_named_metric_accessors() {
        use crate::fitter::Fitter;
        // A simple 2-set overlap. The fitter should get close to exact for circles.
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .intersection(&["A", "B"], 1.0)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        // residuals, region_error should have one entry per combination.
        assert!(!layout.residuals().is_empty());
        let region_err = layout.region_error();
        assert!(!region_err.is_empty());

        // diagError = max(regionError)
        let max_region_err = region_err.values().copied().fold(0.0_f64, f64::max);
        assert!((layout.diag_error() - max_region_err).abs() < 1e-12);

        // stress is bounded [0, 1] for sensible inputs and finite.
        let stress = layout.stress();
        assert!(stress.is_finite());
        assert!(stress >= 0.0);
    }

    #[test]
    fn test_loss_reflects_loss_type() {
        // Layout.loss() should match the LossType the optimizer minimized.
        use crate::fitter::Fitter;
        use crate::loss::LossType;

        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .intersection(&["A", "B"], 1.0)
            .build()
            .unwrap();

        let layout_sse = Fitter::<Circle>::new(&spec)
            .seed(42)
            .loss_type(LossType::sse())
            .fit()
            .unwrap();
        let layout_rmse = Fitter::<Circle>::new(&spec)
            .seed(42)
            .loss_type(LossType::rmse())
            .fit()
            .unwrap();

        assert_eq!(layout_sse.loss_type(), LossType::sse());
        assert_eq!(layout_rmse.loss_type(), LossType::rmse());

        // Given identical fits, RMSE = sqrt(SSE / n). Since both optimized with the
        // same seed and the region structure is the same, at least the relationship
        // sse >= rmse^2 * n_regions / 1 (approximately) should hold. Instead of
        // asserting a tight relationship (the optimizer may take different paths),
        // just check each loss is finite and non-negative.
        assert!(layout_sse.loss().is_finite() && layout_sse.loss() >= 0.0);
        assert!(layout_rmse.loss().is_finite() && layout_rmse.loss() >= 0.0);
    }
}
