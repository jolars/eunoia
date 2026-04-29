//! Canonical Venn-diagram layouts with ellipses.
//!
//! [`VennDiagram::new(n)`] returns a hardcoded n-set ellipse arrangement where
//! every one of the `2ⁿ − 1` regions is non-empty (i.e. a true Venn diagram).
//! The arrangements are taken verbatim from the eulerr R package's `venn_spec`
//! data table, which in turn reproduces the published constructions:
//!
//! - n = 1, 2, 3: rotationally symmetric circles.
//! - n = 4: Venn (1880) 4-ellipse arrangement; see Wilkinson (2012),
//!   *JCGS* "Exact Rotational Symmetry in Venn Diagrams".
//! - n = 5: Grünbaum (1975) symmetric 5-ellipse Venn diagram.
//!
//! No Venn diagram exists for 6 or 7 ellipses (Edwards's higher-n
//! constructions use non-ellipse cogwheel curves), so `n ≥ 6` returns
//! [`DiagramError::UnsupportedSetCount`].

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::error::DiagramError;
use crate::fitter::Layout;
use crate::geometry::primitives::Point;
use crate::geometry::shapes::Ellipse;
use crate::loss::LossType;
use crate::spec::{DiagramSpec, DiagramSpecBuilder, InputType};

/// Ellipse parameters as `(h, k, a, b, phi)` — center, semi-major, semi-minor, rotation.
type EllipseParams = (f64, f64, f64, f64, f64);

/// A canonical Venn-diagram layout with ellipses.
///
/// Use [`VennDiagram::new`] to construct one from a set count `n`, and
/// [`VennDiagram::into_layout`] to obtain a [`Layout<Ellipse>`] suitable
/// for the same plotting/inspection pipeline as a fitted layout.
///
/// # Example
///
/// ```
/// use eunoia::VennDiagram;
///
/// let venn = VennDiagram::new(3).unwrap();
/// assert_eq!(venn.ellipses().len(), 3);
/// assert_eq!(venn.names(), &["A", "B", "C"]);
/// ```
#[derive(Debug, Clone)]
pub struct VennDiagram {
    ellipses: Vec<Ellipse>,
    names: Vec<String>,
}

impl VennDiagram {
    /// Constructs the canonical Venn arrangement for `n` sets.
    ///
    /// Default set names are the first `n` uppercase letters (`"A"`, `"B"`, …).
    /// Use [`VennDiagram::with_names`] to override them.
    ///
    /// # Errors
    ///
    /// Returns [`DiagramError::UnsupportedSetCount`] if `n == 0` or `n >= 6`.
    pub fn new(n: usize) -> Result<Self, DiagramError> {
        let params = venn_params(n).ok_or(DiagramError::UnsupportedSetCount(n))?;
        let ellipses = params
            .iter()
            .map(|&(h, k, a, b, phi)| Ellipse::new(Point::new(h, k), a, b, phi))
            .collect();
        let names = (0..n).map(|i| default_name(i).to_string()).collect();
        Ok(VennDiagram { ellipses, names })
    }

    /// Override the default set names. Names are matched to ellipses by index.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `names.len() != self.ellipses.len()`.
    pub fn with_names(mut self, names: &[&str]) -> Self {
        debug_assert_eq!(
            names.len(),
            self.ellipses.len(),
            "with_names: expected {} names, got {}",
            self.ellipses.len(),
            names.len()
        );
        self.names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Returns the ellipses making up the Venn diagram, in the order of `names()`.
    pub fn ellipses(&self) -> &[Ellipse] {
        &self.ellipses
    }

    /// Returns the set names, in the same order as `ellipses()`.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Builds a synthetic [`DiagramSpec`] where every non-empty subset of
    /// sets is requested with area `1.0` (matching eulerr's
    /// `fitted.values = rep(1, length(orig))`).
    ///
    /// Useful when callers need to drive code paths that take the spec
    /// alongside the layout (e.g. `Layout::region_polygons`).
    pub fn build_spec(&self) -> DiagramSpec {
        let n = self.ellipses.len();
        let mut builder = DiagramSpecBuilder::new().input_type(InputType::Exclusive);

        // Add singletons first to lock in deterministic set ordering.
        for name in &self.names {
            builder = builder.set(name.as_str(), 1.0);
        }

        // Add every non-singleton subset (size 2..=n).
        let total = 1usize << n;
        for mask in 1..total {
            if mask.count_ones() < 2 {
                continue;
            }
            let subset: Vec<&str> = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| self.names[i].as_str())
                .collect();
            builder = builder.intersection(&subset, 1.0);
        }

        builder
            .build()
            .expect("synthetic Venn spec should always build")
    }

    /// Builds a [`Layout<Ellipse>`] over the synthetic specification from
    /// [`Self::build_spec`].
    ///
    /// The fitted areas in the returned layout are the actual ellipse-region
    /// areas — they are not all `1.0`, because the canonical arrangements do
    /// not produce equal regions.
    pub fn into_layout(self) -> Layout<Ellipse> {
        self.into_layout_and_spec().0
    }

    /// Like [`Self::into_layout`] but also returns the synthetic
    /// [`DiagramSpec`] so callers can pass it to spec-taking APIs like
    /// `Layout::region_polygons` without rebuilding it.
    pub fn into_layout_and_spec(self) -> (Layout<Ellipse>, DiagramSpec) {
        let spec = self.build_spec();
        let set_to_shape: HashMap<String, usize> = self
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let layout = Layout::new(self.ellipses, set_to_shape, &spec, 0, LossType::SumSquared);
        (layout, spec)
    }
}

fn default_name(i: usize) -> &'static str {
    const NAMES: [&str; 5] = ["A", "B", "C", "D", "E"];
    NAMES[i]
}

/// Returns the `(h, k, a, b, phi)` quintuples for the canonical n-set Venn
/// arrangement, or `None` if `n` is unsupported (n == 0 or n >= 6).
fn venn_params(n: usize) -> Option<&'static [EllipseParams]> {
    match n {
        1 => Some(&N1),
        2 => Some(&N2),
        3 => Some(&N3),
        4 => Some(&N4),
        5 => Some(&N5),
        _ => None,
    }
}

const N1: [EllipseParams; 1] = [(0.0, 0.0, 1.0, 1.0, 0.0)];

const N2: [EllipseParams; 2] = [(-0.5, 0.0, 1.0, 1.0, 1.0), (0.5, 0.0, 1.0, 1.0, 1.0)];

const N3: [EllipseParams; 3] = [
    (-0.42, -0.36, 1.05, 1.05, 3.76),
    (0.42, -0.36, 1.05, 1.05, 3.76),
    (0.00, 0.36, 1.05, 1.05, 3.76),
];

const N4: [EllipseParams; 4] = [
    (-0.8, 0.0, 1.2, 2.0, PI / 4.0),
    (0.8, 0.0, 1.2, 2.0, -PI / 4.0),
    (0.0, 1.0, 1.2, 2.0, PI / 4.0),
    (0.0, 1.0, 1.2, 2.0, -PI / 4.0),
];

const N5: [EllipseParams; 5] = [
    (0.176, 0.096, 1.0, 0.6, 0.000),
    (-0.037, 0.197, 1.0, 0.6, 1.257),
    (-0.198, 0.026, 1.0, 0.6, 2.513),
    (-0.086, -0.181, 1.0, 0.6, 3.770),
    (0.145, -0.137, 1.0, 0.6, 5.027),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn assert_params(venn: &VennDiagram, expected: &[EllipseParams]) {
        assert_eq!(venn.ellipses().len(), expected.len());
        for (e, &(h, k, a, b, phi)) in venn.ellipses().iter().zip(expected) {
            assert!(
                approx_eq(e.center().x(), h),
                "h: {} vs {}",
                e.center().x(),
                h
            );
            assert!(
                approx_eq(e.center().y(), k),
                "k: {} vs {}",
                e.center().y(),
                k
            );
            assert!(
                approx_eq(e.semi_major(), a),
                "a: {} vs {}",
                e.semi_major(),
                a
            );
            assert!(
                approx_eq(e.semi_minor(), b),
                "b: {} vs {}",
                e.semi_minor(),
                b
            );
            assert!(
                approx_eq(e.rotation(), phi),
                "phi: {} vs {}",
                e.rotation(),
                phi
            );
        }
    }

    #[test]
    fn test_n1_params() {
        let venn = VennDiagram::new(1).unwrap();
        assert_params(&venn, &N1);
    }

    #[test]
    fn test_n2_params() {
        let venn = VennDiagram::new(2).unwrap();
        assert_params(&venn, &N2);
    }

    #[test]
    fn test_n3_params() {
        let venn = VennDiagram::new(3).unwrap();
        assert_params(&venn, &N3);
    }

    #[test]
    fn test_n4_params() {
        let venn = VennDiagram::new(4).unwrap();
        assert_params(&venn, &N4);
    }

    #[test]
    fn test_n5_params() {
        let venn = VennDiagram::new(5).unwrap();
        assert_params(&venn, &N5);
    }

    #[test]
    fn test_default_names() {
        let venn = VennDiagram::new(3).unwrap();
        assert_eq!(venn.names(), &["A", "B", "C"]);

        let venn5 = VennDiagram::new(5).unwrap();
        assert_eq!(venn5.names(), &["A", "B", "C", "D", "E"]);
    }

    #[test]
    fn test_with_names_overrides() {
        let venn = VennDiagram::new(3)
            .unwrap()
            .with_names(&["foo", "bar", "baz"]);
        assert_eq!(venn.names(), &["foo", "bar", "baz"]);
    }

    #[test]
    #[should_panic(expected = "with_names: expected 3 names, got 2")]
    fn test_with_names_length_mismatch_panics() {
        let _ = VennDiagram::new(3).unwrap().with_names(&["foo", "bar"]);
    }

    #[test]
    fn test_zero_sets_unsupported() {
        let err = VennDiagram::new(0).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(0));
    }

    #[test]
    fn test_six_sets_unsupported() {
        let err = VennDiagram::new(6).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(6));
    }

    #[test]
    fn test_seven_sets_unsupported() {
        let err = VennDiagram::new(7).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(7));
    }

    /// For each supported n, the canonical arrangement must produce a true Venn
    /// topology: all 2^n - 1 regions present with non-negligible area.
    /// `Layout::compute_fitted_areas` filters regions with area ≤ 1e-10, so the
    /// fitted map size is the count of visible regions.
    #[test]
    fn test_topology_is_true_venn() {
        for n in 1..=5usize {
            let layout = VennDiagram::new(n).unwrap().into_layout();
            let expected = (1usize << n) - 1;
            let fitted = layout.fitted();
            assert_eq!(
                fitted.len(),
                expected,
                "n={}: expected {} non-empty regions, got {}",
                n,
                expected,
                fitted.len()
            );
            for (combo, &area) in fitted {
                assert!(
                    area > 1e-9,
                    "n={}: region {} has area {} (too small)",
                    n,
                    combo,
                    area
                );
            }
        }
    }

    #[test]
    fn test_into_layout_shape_count() {
        for n in 1..=5usize {
            let layout = VennDiagram::new(n).unwrap().into_layout();
            assert_eq!(layout.shapes().len(), n);
        }
    }
}
