//! Canonical Venn-diagram layouts.
//!
//! [`VennDiagram::new`] returns a hardcoded n-set arrangement where every one
//! of the `2ⁿ − 1` regions is non-empty (i.e. a true Venn diagram). The
//! arrangements per shape live with the shape's [`DiagramShape`] impl
//! ([`canonical_venn_layout`]); see those for the published references.
//!
//! Supported set counts depend on the shape:
//!
//! - [`Ellipse`](crate::geometry::shapes::Ellipse): `n ∈ {1, …, 5}`.
//! - [`Square`](crate::geometry::shapes::Square): `n ∈ {1, 2, 3}` —
//!   axis-aligned squares cannot form a true Venn for `n ≥ 4`.
//!
//! `n = 0` and any `n` outside a shape's supported range return
//! [`DiagramError::UnsupportedSetCount`].
//!
//! [`canonical_venn_layout`]: crate::geometry::traits::DiagramShape::canonical_venn_layout

use std::collections::HashMap;

use crate::error::DiagramError;
use crate::fitter::Layout;
use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::geometry::traits::DiagramShape;
use crate::loss::LossType;
use crate::spec::{DiagramSpec, DiagramSpecBuilder, InputType};

/// A canonical Venn-diagram layout, generic over the shape type `S`.
///
/// Use [`VennDiagram::new`] to construct one from a set count `n`, and
/// [`VennDiagram::into_layout`] to obtain a [`Layout<S>`] suitable for the
/// same plotting/inspection pipeline as a fitted layout.
///
/// # Example
///
/// ```
/// use eunoia::VennDiagram;
/// use eunoia::geometry::shapes::Ellipse;
///
/// let venn = VennDiagram::<Ellipse>::new(3).unwrap();
/// assert_eq!(venn.shapes().len(), 3);
/// assert_eq!(venn.names(), &["A", "B", "C"]);
/// ```
#[derive(Debug, Clone)]
pub struct VennDiagram<S: DiagramShape + Copy> {
    shapes: Vec<S>,
    names: Vec<String>,
    complement: Option<f64>,
}

/// Padding added around the shapes' bounding box when a Venn diagram carries
/// a complement: the container is purely a visual frame (Venn is topological,
/// not area-proportional), so the size is chosen for legibility, not to
/// satisfy any area constraint.
const VENN_CONTAINER_PADDING_FRAC: f64 = 0.15;

impl<S: DiagramShape + Copy + 'static> VennDiagram<S> {
    /// Constructs the canonical Venn arrangement for `n` sets in shape `S`.
    ///
    /// Default set names are the first `n` uppercase letters (`"A"`, `"B"`, …).
    /// Use [`VennDiagram::with_names`] to override them.
    ///
    /// # Errors
    ///
    /// Returns [`DiagramError::UnsupportedSetCount`] if `n == 0` or if the
    /// shape `S` has no canonical Venn arrangement at this `n` (see
    /// [`DiagramShape::canonical_venn_layout`]).
    pub fn new(n: usize) -> Result<Self, DiagramError> {
        let shapes = S::canonical_venn_layout(n).ok_or(DiagramError::UnsupportedSetCount(n))?;
        let names = (0..n).map(|i| default_name(i).to_string()).collect();
        Ok(VennDiagram {
            shapes,
            names,
            complement: None,
        })
    }

    /// Attach a complement (universe-outside-every-set) value to the diagram.
    ///
    /// Venn diagrams are *topological*, not area-proportional, so the
    /// complement value does **not** drive an area-proportional optimisation
    /// — the canonical shape layout is fixed. The complement is carried
    /// through to the resulting [`Layout`] in two ways:
    ///
    /// - The synthetic [`DiagramSpec`] from [`Self::build_spec`] gets the
    ///   value via [`DiagramSpecBuilder::complement`], so renderers can label
    ///   the complement region.
    /// - [`Self::into_layout`] / [`Self::into_layout_and_spec`] build a
    ///   visual container rectangle as the shapes' bounding box plus a
    ///   small padding (no optimisation involved).
    ///
    /// # Errors
    ///
    /// Returns [`DiagramError::InvalidValue`] if `value` is negative.
    pub fn complement(mut self, value: f64) -> Result<Self, DiagramError> {
        if value < 0.0 {
            return Err(DiagramError::InvalidValue {
                combination: "<complement>".to_string(),
                value,
            });
        }
        self.complement = Some(value);
        Ok(self)
    }

    /// Override the default set names. Names are matched to shapes by index.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `names.len() != self.shapes.len()`.
    pub fn with_names(mut self, names: &[&str]) -> Self {
        debug_assert_eq!(
            names.len(),
            self.shapes.len(),
            "with_names: expected {} names, got {}",
            self.shapes.len(),
            names.len()
        );
        self.names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Returns the shapes making up the Venn diagram, in the order of `names()`.
    pub fn shapes(&self) -> &[S] {
        &self.shapes
    }

    /// Returns the set names, in the same order as `shapes()`.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Builds a synthetic [`DiagramSpec`] where every non-empty subset of
    /// sets is requested with area `1.0` (matching eulerr's
    /// `fitted.values = rep(1, length(orig))`).
    ///
    /// If a complement was attached via [`Self::complement`], it is forwarded
    /// to the spec so renderers can label the universe region.
    ///
    /// Useful when callers need to drive code paths that take the spec
    /// alongside the layout (e.g. `Layout::region_polygons`).
    pub fn build_spec(&self) -> DiagramSpec {
        let n = self.shapes.len();
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

        if let Some(c) = self.complement {
            builder = builder.complement(c);
        }

        builder
            .build()
            .expect("synthetic Venn spec should always build")
    }

    /// Bounding rectangle of the canonical shape arrangement plus a small
    /// uniform padding. Used as the visual container when [`Self::complement`]
    /// is set: Venn is topological, so this is purely a frame, not an
    /// area-proportional fit.
    fn padded_bounding_container(&self) -> Rectangle {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for shape in &self.shapes {
            let (bx_min, bx_max, by_min, by_max) = shape.bounding_box().bounds();
            x_min = x_min.min(bx_min);
            x_max = x_max.max(bx_max);
            y_min = y_min.min(by_min);
            y_max = y_max.max(by_max);
        }
        let width = x_max - x_min;
        let height = y_max - y_min;
        let pad = width.max(height) * VENN_CONTAINER_PADDING_FRAC;
        let cx = (x_min + x_max) * 0.5;
        let cy = (y_min + y_max) * 0.5;
        Rectangle::new(Point::new(cx, cy), width + 2.0 * pad, height + 2.0 * pad)
    }

    /// Builds a [`Layout<S>`] over the synthetic specification from
    /// [`Self::build_spec`].
    ///
    /// The fitted areas in the returned layout are the actual shape-region
    /// areas — they are not all `1.0`, because the canonical arrangements do
    /// not produce equal regions.
    pub fn into_layout(self) -> Layout<S> {
        self.into_layout_and_spec().0
    }

    /// Like [`Self::into_layout`] but also returns the synthetic
    /// [`DiagramSpec`] so callers can pass it to spec-taking APIs like
    /// `Layout::region_polygons` without rebuilding it.
    pub fn into_layout_and_spec(self) -> (Layout<S>, DiagramSpec) {
        let spec = self.build_spec();
        let set_to_shape: HashMap<String, usize> = self
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Container is a *visual frame* when complement is set — Venn is
        // topological, so we don't run the fitter; we just pad the shapes'
        // bounding box. The padded box becomes the container so renderers
        // know where to draw the universe outline.
        let container = self.complement.map(|_| self.padded_bounding_container());

        let layout = Layout::new(
            self.shapes,
            set_to_shape,
            &spec,
            0,
            LossType::SumSquared,
            container,
        );
        (layout, spec)
    }
}

fn default_name(i: usize) -> &'static str {
    const NAMES: [&str; 5] = ["A", "B", "C", "D", "E"];
    NAMES[i]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::{Ellipse, Rectangle, Square};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    type EllipseParams = (f64, f64, f64, f64, f64);

    fn assert_ellipse_params(venn: &VennDiagram<Ellipse>, expected: &[EllipseParams]) {
        assert_eq!(venn.shapes().len(), expected.len());
        for (e, &(h, k, a, b, phi)) in venn.shapes().iter().zip(expected) {
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
    fn test_n1_ellipse_params() {
        let venn = VennDiagram::<Ellipse>::new(1).unwrap();
        assert_ellipse_params(&venn, &[(0.0, 0.0, 1.0, 1.0, 0.0)]);
    }

    #[test]
    fn test_n2_ellipse_params() {
        let venn = VennDiagram::<Ellipse>::new(2).unwrap();
        assert_ellipse_params(
            &venn,
            &[(-0.5, 0.0, 1.0, 1.0, 1.0), (0.5, 0.0, 1.0, 1.0, 1.0)],
        );
    }

    #[test]
    fn test_n3_ellipse_params() {
        let venn = VennDiagram::<Ellipse>::new(3).unwrap();
        assert_ellipse_params(
            &venn,
            &[
                (-0.42, -0.36, 1.05, 1.05, 3.76),
                (0.42, -0.36, 1.05, 1.05, 3.76),
                (0.00, 0.36, 1.05, 1.05, 3.76),
            ],
        );
    }

    #[test]
    fn test_n4_ellipse_params() {
        use std::f64::consts::PI;
        let venn = VennDiagram::<Ellipse>::new(4).unwrap();
        assert_ellipse_params(
            &venn,
            &[
                (-0.8, 0.0, 1.2, 2.0, PI / 4.0),
                (0.8, 0.0, 1.2, 2.0, -PI / 4.0),
                (0.0, 1.0, 1.2, 2.0, PI / 4.0),
                (0.0, 1.0, 1.2, 2.0, -PI / 4.0),
            ],
        );
    }

    #[test]
    fn test_n5_ellipse_params() {
        let venn = VennDiagram::<Ellipse>::new(5).unwrap();
        assert_ellipse_params(
            &venn,
            &[
                (0.176, 0.096, 1.0, 0.6, 0.000),
                (-0.037, 0.197, 1.0, 0.6, 1.257),
                (-0.198, 0.026, 1.0, 0.6, 2.513),
                (-0.086, -0.181, 1.0, 0.6, 3.770),
                (0.145, -0.137, 1.0, 0.6, 5.027),
            ],
        );
    }

    #[test]
    fn test_default_names() {
        let venn = VennDiagram::<Ellipse>::new(3).unwrap();
        assert_eq!(venn.names(), &["A", "B", "C"]);

        let venn5 = VennDiagram::<Ellipse>::new(5).unwrap();
        assert_eq!(venn5.names(), &["A", "B", "C", "D", "E"]);
    }

    #[test]
    fn test_with_names_overrides() {
        let venn = VennDiagram::<Ellipse>::new(3)
            .unwrap()
            .with_names(&["foo", "bar", "baz"]);
        assert_eq!(venn.names(), &["foo", "bar", "baz"]);
    }

    #[test]
    #[should_panic(expected = "with_names: expected 3 names, got 2")]
    fn test_with_names_length_mismatch_panics() {
        let _ = VennDiagram::<Ellipse>::new(3)
            .unwrap()
            .with_names(&["foo", "bar"]);
    }

    #[test]
    fn test_complement_round_trips_into_spec() {
        // Setting a complement on a Venn diagram should forward into the
        // synthetic spec, so renderers / spec consumers see it.
        let venn = VennDiagram::<Ellipse>::new(3)
            .unwrap()
            .complement(7.5)
            .unwrap();
        let spec = venn.build_spec();
        assert_eq!(spec.complement(), Some(7.5));
    }

    #[test]
    fn test_complement_negative_rejected() {
        let err = VennDiagram::<Ellipse>::new(2)
            .unwrap()
            .complement(-1.0)
            .unwrap_err();
        assert!(
            matches!(
                err,
                DiagramError::InvalidValue { ref combination, value }
                    if combination == "<complement>" && value < 0.0
            ),
            "expected InvalidValue, got {:?}",
            err,
        );
    }

    #[test]
    fn test_complement_into_layout_produces_visual_container() {
        // Layout should carry a Rectangle container that strictly encloses
        // every shape's bounding box (Venn is topological, container is
        // padded around the canonical arrangement).
        let venn = VennDiagram::<Ellipse>::new(3)
            .unwrap()
            .complement(2.0)
            .unwrap();
        let shapes = venn.shapes().to_vec();
        let (layout, _spec) = venn.into_layout_and_spec();

        let container = layout.container().expect("container present");
        let (cx_min, cx_max, cy_min, cy_max) = (
            container.center().x() - container.width() * 0.5,
            container.center().x() + container.width() * 0.5,
            container.center().y() - container.height() * 0.5,
            container.center().y() + container.height() * 0.5,
        );

        use crate::geometry::traits::BoundingBox;
        for shape in &shapes {
            let (sx_min, sx_max, sy_min, sy_max) = shape.bounding_box().bounds();
            assert!(sx_min > cx_min && sx_max < cx_max);
            assert!(sy_min > cy_min && sy_max < cy_max);
        }
    }

    #[test]
    fn test_no_complement_yields_no_container() {
        let (layout, _) = VennDiagram::<Ellipse>::new(3)
            .unwrap()
            .into_layout_and_spec();
        assert!(layout.container().is_none());
    }

    #[test]
    fn test_zero_sets_unsupported_ellipse() {
        let err = VennDiagram::<Ellipse>::new(0).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(0));
    }

    #[test]
    fn test_six_sets_unsupported_ellipse() {
        let err = VennDiagram::<Ellipse>::new(6).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(6));
    }

    #[test]
    fn test_seven_sets_unsupported_ellipse() {
        let err = VennDiagram::<Ellipse>::new(7).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(7));
    }

    /// For each supported n, the canonical ellipse arrangement must produce a
    /// true Venn topology: all 2^n - 1 regions present with non-negligible
    /// area. `Layout::compute_fitted_areas` filters regions with area ≤ 1e-10,
    /// so the fitted map size is the count of visible regions.
    #[test]
    fn test_topology_is_true_venn_ellipse() {
        for n in 1..=5usize {
            let layout = VennDiagram::<Ellipse>::new(n).unwrap().into_layout();
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
    fn test_into_layout_shape_count_ellipse() {
        for n in 1..=5usize {
            let layout = VennDiagram::<Ellipse>::new(n).unwrap().into_layout();
            assert_eq!(layout.shapes().len(), n);
        }
    }

    /// Square Venn arrangements only exist for n ∈ {1, 2, 3}; the topology
    /// must still satisfy the 2ⁿ − 1 region invariant.
    #[test]
    fn test_topology_is_true_venn_square() {
        for n in 1..=3usize {
            let layout = VennDiagram::<Square>::new(n).unwrap().into_layout();
            let expected = (1usize << n) - 1;
            let fitted = layout.fitted();
            assert_eq!(
                fitted.len(),
                expected,
                "square n={}: expected {} non-empty regions, got {}",
                n,
                expected,
                fitted.len()
            );
            for (combo, &area) in fitted {
                assert!(
                    area > 1e-9,
                    "square n={}: region {} has area {} (too small)",
                    n,
                    combo,
                    area
                );
            }
        }
    }

    #[test]
    fn test_square_default_names() {
        let venn = VennDiagram::<Square>::new(3).unwrap();
        assert_eq!(venn.names(), &["A", "B", "C"]);
    }

    #[test]
    fn test_square_zero_sets_unsupported() {
        let err = VennDiagram::<Square>::new(0).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(0));
    }

    #[test]
    fn test_square_four_sets_unsupported() {
        let err = VennDiagram::<Square>::new(4).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(4));
    }

    #[test]
    fn test_square_five_sets_unsupported() {
        let err = VennDiagram::<Square>::new(5).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(5));
    }

    /// Rectangle Venn arrangements only exist for n ∈ {1, 2, 3} (same cap
    /// as Square — the obstruction is the axis-aligned topology, not the
    /// width/height freedom). The 2ⁿ − 1 region invariant must still hold.
    #[test]
    fn test_topology_is_true_venn_rectangle() {
        for n in 1..=3usize {
            let layout = VennDiagram::<Rectangle>::new(n).unwrap().into_layout();
            let expected = (1usize << n) - 1;
            let fitted = layout.fitted();
            assert_eq!(
                fitted.len(),
                expected,
                "rectangle n={}: expected {} non-empty regions, got {}",
                n,
                expected,
                fitted.len()
            );
            for (combo, &area) in fitted {
                assert!(
                    area > 1e-9,
                    "rectangle n={}: region {} has area {} (too small)",
                    n,
                    combo,
                    area
                );
            }
        }
    }

    #[test]
    fn test_rectangle_default_names() {
        let venn = VennDiagram::<Rectangle>::new(3).unwrap();
        assert_eq!(venn.names(), &["A", "B", "C"]);
    }

    #[test]
    fn test_rectangle_zero_sets_unsupported() {
        let err = VennDiagram::<Rectangle>::new(0).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(0));
    }

    #[test]
    fn test_rectangle_four_sets_unsupported() {
        let err = VennDiagram::<Rectangle>::new(4).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(4));
    }

    #[test]
    fn test_rectangle_five_sets_unsupported() {
        let err = VennDiagram::<Rectangle>::new(5).unwrap_err();
        assert_eq!(err, DiagramError::UnsupportedSetCount(5));
    }
}
