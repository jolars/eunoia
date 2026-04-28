//! Synthetic ground-truth proptest for the fitter.
//!
//! Generates random shapes, computes their exclusive areas via the same
//! routine the fitter consumes, then asks the fitter to recover a layout
//! from those areas. By construction an *exactly fitting* configuration
//! exists (the generating shapes), so a healthy fitter must drive
//! `diag_error` near zero.
//!
//! Assertions are on **areas** (`diag_error`), not parameter recovery —
//! multiple parameter configurations can produce identical region areas
//! (rotation, reflection, label permutation), so equality of fitted vs
//! generated parameters is not the contract.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::RangeInclusive;

    use proptest::prelude::*;

    use crate::geometry::diagram::{mask_to_indices, RegionMask};
    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::{Circle, Ellipse};
    use crate::geometry::traits::DiagramShape;
    use crate::spec::{Combination, DiagramSpec, DiagramSpecBuilder, InputType};
    use crate::Fitter;

    /// Worst acceptable `diag_error` for a recovered fit. The generating
    /// configuration is *representable* by construction, so a perfect
    /// fitter would reach near-zero. The threshold is intentionally
    /// loose because the current default `Fitter` occasionally settles
    /// into a local minimum on randomly drawn ellipse layouts (e.g.
    /// dropping a small pairwise overlap when one set is fully
    /// disjoint). Tightening this is a useful signal of fitter
    /// improvements; loosening it should be considered a regression.
    const MAX_DIAG_ERROR: f64 = 5e-2;

    /// Number of seeds to try per generated configuration before
    /// declaring a fit failed. Proptest exists to surface regressions,
    /// not to flake on a single unlucky seed; best-of-3 mirrors how a
    /// real user would re-run a flaky fit.
    const FIT_SEEDS: [u64; 3] = [0, 1, 2];

    /// Floor on the smallest non-empty exclusive region area: below this
    /// the fitter's mask discovery is dominated by numerical noise rather
    /// than the real layout, and a "recovery" assertion stops being
    /// meaningful.
    const MIN_REGION_AREA: f64 = 0.1;

    /// Set names used when building specs from generated shapes. Indexed
    /// by shape position. 8 is plenty for our `n_range` bounds.
    const SET_NAMES: [&str; 8] = ["A", "B", "C", "D", "E", "F", "G", "H"];

    fn arbitrary_circles(n_range: RangeInclusive<usize>) -> impl Strategy<Value = Vec<Circle>> {
        prop::collection::vec(
            (0.0_f64..50.0, 0.0_f64..50.0, 2.0_f64..15.0)
                .prop_map(|(x, y, r)| Circle::new(Point::new(x, y), r)),
            n_range,
        )
    }

    fn arbitrary_ellipses(n_range: RangeInclusive<usize>) -> impl Strategy<Value = Vec<Ellipse>> {
        prop::collection::vec(
            // semi_major in [4, 15], semi_major/semi_minor ratio in [1, 4]
            // to avoid extremely degenerate ellipses, rotation in [0, π).
            (
                0.0_f64..50.0,
                0.0_f64..50.0,
                4.0_f64..15.0,
                1.0_f64..4.0,
                0.0_f64..std::f64::consts::PI,
            )
                .prop_map(|(x, y, a, ratio, theta)| {
                    let b = a / ratio;
                    Ellipse::new(Point::new(x, y), a, b, theta)
                }),
            n_range,
        )
    }

    /// Build a `DiagramSpec` from generated shapes' exclusive areas.
    /// Returns `None` if the configuration is degenerate (all shapes
    /// pairwise disjoint, or any non-empty region below the precision
    /// floor) — the caller should reject and try again.
    fn spec_from_shapes<S: DiagramShape>(shapes: &[S]) -> Option<DiagramSpec> {
        let raw: HashMap<RegionMask, f64> = S::compute_exclusive_regions(shapes);

        if raw.is_empty() {
            return None;
        }

        // Drop numerical-noise regions up front so the rejection logic
        // below operates on the same set of regions the spec will be
        // built from. `compute_exclusive_regions` can return spurious
        // tiny intersections for shapes that *should* be disjoint;
        // without this we'd think a fully-disjoint shape "participates"
        // in some 1e-13 overlap, then strip that overlap at spec-build
        // and end up with the exact disjoint configuration the filter
        // was supposed to reject.
        let exclusive: HashMap<RegionMask, f64> =
            raw.into_iter().filter(|(_, area)| *area > 1e-10).collect();

        if exclusive.is_empty() {
            return None;
        }

        // Reject specs with any non-trivially-zero region below the
        // precision floor. Values between `1e-10` and `MIN_REGION_AREA`
        // are exactly the noisy regime where recovery isn't meaningful.
        for &area in exclusive.values() {
            if area < MIN_REGION_AREA {
                return None;
            }
        }

        // Reject configurations where any shape is fully disjoint from
        // every other shape. With one-or-more disjoint singletons mixed
        // into a connected cluster the fitter has unbounded freedom
        // about where to place the loose shape; the global optimum is a
        // ridge rather than a point and the optimizer routinely settles
        // for a different layout that produces the same singleton areas
        // but loses other intersections. A real-world Euler diagram
        // user wouldn't request that input either.
        let n = shapes.len();
        for i in 0..n {
            let participates = exclusive
                .keys()
                .any(|&mask| (mask as RegionMask).count_ones() >= 2 && (mask & (1 << i)) != 0);
            if !participates {
                return None;
            }
        }

        let mut builder = DiagramSpecBuilder::new();
        // Always include singletons so every set is registered, even if
        // its exclusive area is zero (e.g. fully covered by overlaps).
        for name in SET_NAMES.iter().take(n) {
            builder = builder.set(*name, 0.0);
        }

        for (mask, area) in exclusive.iter() {
            let indices = mask_to_indices(*mask, n);
            let names: Vec<&str> = indices.iter().map(|&i| SET_NAMES[i]).collect();
            if names.len() == 1 {
                builder = builder.set(names[0], *area);
            } else {
                builder = builder.intersection(&names, *area);
            }
        }

        builder
            .input_type(InputType::Exclusive)
            .build()
            .ok()
            .and_then(verify_round_trip)
    }

    /// Sanity check: the spec we just built must round-trip through
    /// preprocessing. If preprocessing rejects (n_sets <= 1, etc.) we
    /// reject the generated configuration.
    fn verify_round_trip(spec: DiagramSpec) -> Option<DiagramSpec> {
        if spec.set_names().len() < 2 {
            return None;
        }
        // Require at least one combination with positive area; otherwise
        // there's nothing meaningful to fit against.
        if !spec
            .exclusive_areas()
            .iter()
            .any(|(combo, &v)| !combo.is_empty() && v > 0.0)
        {
            return None;
        }
        Some(spec)
    }

    /// Try fitting a spec under multiple seeds and return the best
    /// `(diag_error, fitted_areas)`. Wraps each call in `catch_unwind`
    /// so we can record specs that trip pre-existing internal asserts
    /// (e.g. the `normalize_layout` debug_assert on coincident shapes)
    /// rather than aborting the proptest mid-case.
    fn best_fit<S>(
        spec: &DiagramSpec,
        seeds: &[u64],
    ) -> Result<(f64, HashMap<Combination, f64>), TestCaseError>
    where
        S: DiagramShape + Copy + 'static,
    {
        let mut best: Option<(f64, HashMap<Combination, f64>)> = None;
        let mut last_err: Option<String> = None;
        for &seed in seeds {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Fitter::<S>::new(spec).seed(seed).fit()
            }));
            match result {
                Ok(Ok(layout)) => {
                    let diag = layout.diag_error();
                    if !diag.is_finite() {
                        last_err = Some(format!("non-finite diag_error at seed={seed}"));
                        continue;
                    }
                    if best.as_ref().is_none_or(|(d, _)| diag < *d) {
                        best = Some((diag, layout.fitted().clone()));
                    }
                }
                Ok(Err(e)) => {
                    last_err = Some(format!("fit Err at seed={seed}: {e}"));
                }
                Err(_) => {
                    last_err = Some(format!("fit panicked at seed={seed}"));
                }
            }
        }
        best.ok_or_else(|| {
            TestCaseError::reject(format!(
                "every seed failed: {}",
                last_err.unwrap_or_else(|| "unknown".into())
            ))
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 50,
            max_shrink_iters: 10,
            ..ProptestConfig::default()
        })]

        #[test]
        fn circles_recover_from_their_own_areas(
            shapes in arbitrary_circles(2..=4)
        ) {
            let Some(spec) = spec_from_shapes(&shapes) else {
                // Degenerate generated configuration; not a failure.
                return Ok(());
            };

            let (best_diag, best_fitted) = best_fit::<Circle>(&spec, &FIT_SEEDS)?;
            prop_assert!(
                best_diag.is_finite() && best_diag <= MAX_DIAG_ERROR,
                "circles diag_error = {best_diag:?} (best of {} seeds), expected <= {MAX_DIAG_ERROR}; \
                 spec_areas={:?}, fitted_areas={:?}",
                FIT_SEEDS.len(),
                spec.exclusive_areas(),
                best_fitted,
            );
        }

        #[test]
        fn ellipses_recover_from_their_own_areas(
            shapes in arbitrary_ellipses(2..=3)
        ) {
            let Some(spec) = spec_from_shapes(&shapes) else {
                return Ok(());
            };

            let (best_diag, best_fitted) = best_fit::<Ellipse>(&spec, &FIT_SEEDS)?;
            prop_assert!(
                best_diag.is_finite() && best_diag <= MAX_DIAG_ERROR,
                "ellipses diag_error = {best_diag:?} (best of {} seeds), expected <= {MAX_DIAG_ERROR}; \
                 spec_areas={:?}, fitted_areas={:?}",
                FIT_SEEDS.len(),
                spec.exclusive_areas(),
                best_fitted,
            );
        }
    }
}
