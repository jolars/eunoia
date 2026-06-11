//! Walkthrough of the eunoia plotting API from a binding author's
//! perspective (e.g. eulerr's R wrapper, or a Python/Julia consumer).
//!
//! This file is structured as a single end-to-end test that exercises every
//! seam the recent friction-feedback addressed:
//!
//! 1. **Validated shape construction** — `try_new` over `new` for shapes
//!    coming from untrusted user input.
//! 2. **One-shot bundled output** — `Layout::plot_data` for renderers.
//! 3. **String ↔ `Combination` round-tripping** — `Combination::from_str`
//!    so bindings don't have to split-and-trim themselves.
//! 4. **Deterministic iteration** — `iter_in_input_order(spec.set_names())`
//!    so bindings emit regions in a stable order.
//! 5. **Composable clip → pieces** — `polygon_clip` chained into the
//!    now-public `classify_into_pieces` for one-shape output without
//!    going through `decompose_regions`.
//!
//! Run with `cargo test --test binding_walkthrough`.

use eunoia::geometry::primitives::Point;
use eunoia::geometry::shapes::{Circle, Ellipse, Square};
use eunoia::geometry::traits::Polygonize;
use eunoia::plotting::{ClipOperation, PlotData, PlotOptions, classify_into_pieces, polygon_clip};
use eunoia::spec::{Combination, DiagramSpecBuilder, InputType};
use eunoia::{DiagramError, Fitter};

#[test]
fn binding_author_walkthrough() {
    // ---------------------------------------------------------------------
    // 1. Validated construction at the FFI boundary.
    //
    // The non-fallible `new` constructors panic on `<= 0` parameters now,
    // so binding authors should reach for `try_new` whenever the values
    // come from outside Rust.
    // ---------------------------------------------------------------------
    assert!(Circle::try_new(Point::new(0.0, 0.0), 1.0).is_ok());
    assert!(Ellipse::try_new(Point::new(0.0, 0.0), 4.0, 3.0, 0.0).is_ok());
    assert!(Square::try_new(Point::new(0.0, 0.0), 2.0).is_ok());

    // The error carries enough structured context (shape name, parameter
    // name, offending value) for bindings to surface a meaningful message
    // to the host language without string-matching.
    let err = Ellipse::try_new(Point::new(0.0, 0.0), 4.0, -1.0, 0.0).unwrap_err();
    match err {
        DiagramError::InvalidShapeParameter {
            shape,
            param,
            value,
        } => {
            assert_eq!(shape, "Ellipse");
            assert_eq!(param, "semi_minor");
            assert!(value < 0.0);
        }
        other => panic!("expected InvalidShapeParameter, got {other:?}"),
    }

    // ---------------------------------------------------------------------
    // 2. Build a spec and fit a layout.
    //
    // The spec is shape-agnostic; the binding picks the shape type when
    // it constructs the `Fitter`. Set names are written down here in the
    // order the host language sent them — that order is the canonical
    // input order we'll use later for deterministic iteration.
    // ---------------------------------------------------------------------
    let spec = DiagramSpecBuilder::new()
        .set("A", 10.0)
        .set("B", 8.0)
        .set("C", 4.0)
        .intersection(&["A", "B"], 3.0)
        .intersection(&["A", "C"], 2.0)
        .intersection(&["B", "C"], 2.0)
        .intersection(&["A", "B", "C"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

    // ---------------------------------------------------------------------
    // 3. One-shot plot data for the renderer.
    //
    // `PlotData` bundles everything a renderer needs: per-region pieces
    // (with outer + holes), per-region label anchors, per-region areas,
    // per-set label anchors (with the eulerr fallback chain), and per-set
    // shape outlines. Region-keyed maps use canonical `"A&B&C"` strings so
    // the binding can serialise them through JSON without going through
    // the `Combination` type.
    // ---------------------------------------------------------------------
    let plot = layout.plot_data(&spec, PlotOptions::default());

    // Region anchors and region areas share the same string keys.
    for combo_str in plot.region_anchors.keys() {
        assert!(plot.region_areas.contains_key(combo_str));
    }
    // One outline + one set-label anchor per set, keyed by set name.
    for name in spec.set_names() {
        assert!(plot.shape_outlines.contains_key(name));
        assert!(plot.set_anchors.contains_key(name));
    }

    // ---------------------------------------------------------------------
    // 4. Round-tripping `Combination` through strings.
    //
    // `PlotData` keys are strings; `RegionPolygons` keys are
    // `Combination`s. Bindings that need to bounce between the two used
    // to hand-roll a split-and-trim. Now `Combination` implements
    // `FromStr` (with `Err = Infallible`) so the parse is just `parse()`.
    // ---------------------------------------------------------------------
    for combo_str in plot.region_anchors.keys() {
        let combo: Combination = combo_str.parse().unwrap();
        // The pieces are accessible from either the typed key or the
        // string helper on `PlotData`.
        let pieces_via_combo = plot.regions.get(&combo).expect("region must exist");
        let pieces_via_str = plot.pieces_for(combo_str).expect("region must exist");
        assert_eq!(pieces_via_combo.len(), pieces_via_str.len());

        // `Display` is the inverse of `FromStr` for canonical inputs.
        assert_eq!(combo.to_string(), combo_str.as_str());
    }

    // Whitespace in user-supplied combination strings is tolerated.
    let combo: Combination = " A & B ".parse().unwrap();
    assert_eq!(combo, Combination::new(&["A", "B"]));

    // ---------------------------------------------------------------------
    // 5. Deterministic iteration in input order.
    //
    // `iter_in_input_order(spec.set_names())` returns regions ordered by
    // the position of their member sets in the spec — singletons before
    // pairs before triples — so a binding emitting a list of regions
    // (e.g. for a legend or a CSV export) gets stable output without
    // sorting itself.
    // ---------------------------------------------------------------------
    let order: Vec<String> = plot
        .regions
        .iter_in_input_order(spec.set_names())
        .map(|(combo, _)| combo.to_string())
        .collect();

    // Singletons in input order, then pairs in canonical-index order.
    let expected_order = ["A", "B", "C", "A&B", "A&C", "B&C", "A&B&C"];
    let mut expected_present: Vec<&str> = expected_order
        .iter()
        .copied()
        .filter(|s| plot.region_anchors.contains_key(*s))
        .collect();
    expected_present.retain(|s| order.iter().any(|o| o == s));
    assert_eq!(order, expected_present);

    // For ad-hoc renderers that don't have the spec at hand, `iter_sorted`
    // gives the same canonical order without needing `set_names`.
    let _canonical: Vec<&Combination> =
        plot.regions.iter_sorted().map(|(combo, _)| combo).collect();

    // ---------------------------------------------------------------------
    // 6. Composable polygon ops: `polygon_clip` → `classify_into_pieces`.
    //
    // When a binding wants to do its own boolean stack (e.g. computing a
    // background mask, or building a custom region the spec doesn't
    // describe), it can chain `polygon_clip` calls itself and then run
    // the result through the now-public `classify_into_pieces` to get
    // the same outer + holes contract that `decompose_regions` produces.
    // ---------------------------------------------------------------------
    let outer_outline = plot.shape_outlines.get("A").cloned().unwrap();
    let inner_outline = plot.shape_outlines.get("B").cloned().unwrap();

    // Take A minus B's outline. `polygon_clip` returns a flat ring list;
    // `classify_into_pieces` resolves it into oriented outer + holes.
    let raw_rings = polygon_clip(&outer_outline, &inner_outline, ClipOperation::Difference);
    let pieces = classify_into_pieces(raw_rings);

    // Every retained piece has positive net area, just like
    // `decompose_regions` output.
    for piece in &pieces {
        assert!(piece.area() > 0.0);
        // Holes are CW (signed area < 0); outer is CCW.
        // Renderers can fill with `fill-rule: nonzero` directly.
        for hole in &piece.holes {
            assert!(hole.vertices().len() >= 3);
        }
    }

    // ---------------------------------------------------------------------
    // 7. Hand-rolled outline at higher resolution (no clipping needed).
    //
    // `Polygonize` is still available for bindings that want a cleaner
    // analytical stroke than the polygonised outlines bundled in
    // `PlotData::shape_outlines` (which are at `n_vertices` resolution).
    //
    // Note: `shape_outlines` rings are stored in **open polyline** form;
    // renderers that auto-close (SVG `<polygon>`, Canvas `closePath()`,
    // R `polygon()`) draw the closing segment for free. Renderers like
    // `grid::polylineGrob` that draw the polyline literally must append
    // the first vertex to close the ring.
    // ---------------------------------------------------------------------
    let circle = layout.shape_for_set("A").unwrap();
    let smooth = circle.polygonize(512);
    assert_eq!(smooth.vertices().len(), 512);

    // ---------------------------------------------------------------------
    // 8. Replotting from stored params without a `Layout`.
    //
    // A common FFI workflow: fit once, hand the fitted shape parameters
    // back to the host language, then later receive them again and re-
    // polygonise at a different resolution without re-fitting.
    // `PlotData::from_shapes` is the entry point for that — same output
    // as `Layout::plot_data`, but you only need the shapes plus the
    // original spec.
    // ---------------------------------------------------------------------
    let stored_shapes: Vec<Circle> = spec
        .set_names()
        .iter()
        .map(|n| *layout.shape_for_set(n).unwrap())
        .collect();
    let replot = PlotData::from_shapes(
        &stored_shapes,
        &spec,
        layout.container(),
        PlotOptions::default().n_vertices(32),
    );
    // Same regions appear at the lower resolution.
    assert_eq!(replot.regions.len(), plot.regions.len());

    // ---------------------------------------------------------------------
    // 9. `iter_in_input_order` accepts any string-like slice.
    //
    // For ad-hoc renderers that don't carry a `&[String]` around,
    // `iter_in_input_order` accepts `&[impl AsRef<str>]` — including
    // `&[&str]` literals and `&[String]` from a spec.
    // ---------------------------------------------------------------------
    let custom_names = ["A", "B", "C"];
    let _ad_hoc_order: Vec<_> = plot
        .regions
        .iter_in_input_order(&custom_names)
        .map(|(combo, _)| combo.to_string())
        .collect();

    // ---------------------------------------------------------------------
    // 10. FFI-safety lint pattern (informational, not exercised here).
    //
    // The `eunoia-wasm` crate ships a `clippy.toml` that puts
    // `Circle::new`, `Ellipse::new`, and `Square::new` on the
    // `disallowed-methods` list with `#![deny(clippy::disallowed_methods)]`
    // in `lib.rs`. Every direct shape construction inside that crate has
    // to use `try_new`, surfacing the validation error to JS instead of
    // panicking across the WASM boundary. Downstream binding crates
    // (R, Python, Julia) can copy `crates/eunoia-wasm/clippy.toml`
    // verbatim into their own crate root to get the same machine-enforced
    // safety net.
    // ---------------------------------------------------------------------
}
