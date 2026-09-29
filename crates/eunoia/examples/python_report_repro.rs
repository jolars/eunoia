//! Isolates core behavior discussed in the eunoia-py 0.6.0 report.
//!
//! Run: `cargo run -p eunoia --release --example python_report_repro`
//!
//! The packing cases use synthetic rectangles measured in layout units, avoiding
//! Python, fonts, and viewport changes. The fit uses exclusive counts derived
//! from the membership data and seed 0 at https://github.com/jolars/eunoia/issues/133.
//! Output describes current behavior rather
//! than asserting that these shortcomings must persist. See the report triage
//! in the repository's TODO.md for ownership and proposed changes.

use std::collections::HashMap;

use eunoia::geometry::primitives::Point;
use eunoia::geometry::shapes::{Circle, Ellipse, Polygon, Rectangle, RotatedRectangle, Square};
use eunoia::geometry::traits::{DiagramShape, Polygonize};
use eunoia::plotting::{
    GlyphBoxOptions, GlyphOptions, PlacementStrategy, RegionPolygons, TetherSource,
    classify_into_pieces, place_glyph_boxes, place_glyphs, place_labels,
};
use eunoia::spec::DiagramSpec;
use eunoia::{Combination, DiagramSpecBuilder, Fitter, InputType};

fn square_region(side: f64) -> RegionPolygons {
    let ring = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(side, 0.0),
        Point::new(side, side),
        Point::new(0.0, side),
    ]);
    RegionPolygons::from_map(HashMap::from([(
        Combination::new(&["A"]),
        classify_into_pieces(vec![ring]),
    )]))
}

fn overlaps(a: &Rectangle, b: &Rectangle) -> bool {
    (a.center().x() - b.center().x()).abs() < (a.width() + b.width()) / 2.0
        && (a.center().y() - b.center().y()).abs() < (a.height() + b.height()) / 2.0
}

fn obstacles() {
    println!("Item 1: label obstacles are best-effort");
    let regions = square_region(2.0);
    // A label covering the region makes honoring it incompatible with placing
    // any members. This isolates the final packer's obstacle fallback.
    let label = Rectangle::new(Point::new(1.0, 1.0), 2.0, 2.0);
    let sizes = HashMap::from([("A".into(), vec![(0.5, 0.2); 3])]);
    let placed = place_glyph_boxes(
        &regions,
        &sizes,
        &GlyphBoxOptions::default().obstacles([label]),
    );
    let boxes = &placed.boxes["A"];
    println!(
        "  boxes: scale={:.3}, placed={}, overlapping_label={}, unplaced={}",
        placed.scale,
        boxes.len(),
        boxes.iter().filter(|b| overlaps(b, &label)).count(),
        placed.unplaced.get("A").copied().unwrap_or(0),
    );

    let dots = place_glyphs(
        &regions,
        &HashMap::from([("A".into(), 3)]),
        &GlyphOptions::default().obstacles([label]),
    );
    println!(
        "  dots: placed={} beneath the same label, unplaced={}",
        dots.positions.get("A").map_or(0, Vec::len),
        dots.unplaced.get("A").copied().unwrap_or(0),
    );

    // An over-wide label must go outside, exposing the supported tether modes.
    let label_sizes = HashMap::from([("A".into(), (3.0, 0.2))]);
    for source in [TetherSource::Poi, TetherSource::Boundary] {
        let labels = place_labels(
            &regions,
            &label_sizes,
            None,
            &PlacementStrategy::default().tether(source),
        );
        let tether = labels["A"].tether.expect("label must be exterior");
        println!(
            "  {source:?} tether: ({:.3}, {:.3})",
            tether.x(),
            tether.y(),
        );
    }
}

fn rows_and_scale() {
    println!("Items 2/4: fewest rows and shrink-only automatic scale");
    let sizes = HashMap::from([("A".into(), vec![(2.0, 0.5); 12])]);
    for side in [20.0, 40.0] {
        let placed = place_glyph_boxes(&square_region(side), &sizes, &GlyphBoxOptions::default());
        let boxes = &placed.boxes["A"];
        let mut ys: Vec<_> = boxes.iter().map(|b| b.center().y()).collect();
        ys.sort_by(f64::total_cmp);
        ys.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let bottom = boxes
            .iter()
            .map(|b| b.center().y() - b.height() / 2.0)
            .fold(f64::INFINITY, f64::min);
        let top = boxes
            .iter()
            .map(|b| b.center().y() + b.height() / 2.0)
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {side}x{side}: scale={:.3}, placed={}, rows={}, occupied_height={:.3}",
            placed.scale,
            boxes.len(),
            ys.len(),
            top - bottom,
        );
    }
}

fn prefix_overflow() {
    println!("Item 4: an unplaceable first name drops the entire suffix");
    let regions = square_region(2.0);
    for (name, items) in [
        ("wide first", vec![(7.0, 0.3), (0.4, 0.3), (0.4, 0.3)]),
        ("wide last", vec![(0.4, 0.3), (0.4, 0.3), (7.0, 0.3)]),
    ] {
        let placed = place_glyph_boxes(
            &regions,
            &HashMap::from([("A".into(), items)]),
            &GlyphBoxOptions::default(),
        );
        println!(
            "  {name}: scale={:.3}, placed={}, unplaced={}",
            placed.scale,
            placed.boxes.get("A").map_or(0, Vec::len),
            placed.unplaced.get("A").copied().unwrap_or(0),
        );
    }
}

fn fit_shape<S: DiagramShape + Polygonize + Copy + 'static>(name: &str, spec: &DiagramSpec) {
    let layout = Fitter::<S>::new(spec).seed(0).n_restarts(10).fit().unwrap();
    let triple = Combination::new(&["A", "S", "U"]);
    let regions = layout.region_polygons(spec, 256);
    let polygon_area: f64 = regions
        .get(&triple)
        .into_iter()
        .flatten()
        .map(|p| p.area())
        .sum();
    // Tiny measured names separate a missing polygon from a font-size overflow.
    // The names in the report are CYP3A4 and NR1I2.
    let boxes = place_glyph_boxes(
        &regions,
        &HashMap::from([(triple.to_string(), vec![(0.01, 0.01); 2])]),
        &GlyphBoxOptions::default(),
    );
    let squared_error: f64 = layout.residuals().values().map(|r| r * r).sum();
    println!(
        "  {name}: loss={:.6}, raw_sse={squared_error:.6}, triple_target={:.1}, triple_fitted={:.6}, \
         triple_polygon={polygon_area:.6}, names_placed={}, names_unplaced={}",
        layout.loss(),
        layout.requested()[&triple],
        layout.fitted().get(&triple).copied().unwrap_or(0.0),
        boxes.boxes.get(&triple.to_string()).map_or(0, Vec::len),
        boxes
            .unplaced
            .get(&triple.to_string())
            .copied()
            .unwrap_or(0),
    );
}

fn topology() {
    println!("Item 5: fitted geometry versus member packing (seed=0, restarts=10)");
    // These are exclusive region counts, not total set sizes. A, S, and U stand
    // for Atorvastatin, Simvastatin, and Sunitinib, respectively.
    let spec = DiagramSpecBuilder::new()
        .input_type(InputType::Exclusive)
        .set("A", 35.0)
        .set("S", 32.0)
        .set("U", 35.0)
        .intersection(&["A", "S"], 12.0)
        .intersection(&["A", "U"], 2.0)
        .intersection(&["S", "U"], 0.0)
        .intersection(&["A", "S", "U"], 2.0)
        .build()
        .unwrap();
    fit_shape::<Circle>("circle", &spec);
    fit_shape::<Ellipse>("ellipse", &spec);
    fit_shape::<Square>("square", &spec);
    fit_shape::<Rectangle>("rectangle", &spec);
    fit_shape::<RotatedRectangle>("rotated rectangle", &spec);

    // Absent region keys are intentionally ignored by the low-level API. A
    // renderer must also compare the requested member regions with the geometry;
    // checking `unplaced` alone cannot detect names lost with a missing region.
    let absent = place_glyph_boxes(
        &square_region(2.0),
        &HashMap::from([("A&S&U".into(), vec![(0.01, 0.01); 2])]),
        &GlyphBoxOptions::default(),
    );
    println!(
        "  absent region control: requested=2, placed={}, unplaced={}",
        absent.boxes.values().map(Vec::len).sum::<usize>(),
        absent.unplaced.values().sum::<usize>(),
    );
}

fn main() {
    obstacles();
    rows_and_scale();
    prefix_overflow();
    topology();
}
