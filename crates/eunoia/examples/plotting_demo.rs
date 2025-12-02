//! Demonstrates the polygon clipping and region decomposition features.
//!
//! Run with: cargo run -p eunoia --example plotting_demo --features plotting

#[cfg(feature = "plotting")]
fn main() {
    use eunoia::geometry::shapes::Circle;
    use eunoia::spec::{DiagramSpecBuilder, InputType};
    use eunoia::Fitter;

    println!("Eunoia Plotting Demo\n");
    println!("===================\n");

    // Create a simple three-set Venn diagram
    let spec = DiagramSpecBuilder::new()
        .set("A", 5.0)
        .set("B", 4.0)
        .set("C", 3.0)
        .intersection(&["A", "B"], 1.0)
        .intersection(&["B", "C"], 0.8)
        .intersection(&["A", "C"], 0.6)
        .intersection(&["A", "B", "C"], 0.3)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    println!("Specification:");
    println!("  Sets: A=5.0, B=4.0, C=3.0");
    println!("  A∩B=1.0, B∩C=0.8, A∩C=0.6, A∩B∩C=0.3\n");

    // Fit the diagram
    let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

    println!("Fitted Layout:");
    println!("  Loss: {:.6}", layout.loss());
    println!("  Iterations: {}\n", layout.iterations());

    // Decompose into region polygons
    let regions = layout.region_polygons(&spec, 64);

    println!("Region Decomposition:");
    println!("  Total regions: {}\n", regions.len());

    // Print details for each region
    let mut sorted_regions: Vec<_> = regions.iter().collect();
    sorted_regions.sort_by_key(|(combo, _)| combo.to_string());

    for (combination, polygons) in sorted_regions {
        let total_area: f64 = polygons.iter().map(|p| p.area()).sum();
        let num_pieces = polygons.len();
        let avg_vertices: f64 =
            polygons.iter().map(|p| p.vertices().len()).sum::<usize>() as f64 / num_pieces as f64;

        println!("  Region: {}", combination);
        println!("    Area: {:.3}", total_area);
        println!("    Pieces: {}", num_pieces);
        println!("    Avg vertices per piece: {:.1}", avg_vertices);

        // Show centroid of first piece
        if let Some(first_poly) = polygons.first() {
            let centroid = first_poly.centroid();
            println!(
                "    First piece centroid: ({:.3}, {:.3})",
                centroid.x(),
                centroid.y()
            );
        }
        println!();
    }

    // Verify total area
    let total_area: f64 = regions.areas().values().sum();
    let expected_total: f64 = spec.exclusive_areas().values().sum();

    println!("Summary:");
    println!("  Total polygon area: {:.3}", total_area);
    println!("  Expected total: {:.3}", expected_total);
    println!("  Difference: {:.3}", (total_area - expected_total).abs());
    println!("\nNote: Small differences are due to polygonization approximation.");
}

#[cfg(not(feature = "plotting"))]
fn main() {
    eprintln!("This example requires the 'plotting' feature.");
    eprintln!("Run with: cargo run -p eunoia --example plotting_demo --features plotting");
    std::process::exit(1);
}
