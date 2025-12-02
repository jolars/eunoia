//! Example demonstrating pole of inaccessibility for label placement.
//!
//! This example shows how to use the pole_of_inaccessibility method
//! to find optimal label positions for polygons.
//!
//! Run with: cargo run --example label_placement --features plotting

#[cfg(feature = "plotting")]
use eunoia::geometry::primitives::Point;
#[cfg(feature = "plotting")]
use eunoia::geometry::shapes::Polygon;

#[cfg(not(feature = "plotting"))]
fn main() {
    eprintln!("This example requires the 'plotting' feature.");
    eprintln!("Run with: cargo run --example label_placement --features plotting");
    std::process::exit(1);
}

#[cfg(feature = "plotting")]
fn main() {
    println!("=== Label Placement Example ===\n");

    // Example 1: Square - pole should be near center
    println!("1. Square:");
    let square = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        Point::new(10.0, 10.0),
        Point::new(0.0, 10.0),
    ]);
    let centroid = square.centroid();
    let pole = square.pole_of_inaccessibility(0.01);
    println!("   Centroid: ({:.2}, {:.2})", centroid.x(), centroid.y());
    println!("   Pole:     ({:.2}, {:.2})", pole.x(), pole.y());
    println!("   → Similar positions (symmetric shape)\n");

    // Example 2: L-shape - pole is better than centroid
    println!("2. L-shaped polygon:");
    let l_shape = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(4.0, 0.0),
        Point::new(4.0, 1.0),
        Point::new(1.0, 1.0),
        Point::new(1.0, 4.0),
        Point::new(0.0, 4.0),
    ]);
    let centroid = l_shape.centroid();
    let pole = l_shape.pole_of_inaccessibility(0.1);
    println!("   Centroid: ({:.2}, {:.2})", centroid.x(), centroid.y());
    println!("   Pole:     ({:.2}, {:.2})", pole.x(), pole.y());
    println!("   → Pole is in the thick part (better for labels!)\n");

    // Example 3: Rectangle - pole should be at center
    println!("3. Rectangle:");
    let rect = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(20.0, 0.0),
        Point::new(20.0, 5.0),
        Point::new(0.0, 5.0),
    ]);
    let centroid = rect.centroid();
    let pole = rect.pole_of_inaccessibility(0.1);
    println!("   Centroid: ({:.2}, {:.2})", centroid.x(), centroid.y());
    println!("   Pole:     ({:.2}, {:.2})", pole.x(), pole.y());
    println!("   → Similar positions (symmetric shape)\n");

    // Example 4: C-shaped polygon
    println!("4. C-shaped polygon:");
    let c_shape = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        Point::new(10.0, 2.0),
        Point::new(2.0, 2.0),
        Point::new(2.0, 8.0),
        Point::new(10.0, 8.0),
        Point::new(10.0, 10.0),
        Point::new(0.0, 10.0),
    ]);
    let centroid = c_shape.centroid();
    let pole = c_shape.pole_of_inaccessibility(0.1);
    println!("   Centroid: ({:.2}, {:.2})", centroid.x(), centroid.y());
    println!("   Pole:     ({:.2}, {:.2})", pole.x(), pole.y());
    println!("   → Pole avoids the opening (better for labels!)\n");

    println!("=== Summary ===");
    println!("For symmetric shapes: pole ≈ centroid");
    println!("For complex/concave shapes: pole > centroid (better visual center)");
    println!("\nUse pole_of_inaccessibility() for better label placement!");
}
