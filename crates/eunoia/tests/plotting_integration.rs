//! Integration tests for the plotting module.

#[cfg(feature = "plotting")]
use eunoia::geometry::shapes::Circle;
#[cfg(feature = "plotting")]
use eunoia::spec::{DiagramSpecBuilder, InputType};
#[cfg(feature = "plotting")]
use eunoia::Fitter;

#[test]
#[cfg(feature = "plotting")]
fn test_region_decomposition_two_sets() {
    let spec = DiagramSpecBuilder::new()
        .set("A", 10.0)
        .set("B", 8.0)
        .intersection(&["A", "B"], 2.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

    // Get region polygons using the convenient method
    let regions = layout.region_polygons(&spec, 64);

    // Should have at least A-only, B-only, and A&B regions
    assert!(regions.len() >= 2);

    // Check that total polygon area is close to expected
    let total_area: f64 = regions.areas().values().sum();
    let expected_total: f64 = spec.exclusive_areas().values().sum();

    assert!(
        (total_area - expected_total).abs() < 1.0,
        "Total area {:.3} should be close to expected {:.3}",
        total_area,
        expected_total
    );
}

#[test]
#[cfg(feature = "plotting")]
fn test_region_decomposition_three_sets() {
    let spec = DiagramSpecBuilder::new()
        .set("A", 5.0)
        .set("B", 5.0)
        .set("C", 5.0)
        .intersection(&["A", "B"], 1.0)
        .intersection(&["B", "C"], 1.0)
        .intersection(&["A", "C"], 1.0)
        .intersection(&["A", "B", "C"], 0.5)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    let layout = Fitter::<Circle>::new(&spec).seed(123).fit().unwrap();
    let regions = layout.region_polygons(&spec, 128);

    // Should have multiple regions
    assert!(regions.len() >= 3);

    // Verify each region has polygons
    for (combo, polys) in regions.iter() {
        assert!(!polys.is_empty(), "Region {:?} should have polygons", combo);

        // Verify each polygon has vertices
        for poly in polys {
            assert!(
                poly.vertices().len() >= 3,
                "Polygon should have at least 3 vertices"
            );
        }
    }
}

#[test]
#[cfg(feature = "plotting")]
fn test_region_polygons_with_ellipses() {
    use eunoia::geometry::shapes::Ellipse;

    let spec = DiagramSpecBuilder::new()
        .set("A", 8.0)
        .set("B", 6.0)
        .intersection(&["A", "B"], 2.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    let layout = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();
    let regions = layout.region_polygons(&spec, 96);

    assert!(regions.len() >= 2);

    // Check areas are reasonable
    let areas = regions.areas();
    for (_, &area) in areas.iter() {
        assert!(area > 0.0, "All regions should have positive area");
    }
}
