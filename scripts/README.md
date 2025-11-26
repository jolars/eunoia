# Scripts Directory

This directory contains utility scripts for the Eunoia project.

## generate_eulerr_test_cases.R

R script to generate test cases for comparing Eunoia's area calculations against the reference eulerr R package.

### Usage

```bash
Rscript scripts/generate_eulerr_test_cases.R > test_output.txt
```

### Requirements

- R (tested with R 4.5.1)
- eulerr R package

Install the eulerr package if needed:

```r
install.packages("eulerr")
```

### What it does

The script:

1. Defines multiple test configurations of three circles:
   - Simple overlapping circles
   - Well-separated circles (no intersections)
   - Highly overlapping circles
   - Nested configuration (one circle inside intersection of two others)
   - Original failing test case

2. For each configuration:
   - Computes exact intersection areas using `eulerr:::intersect_ellipses()`
   - Generates Rust test code with the circle definitions and expected areas
   - Formats output for easy copy-paste into Rust test files

3. Output includes:
   - Circle parameters (center h, k and radius a, b for each set)
   - Exact area values from eulerr
   - Ready-to-use Rust test functions

### Test cases generated

The script generates test functions that can be added to `crates/eunoia/src/geometry/diagram.rs`:

**3-circle tests (passing with <1% error):**
- `test_eulerr_comparison_simple_overlapping()`
- `test_eulerr_comparison_well_separated()`
- `test_eulerr_comparison_highly_overlapping()`
- `test_eulerr_comparison_nested_configuration()`
- `test_eulerr_comparison_original_failing()`

**4 and 5-circle tests (currently marked as `#[ignore]` due to accuracy issues):**
- `test_eulerr_comparison_four_circles_square()`
- `test_eulerr_comparison_four_circles_center()`
- `test_eulerr_comparison_five_circles_circular()`
- `test_eulerr_comparison_five_circles_cross()`

**Note**: The 4 and 5 circle tests currently show accuracy issues (>10% error for some regions),
suggesting potential problems with the polygon-based area calculation or region discovery
for higher-order intersections. These tests are kept as ignored tests to track progress
on improving 4+ circle support.

### Adding new test cases

To add new test cases, edit the script and add a new configuration:

```r
# Test Case N: Description
cat("=== Test Case N: Description ===\n")
circlesN <- data.frame(
  A = c(h_A, k_A, r_A, r_A, 0),
  B = c(h_B, k_B, r_B, r_B, 0),
  C = c(h_C, k_C, r_C, r_C, 0),
  row.names = c("h", "k", "a", "b", "phi")
)
areasN <- eulerr:::intersect_ellipses(as.matrix(circlesN), FALSE, FALSE)
cat("Circles:\n")
print(circlesN)
cat("\nAreas:", areasN, "\n\n")
```

Then add it to the test_cases list:

```r
test_cases <- list(
  # ... existing cases ...
  list(name = "my_new_test", circles = circlesN, areas = areasN)
)
```

### Notes

- The script uses `eulerr:::intersect_ellipses()` which is an internal eulerr function
- For circles, we set `a = b = radius` and `phi = 0` (no rotation)
- The eulerr package uses the ellipse parameterization where:
  - `h, k`: center coordinates
  - `a, b`: semi-major and semi-minor axes (for circles, a = b = radius)
  - `phi`: rotation angle (0 for circles)
