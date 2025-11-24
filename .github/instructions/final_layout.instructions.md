---
applyTo: "src/fitter/final_layout.rs"
---

# Region Discovery and Area Computation Strategy

This module implements the final layout optimization by computing region areas
and comparing them to target areas. The key challenge is efficiently discovering
which regions exist in the diagram and computing their areas.

### Core Concepts

**Region Representation with Bit Masks:**

- A region is a combination of circles (sets) represented as a bit mask
  (integer)
- Each bit position represents whether a circle is part of the region
- Example: For 4 circles (A=0, B=1, C=2, D=3):
  - Circle A only: `0b0001` = 1
  - Circles A and B: `0b0011` = 3
  - Circles A, B, and C: `0b0111` = 7
- Use `type RegionMask = usize;` for clarity

**Sparse Region Discovery:**

- **DO NOT** enumerate all 2^n - 1 combinations (exponential!)
- **DO** discover only regions that actually exist from three sources:
  1. **Singles** - All base circles (always exist)
  2. **From intersection points** - Group by `adopters` field
  3. **From containment** - Check pairs with no intersection points

### Implementation Approach

#### Step 1: Discover Regions (Sparse)

```rust
type RegionMask = usize;

fn discover_regions(
    shapes: &[Circle],
    intersections: &[IntersectionInfo],
    n_sets: usize,
) -> HashSet<RegionMask> {
    let mut regions = HashSet::new();

    // 1. Singles always exist
    for i in 0..n_sets {
        regions.insert(1 << i);
    }

    // 2. From intersection points - convert adopters to masks
    for info in intersections {
        let mask = adopters_to_mask(&info.adopters);
        regions.insert(mask);
    }

    // 3. From containment (pairs with no edge intersection)
    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let has_edge_intersection = intersections.iter().any(|info| info.parents == (i, j));

            if !has_edge_intersection {
                if shapes[i].contains(&shapes[j]) || shapes[j].contains(&shapes[i]) {
                    regions.insert((1 << i) | (1 << j));
                }
            }
        }
    }

    regions
}

fn adopters_to_mask(adopters: &[usize]) -> RegionMask {
    adopters.iter().fold(0, |mask, &i| mask | (1 << i))
}
```

#### Step 2: Compute Region Areas

For each discovered region, compute its area based on its type:

```rust
fn compute_region_area(
    mask: RegionMask,
    shapes: &[Circle],
    intersections: &[IntersectionInfo],
    n_sets: usize,
) -> f64 {
    let circle_count = mask.count_ones();

    match circle_count {
        1 => {
            // Single circle - just return its area
            let idx = mask.trailing_zeros() as usize;
            shapes[idx].area()
        }
        2 => {
            // Two circles - use intersection_area
            let indices = mask_to_indices(mask, n_sets);
            shapes[indices[0]].intersection_area(&shapes[indices[1]])
        }
        _ => {
            // 3+ circles - need polygon area computation
            // Find intersection points that belong to this region
            let region_points: Vec<_> = intersections
                .iter()
                .filter(|info| adopters_to_mask(&info.adopters) == mask)
                .collect();

            if region_points.is_empty() {
                0.0 // No geometry
            } else {
                compute_polygon_area(&region_points, shapes)
            }
        }
    }
}
```

#### Step 3: Compute Polygon Area (for 3+ way intersections)

Use the shoelace formula with circular segments:

```rust
fn compute_polygon_area(points: &[&IntersectionInfo], shapes: &[Circle]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // Sort points by angle around centroid
    let centroid = compute_centroid(points);
    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| {
        let angle_a = (a.point.y() - centroid.1).atan2(a.point.x() - centroid.0);
        let angle_b = (b.point.y() - centroid.1).atan2(b.point.x() - centroid.0);
        angle_a.partial_cmp(&angle_b).unwrap()
    });

    let mut area = 0.0;
    let n = sorted_points.len();

    for i in 0..n {
        let p0 = &sorted_points[i].point;
        let p1 = &sorted_points[(i + 1) % n].point;

        // Triangular part (shoelace)
        area += 0.5 * (p0.x() * p1.y() - p1.x() * p0.y());

        // Add circular segment between points
        // Find which circle the edge belongs to (common parent)
        let common_circles = find_common_parents(
            &sorted_points[i].parents,
            &sorted_points[(i + 1) % n].parents,
        );

        if !common_circles.is_empty() {
            // Add segment area from the appropriate circle
            let circle_idx = common_circles[0];
            let segment = compute_circular_segment(&shapes[circle_idx], p0, p1);
            area += segment;
        }
    }

    area.abs()
}
```

#### Step 4: Convert to Disjoint Areas

Use inclusion-exclusion principle to get disjoint areas from overlapping areas:

```rust
fn to_disjoint_areas(overlapping_areas: &HashMap<RegionMask, f64>) -> HashMap<RegionMask, f64> {
    let mut disjoint = overlapping_areas.clone();

    // Sort masks by bit count (process larger sets first)
    let mut masks: Vec<_> = overlapping_areas.keys().copied().collect();
    masks.sort_by_key(|m| std::cmp::Reverse(m.count_ones()));

    // For each region, subtract all its supersets
    for &mask_i in &masks {
        for &mask_j in &masks {
            if mask_i != mask_j && is_subset(mask_i, mask_j) {
                *disjoint.get_mut(&mask_i).unwrap() -= disjoint[&mask_j];
            }
        }
    }

    // Clamp to non-negative
    for area in disjoint.values_mut() {
        *area = area.max(0.0);
    }

    disjoint
}

fn is_subset(mask1: RegionMask, mask2: RegionMask) -> bool {
    (mask1 & mask2) == mask1
}
```

### Reference Implementation

Refer to `eulerr/src/optim_final.cpp` and `eulerr/src/compute-areas.cpp` for the
reference algorithm. Key differences:

- eulerr enumerates all 2^n combinations (dense)
- eunoia discovers only existing regions (sparse) - O(n²) vs O(2^n)
- Both use similar polygon area computation with circular segments

### Helper Functions Needed

```rust
fn mask_to_indices(mask: RegionMask, n_sets: usize) -> Vec<usize> {
    (0..n_sets).filter(|&i| (mask & (1 << i)) != 0).collect()
}

fn compute_centroid(points: &[&IntersectionInfo]) -> (f64, f64) {
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|p| p.point.x()).sum();
    let sum_y: f64 = points.iter().map(|p| p.point.y()).sum();
    (sum_x / n, sum_y / n)
}

fn find_common_parents(parents1: &(usize, usize), parents2: &(usize, usize)) -> Vec<usize> {
    let set1: HashSet<_> = [parents1.0, parents1.1].into_iter().collect();
    let set2: HashSet<_> = [parents2.0, parents2.1].into_iter().collect();
    set1.intersection(&set2).copied().collect()
}

fn compute_circular_segment(circle: &Circle, p0: &Point, p1: &Point) -> f64 {
    // Use the segment_area methods from Circle
    let chord_length = p0.distance(p1);
    circle.segment_area_from_chord(chord_length)
}
```

### Error Computation

Once you have disjoint areas:

```rust
fn compute_region_error(
    fitted_areas: &HashMap<RegionMask, f64>,
    target_areas: &HashMap<RegionMask, f64>,
) -> f64 {
    let mut error = 0.0;

    for (mask, &target) in target_areas {
        let fitted = fitted_areas.get(mask).copied().unwrap_or(0.0);
        let diff = fitted - target;
        error += diff * diff;
    }

    error
}
```

### Performance Notes

- Bit mask operations are O(1) - single CPU instructions
- Sparse discovery is O(n²) for containment checks
- Dense enumeration would be O(2^n) - avoid at all costs!
- For n=10: sparse = ~100 regions, dense = 1023 regions (10x faster)
- For n=20: sparse = ~500 regions, dense = 1,048,575 regions (2000x faster!)

### Testing Strategy

1. Test region discovery with various configurations
2. Test area computation for 2-way, 3-way overlaps
3. Test conversion to disjoint areas
4. Compare against eulerr for validation
5. Test edge cases: containment, disjoint, touching circles
