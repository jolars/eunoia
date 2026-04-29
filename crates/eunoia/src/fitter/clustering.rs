//! Cluster detection for disjoint groups of shapes.

use crate::geometry::diagram::RegionMask;
use crate::geometry::traits::Closed;

/// Identifies disjoint clusters of shapes based on region overlap.
///
/// Two shapes are in the same cluster if their regions overlap (boundaries
/// cross OR one contains the other), or if they share an overlap with a common
/// third shape (transitive closure). Containment counts as overlap here even
/// though `Closed::intersects` returns `false` for it — that method only
/// reports boundary crossings, but for clustering purposes a contained shape
/// is plainly part of the same cluster as its container.
///
/// Returns a vector of clusters, where each cluster is a vector of shape indices.
pub fn find_clusters<S: Closed>(shapes: &[S]) -> Vec<Vec<usize>> {
    let n = shapes.len();

    if n == 0 {
        return vec![];
    }

    // Build adjacency matrix based on region overlap (intersection or containment).
    let mut adjacency = vec![vec![false; n]; n];

    for i in 0..n {
        adjacency[i][i] = true; // Each shape is connected to itself
        for j in (i + 1)..n {
            let connected = shapes[i].intersects(&shapes[j])
                || shapes[i].contains(&shapes[j])
                || shapes[j].contains(&shapes[i]);
            if connected {
                adjacency[i][j] = true;
                adjacency[j][i] = true;
            }
        }
    }

    // Compute transitive closure (Floyd-Warshall style)
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if adjacency[i][k] && adjacency[k][j] {
                    adjacency[i][j] = true;
                }
            }
        }
    }

    // Extract unique clusters
    let mut seen = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if seen[i] {
            continue;
        }

        let mut cluster = Vec::new();
        for (j, &connected) in adjacency[i].iter().enumerate() {
            if connected {
                cluster.push(j);
                seen[j] = true;
            }
        }

        if !cluster.is_empty() {
            clusters.push(cluster);
        }
    }

    clusters
}

/// Cluster shapes from the exclusive-region area map produced by
/// [`crate::geometry::traits::DiagramShape::compute_exclusive_regions`].
///
/// Two shapes are connected if any *exclusive* region whose bitmask contains
/// both of their indices has area greater than `tolerance`. Transitive
/// closure then groups them. This is the same shared-region notion as the
/// geometric `find_clusters`, but it consumes the exact-conic area math
/// the optimizer already runs — eliminating the
/// `Closed::intersects`-vs-`compute_exclusive_regions` agreement gap that
/// causes `normalize_layout` to flake on near-coincident ellipse fits
/// (the `intersects` quick-reject + boundary-crossing path can disagree
/// with the inclusion-exclusion-derived exclusive areas in floating-point
/// edge cases). See `crates/eunoia/src/fitter/normalize.rs` and
/// `crates/eunoia/src/fitter.rs::Fitter::fit`.
///
/// `tolerance` is the absolute area threshold below which a region is
/// treated as empty. The caller should pass something derived from the
/// global region scale, not a fixed constant — e.g. `1e-10 * max_region`.
pub fn find_clusters_from_exclusive_regions(
    n_shapes: usize,
    exclusive_areas: &std::collections::HashMap<RegionMask, f64>,
    tolerance: f64,
) -> Vec<Vec<usize>> {
    if n_shapes == 0 {
        return vec![];
    }

    let mut adjacency = vec![vec![false; n_shapes]; n_shapes];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_shapes {
        adjacency[i][i] = true;
    }

    for (&mask, &area) in exclusive_areas {
        if !area.is_finite() || area <= tolerance {
            continue;
        }
        // Bits set in `mask` are the shape indices that share this exclusive
        // region. Anything with at least two participants is a non-trivial
        // overlap and connects every pair of those participants.
        if (mask as u64).count_ones() < 2 {
            continue;
        }
        let indices: Vec<usize> = (0..n_shapes).filter(|&i| (mask >> i) & 1 == 1).collect();
        for &i in &indices {
            for &j in &indices {
                adjacency[i][j] = true;
            }
        }
    }

    // Transitive closure (Floyd-Warshall).
    for k in 0..n_shapes {
        for i in 0..n_shapes {
            for j in 0..n_shapes {
                if adjacency[i][k] && adjacency[k][j] {
                    adjacency[i][j] = true;
                }
            }
        }
    }

    let mut seen = vec![false; n_shapes];
    let mut clusters = Vec::new();
    for i in 0..n_shapes {
        if seen[i] {
            continue;
        }
        let mut cluster = Vec::new();
        for (j, &connected) in adjacency[i].iter().enumerate() {
            if connected {
                cluster.push(j);
                seen[j] = true;
            }
        }
        if !cluster.is_empty() {
            clusters.push(cluster);
        }
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::Circle;

    #[test]
    fn test_single_shape() {
        let shapes = vec![Circle::new(Point::new(0.0, 0.0), 1.0)];
        let clusters = find_clusters(&shapes);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0]);
    }

    #[test]
    fn test_two_disjoint_shapes() {
        let shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(10.0, 0.0), 1.0),
        ];
        let clusters = find_clusters(&shapes);
        assert_eq!(clusters.len(), 2);
        assert!(clusters.contains(&vec![0]));
        assert!(clusters.contains(&vec![1]));
    }

    #[test]
    fn test_two_overlapping_shapes() {
        let shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 2.0),
            Circle::new(Point::new(2.0, 0.0), 2.0),
        ];
        let clusters = find_clusters(&shapes);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn test_three_shapes_two_clusters() {
        let shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 1.5),
            Circle::new(Point::new(2.0, 0.0), 1.5),
            Circle::new(Point::new(10.0, 0.0), 1.0),
        ];
        let clusters = find_clusters(&shapes);
        assert_eq!(clusters.len(), 2);

        // Find which cluster has 2 elements
        let large_cluster = clusters.iter().find(|c| c.len() == 2).unwrap();
        let small_cluster = clusters.iter().find(|c| c.len() == 1).unwrap();

        assert!(large_cluster.contains(&0));
        assert!(large_cluster.contains(&1));
        assert_eq!(small_cluster[0], 2);
    }

    #[test]
    fn test_transitive_clustering() {
        // A intersects B, B intersects C, but A doesn't intersect C
        // All three should be in same cluster due to transitive closure
        let shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 1.5),
            Circle::new(Point::new(2.0, 0.0), 1.5),
            Circle::new(Point::new(4.0, 0.0), 1.5),
        ];
        let clusters = find_clusters(&shapes);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn area_based_two_disjoint_shapes() {
        // Two shapes, no shared region: two separate clusters.
        let mut areas: std::collections::HashMap<RegionMask, f64> =
            std::collections::HashMap::new();
        areas.insert(0b01, 3.0); // only shape 0
        areas.insert(0b10, 4.0); // only shape 1
        let clusters = find_clusters_from_exclusive_regions(2, &areas, 1e-12);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn area_based_overlapping_pair() {
        // Two shapes with a non-empty pairwise region: one cluster.
        let mut areas: std::collections::HashMap<RegionMask, f64> =
            std::collections::HashMap::new();
        areas.insert(0b01, 1.0);
        areas.insert(0b10, 1.0);
        areas.insert(0b11, 0.5);
        let clusters = find_clusters_from_exclusive_regions(2, &areas, 1e-12);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn area_based_transitive() {
        // 3 shapes: only {0,1} and {1,2} have non-zero exclusive regions.
        // Transitive closure should merge all three.
        let mut areas: std::collections::HashMap<RegionMask, f64> =
            std::collections::HashMap::new();
        areas.insert(0b001, 1.0);
        areas.insert(0b010, 1.0);
        areas.insert(0b100, 1.0);
        areas.insert(0b011, 0.5); // shapes 0,1
        areas.insert(0b110, 0.5); // shapes 1,2
        let clusters = find_clusters_from_exclusive_regions(3, &areas, 1e-12);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn area_based_tolerance_rejects_noise() {
        // A "shared" region of size 1e-15 should NOT connect shapes when
        // tolerance is 1e-10 — keeps inclusion-exclusion roundoff from
        // spuriously merging clusters.
        let mut areas: std::collections::HashMap<RegionMask, f64> =
            std::collections::HashMap::new();
        areas.insert(0b01, 5.0);
        areas.insert(0b10, 5.0);
        areas.insert(0b11, 1e-15);
        let clusters = find_clusters_from_exclusive_regions(2, &areas, 1e-10);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn area_based_triple_intersection_connects_all() {
        // A 3-set Venn shape where ONLY the triple region (0b111) has
        // positive area — pairwise regions are zero. The triple still
        // implies all three shapes overlap each other.
        let mut areas: std::collections::HashMap<RegionMask, f64> =
            std::collections::HashMap::new();
        areas.insert(0b001, 1.0);
        areas.insert(0b010, 1.0);
        areas.insert(0b100, 1.0);
        areas.insert(0b111, 2.0); // all three
        let clusters = find_clusters_from_exclusive_regions(3, &areas, 1e-12);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }
}
