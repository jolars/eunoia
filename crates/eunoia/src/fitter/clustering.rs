//! Cluster detection for disjoint groups of shapes.

use crate::geometry::traits::Closed;

/// Identifies disjoint clusters of shapes based on intersection relationships.
///
/// Two shapes are in the same cluster if they intersect, or if they both
/// intersect a common third shape (transitive closure).
///
/// Returns a vector of clusters, where each cluster is a vector of shape indices.
pub fn find_clusters<S: Closed>(shapes: &[S]) -> Vec<Vec<usize>> {
    let n = shapes.len();

    if n == 0 {
        return vec![];
    }

    // Build adjacency matrix based on intersections
    let mut adjacency = vec![vec![false; n]; n];

    for i in 0..n {
        adjacency[i][i] = true; // Each shape is connected to itself
        for j in (i + 1)..n {
            if shapes[i].intersects(&shapes[j]) {
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

/// Groups shapes into clusters based on fitted areas having non-zero intersections.
///
/// This is an alternative to geometric intersection testing that uses the
/// computed region areas to determine connectivity.
#[allow(dead_code)]
pub fn find_clusters_from_areas(
    n_shapes: usize,
    fitted_areas: &std::collections::HashMap<crate::spec::Combination, f64>,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![vec![false; n_shapes]; n_shapes];

    // Mark self-connections
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_shapes {
        adjacency[i][i] = true;
    }

    // Check all combinations to find intersections
    for (combo, &area) in fitted_areas {
        if area > 1e-10 && combo.sets().len() >= 2 {
            // This combination has positive area, so all pairs in it intersect
            let indices: Vec<usize> = combo
                .sets()
                .iter()
                .filter_map(|_| None) // We need set names -> indices mapping
                .collect();

            // Mark all pairs as connected
            for i in 0..indices.len() {
                for j in 0..indices.len() {
                    adjacency[indices[i]][indices[j]] = true;
                }
            }
        }
    }

    // Compute transitive closure
    for k in 0..n_shapes {
        for i in 0..n_shapes {
            for j in 0..n_shapes {
                if adjacency[i][k] && adjacency[k][j] {
                    adjacency[i][j] = true;
                }
            }
        }
    }

    // Extract unique clusters
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
}
