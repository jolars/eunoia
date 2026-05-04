//! Region decomposition for diagram visualization.
//!
//! This module provides utilities for decomposing fitted shapes into
//! exclusive regions (one per set combination) for plotting.
//!
//! # Output structure
//!
//! [`RegionPolygons`] stores each region as a list of [`RegionPiece`]s, where
//! a *piece* is a single connected component with one outer boundary and
//! zero or more interior holes. Disconnected regions (e.g. an "A only"
//! region split by other sets cutting across A) produce multiple pieces.
//!
//! Outer rings are normalised to **CCW** (positive signed area) and holes
//! to **CW** (negative signed area) so renderers can draw each piece with
//! `fill-rule: nonzero` (the SVG default) without further bookkeeping.

use crate::geometry::diagram::{discover_regions, IntersectionPoint};
use crate::geometry::primitives::Point;
use crate::geometry::shapes::Polygon;
use crate::geometry::traits::{Closed, DiagramShape, Polygonize};
use crate::plotting::clip::{polygon_clip_many, ClipOperation};
use crate::spec::{Combination, DiagramSpec};
use std::collections::HashMap;

/// A connected component of a region: one outer boundary plus zero or more
/// interior holes (other regions or shapes cutting through this piece).
///
/// Orientation is normalised by the core library so consumers can render
/// each piece with the SVG default `fill-rule: nonzero` (or any equivalent
/// nonzero / "even-odd-of-rings" rule) **without classifying rings on their
/// own side**:
///
/// * `outer` — CCW (positive signed area).
/// * `holes` — CW (negative signed area), each strictly inside `outer`.
///
/// Use [`RegionPiece::area`] for the piece's net area (outer minus holes).
///
/// # Renderer recipe
///
/// SVG: emit one `<path>` per piece with `d` formed by concatenating the
/// outer ring's `M…Z` and each hole's `M…Z`. Default `fill-rule: nonzero`
/// fills only the donut/cookie shape because the rings have opposite
/// winding.
///
/// Canvas: `ctx.beginPath()`, walk the outer ring, walk each hole ring,
/// `ctx.fill()` (Canvas's default fill is also nonzero).
///
/// # Construction
///
/// `RegionPiece`s are produced by [`decompose_regions`] (and the higher-
/// level [`crate::Layout::region_polygons`]). Bindings authors should not
/// build them by hand: the topological classification that maps a flat
/// list of clipper-output rings to outer/holes pairs is not part of the
/// public API.
#[derive(Debug, Clone)]
pub struct RegionPiece {
    /// CCW outer boundary of the piece (always non-degenerate after
    /// [`decompose_regions`]).
    pub outer: Polygon,
    /// CW hole rings, each strictly inside `outer`. Empty when the piece
    /// is simply connected.
    pub holes: Vec<Polygon>,
}

impl RegionPiece {
    /// Net area of the piece — `outer` minus the sum of `holes`. Always
    /// non-negative because orientations are normalised.
    pub fn area(&self) -> f64 {
        let outer_area = self.outer.area();
        let hole_area: f64 = self.holes.iter().map(|h| h.area()).sum();
        (outer_area - hole_area).max(0.0)
    }
}

/// Collection of region pieces for each exclusive region in a diagram.
///
/// Each key is a [`Combination`] of set names, and the value is a list of
/// [`RegionPiece`]s that together represent that exclusive region. Multiple
/// pieces occur when a region is disconnected (e.g. an "A only" lobe split
/// by an intersection cutting across A); piece orientations and hole
/// containment are guaranteed by [`decompose_regions`] — see
/// [`RegionPiece`] for the rendering contract.
///
/// Pieces below the per-diagram sliver threshold (see [`decompose_regions`])
/// are filtered out before insertion; consumers can assume every retained
/// piece has positive net area.
#[derive(Debug, Clone)]
pub struct RegionPolygons {
    regions: HashMap<Combination, Vec<RegionPiece>>,
}

impl RegionPolygons {
    /// Creates a new empty `RegionPolygons` collection. Most callers should
    /// instead use [`decompose_regions`] (or [`crate::Layout::region_polygons`])
    /// to populate one from fitted shapes.
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
        }
    }

    /// Adds pieces for a given region, replacing any previous entry for
    /// `combination`. The caller is responsible for ensuring the pieces
    /// satisfy the [`RegionPiece`] orientation contract — typically only
    /// useful for tests; production code should rely on [`decompose_regions`].
    pub fn insert(&mut self, combination: Combination, pieces: Vec<RegionPiece>) {
        self.regions.insert(combination, pieces);
    }

    /// Returns the pieces for a given region, or `None` if the combination
    /// has no fitted (non-empty) area.
    pub fn get(&self, combination: &Combination) -> Option<&Vec<RegionPiece>> {
        self.regions.get(combination)
    }

    /// Iterates over every non-empty region and its pieces. Iteration order
    /// is unspecified (backed by a `HashMap`); sort the result if you need
    /// determinism.
    pub fn iter(&self) -> impl Iterator<Item = (&Combination, &Vec<RegionPiece>)> {
        self.regions.iter()
    }

    /// Number of distinct non-empty regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// `true` when no regions are stored.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Computes the total (net) area for each region — sum of piece areas,
    /// where each piece's area is its outer area minus its holes.
    pub fn areas(&self) -> HashMap<Combination, f64> {
        self.regions
            .iter()
            .map(|(combo, pieces)| {
                let area = pieces.iter().map(|p| p.area()).sum();
                (combo.clone(), area)
            })
            .collect()
    }

    /// Computes a hole-aware label anchor point for every non-empty region.
    ///
    /// For each region, this returns the *pole of inaccessibility* — the
    /// interior point that maximises the minimum distance to **every**
    /// boundary, both the outer ring and any holes (the Polylabel algorithm
    /// extended to hole-bearing shapes). When a region is disconnected into
    /// multiple [`RegionPiece`]s, the piece with the largest pole-to-boundary
    /// clearance wins — i.e. the piece with the most breathing room for a
    /// label, which is not always the piece with the largest area (a thin
    /// crescent can have large area but no spot wide enough to fit text).
    ///
    /// Regions with no pieces (or whose pieces are degenerate) are omitted.
    ///
    /// # Arguments
    ///
    /// * `precision` - Polylabel precision, in the same units as the polygon
    ///   coordinates. Smaller values yield more accurate anchors at higher
    ///   cost. A value of roughly 1% of the diagram's extent is typical
    ///   (e.g. `0.01` when coordinates are normalized to `[0, 1]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 5.0)
    ///     .set("B", 3.0)
    ///     .intersection(&["A", "B"], 1.0)
    ///     .input_type(InputType::Exclusive)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
    /// let regions = layout.region_polygons(&spec, 64);
    /// let labels = regions.label_points(0.01);
    ///
    /// // One label point per non-empty region.
    /// for (combo, point) in &labels {
    ///     println!("{}: ({:.2}, {:.2})", combo, point.x(), point.y());
    /// }
    /// ```
    pub fn label_points(&self, precision: f64) -> HashMap<Combination, Point> {
        self.regions
            .iter()
            .filter_map(|(combo, pieces)| {
                let (point, _) = poi_with_holes(pieces, precision)?;
                Some((combo.clone(), point))
            })
            .collect()
    }

    /// Computes a label anchor point for every set in `set_names`, by
    /// unioning every region containing the set, reclassifying the union
    /// into [`RegionPiece`]s, and returning the hole-aware pole of
    /// inaccessibility of the highest-clearance piece.
    ///
    /// This works for nested sets in the same way it works for overlapping
    /// ones — but if you also want the eulerr-style fallback ("if a set has
    /// no exclusive region of its own, place its label inside the largest
    /// containing intersection"), use [`crate::plotting::PlotData::set_anchors`]
    /// instead. That builds anchors directly from the shape outlines and
    /// applies the fallback automatically; this method only sees the region
    /// decomposition.
    ///
    /// Sets that are absent from every region are omitted from the result.
    ///
    /// # Arguments
    ///
    /// * `set_names` - The full list of set names in the diagram (typically
    ///   `spec.set_names()`).
    /// * `precision` - Polylabel precision, in the same units as the polygon
    ///   coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 5.0)
    ///     .set("B", 3.0)
    ///     .intersection(&["A", "B"], 1.0)
    ///     .input_type(InputType::Exclusive)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
    /// let regions = layout.region_polygons(&spec, 64);
    /// let set_labels = regions.set_label_points(spec.set_names(), 0.01);
    ///
    /// assert!(set_labels.contains_key("A"));
    /// assert!(set_labels.contains_key("B"));
    /// ```
    pub fn set_label_points(&self, set_names: &[String], precision: f64) -> HashMap<String, Point> {
        let mut result = HashMap::new();

        for name in set_names {
            // Flatten every piece (outer + holes) of every region containing
            // this set into a single ring list, union them, and reclassify
            // into pieces — gives the hole-aware POI of the unioned coverage.
            let mut owned: Vec<Polygon> = Vec::new();
            for (combo, pieces) in self.regions.iter() {
                if !combo.sets().iter().any(|s| s == name) {
                    continue;
                }
                for piece in pieces {
                    owned.push(piece.outer.clone());
                    owned.extend(piece.holes.iter().cloned());
                }
            }

            if owned.is_empty() {
                continue;
            }

            let mut merged = vec![owned.remove(0)];
            for p in owned {
                merged = polygon_clip_many(&merged, &p, ClipOperation::Union);
                if merged.is_empty() {
                    break;
                }
            }

            let pieces = classify_into_pieces(merged);
            if let Some((point, _)) = poi_with_holes(&pieces, precision) {
                result.insert(name.clone(), point);
            }
        }

        result
    }
}

/// Signed area of a polygon ring via the shoelace formula. Positive = CCW,
/// negative = CW.
fn signed_polygon_area(p: &Polygon) -> f64 {
    let v = p.vertices();
    let n = v.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += v[i].x() * v[j].y() - v[j].x() * v[i].y();
    }
    0.5 * s
}

fn point_in_polygon(p: &Point, poly: &Polygon) -> bool {
    let v = poly.vertices();
    let n = v.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (v[i].x(), v[i].y());
        let (xj, yj) = (v[j].x(), v[j].y());
        let intersect = ((yi > p.y()) != (yj > p.y()))
            && (p.x() < (xj - xi) * (p.y() - yi) / (yj - yi + f64::EPSILON) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Reverse the vertex order of a polygon, flipping its winding.
fn reverse_polygon(p: &Polygon) -> Polygon {
    let mut v = p.vertices().to_vec();
    v.reverse();
    Polygon::new(v)
}

/// Classify a flat list of clipper-output rings into [`RegionPiece`]s by
/// topological containment. Each ring's depth is the number of other rings
/// strictly containing one of its vertices: even depth → outer, odd depth →
/// hole. Holes are assigned to the smallest enclosing outer.
///
/// Outputs are normalised so `outer` is CCW (positive signed area) and every
/// `hole` is CW (negative signed area). Degenerate rings (fewer than three
/// vertices, or near-zero area) are dropped.
///
/// We classify by containment rather than winding because i_overlay's output
/// orientation is not stable across operations — relying on the sign of the
/// signed area to identify outers vs holes silently breaks for chained
/// difference/union operations.
pub(crate) fn classify_into_pieces(rings: Vec<Polygon>) -> Vec<RegionPiece> {
    // Drop degenerate rings.
    let kept: Vec<(Polygon, f64)> = rings
        .into_iter()
        .filter_map(|p| {
            let area = signed_polygon_area(&p).abs();
            if p.vertices().len() < 3 || area < 1e-12 {
                None
            } else {
                Some((p, area))
            }
        })
        .collect();
    if kept.is_empty() {
        return Vec::new();
    }

    let n = kept.len();
    let mut containment_depth = vec![0usize; n];
    let mut smallest_container: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let probe = match kept[i].0.vertices().first() {
            Some(p) => *p,
            None => continue,
        };
        for (j, (other, area_j)) in kept.iter().enumerate() {
            if i == j {
                continue;
            }
            if point_in_polygon(&probe, other) {
                containment_depth[i] += 1;
                let cur_area = smallest_container[i]
                    .and_then(|k| kept.get(k).map(|(_, a)| *a))
                    .unwrap_or(f64::INFINITY);
                if *area_j < cur_area {
                    smallest_container[i] = Some(j);
                }
            }
        }
    }

    let mut outer_idx_to_piece_idx: Vec<Option<usize>> = vec![None; n];
    let mut pieces: Vec<RegionPiece> = Vec::new();
    for (i, (ring, _)) in kept.iter().enumerate() {
        if containment_depth[i] % 2 == 0 {
            outer_idx_to_piece_idx[i] = Some(pieces.len());
            // Normalise outer to CCW (positive signed area).
            let outer = if signed_polygon_area(ring) >= 0.0 {
                ring.clone()
            } else {
                reverse_polygon(ring)
            };
            pieces.push(RegionPiece {
                outer,
                holes: Vec::new(),
            });
        }
    }
    for (i, (ring, _)) in kept.iter().enumerate() {
        if containment_depth[i] % 2 == 1 {
            if let Some(parent_idx) = smallest_container[i].and_then(|k| outer_idx_to_piece_idx[k])
            {
                // Normalise holes to CW (negative signed area).
                let hole = if signed_polygon_area(ring) <= 0.0 {
                    ring.clone()
                } else {
                    reverse_polygon(ring)
                };
                pieces[parent_idx].holes.push(hole);
            }
        }
    }
    pieces
}

/// Compute the pole of inaccessibility of a region described as a list of
/// already-classified [`RegionPiece`]s (one outer + holes per connected
/// component). Returns the (point, clearance) of the piece whose POI has
/// the largest minimum distance to its outer boundary and any of its holes.
///
/// Implemented in-house rather than via `polylabel-mini` because the latter
/// scores cells using distance to the exterior ring only — it ignores hole
/// boundaries when ranking candidates, so a point sitting right next to a
/// hole gets credited with the same clearance as a point far from any hole.
///
/// Returns `None` when there are no pieces with positive area or every
/// search-cell clearance comes out non-positive (degenerate input).
#[cfg(feature = "plotting")]
pub(crate) fn poi_with_holes(pieces: &[RegionPiece], precision: f64) -> Option<(Point, f64)> {
    fn min_dist_to_rings(px: f64, py: f64, rings: &[&[Point]]) -> f64 {
        let mut best = f64::INFINITY;
        for ring in rings {
            let n = ring.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let a = ring[i];
                let b = ring[(i + 1) % n];
                let dx = b.x() - a.x();
                let dy = b.y() - a.y();
                let len2 = dx * dx + dy * dy;
                let t = if len2 > 0.0 {
                    (((px - a.x()) * dx + (py - a.y()) * dy) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let qx = a.x() + t * dx;
                let qy = a.y() + t * dy;
                let d = ((px - qx).powi(2) + (py - qy).powi(2)).sqrt();
                if d < best {
                    best = d;
                }
            }
        }
        best
    }

    /// Signed distance: positive when (px, py) is inside `outer` and outside
    /// every hole; negative otherwise. Magnitude is min distance to any ring.
    fn signed_clearance(px: f64, py: f64, piece: &RegionPiece) -> f64 {
        let mut all: Vec<&[Point]> = Vec::with_capacity(1 + piece.holes.len());
        all.push(piece.outer.vertices());
        for h in &piece.holes {
            all.push(h.vertices());
        }
        let dist = min_dist_to_rings(px, py, &all);
        let probe = Point::new(px, py);
        let in_outer = point_in_polygon(&probe, &piece.outer);
        let in_any_hole = piece.holes.iter().any(|h| point_in_polygon(&probe, h));
        if in_outer && !in_any_hole {
            dist
        } else {
            -dist
        }
    }

    let mut best_overall: Option<(Point, f64)> = None;
    for piece in pieces {
        let outer_verts = piece.outer.vertices();
        if outer_verts.len() < 3 {
            continue;
        }

        // Bounding box of the outer ring.
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for p in outer_verts {
            min_x = min_x.min(p.x());
            max_x = max_x.max(p.x());
            min_y = min_y.min(p.y());
            max_y = max_y.max(p.y());
        }
        let width = max_x - min_x;
        let height = max_y - min_y;
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        let cell_size = width.min(height);

        // Priority queue keyed by upper bound on a cell's clearance.
        #[derive(Copy, Clone)]
        struct Cell {
            x: f64,
            y: f64,
            h: f64,
            d: f64,  // signed clearance at the centre
            ub: f64, // upper bound on clearance anywhere in the cell
        }
        impl Eq for Cell {}
        impl PartialEq for Cell {
            fn eq(&self, other: &Self) -> bool {
                self.ub == other.ub
            }
        }
        impl PartialOrd for Cell {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Cell {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.ub
                    .partial_cmp(&other.ub)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let make_cell = |x: f64, y: f64, h: f64| -> Cell {
            let d = signed_clearance(x, y, piece);
            Cell {
                x,
                y,
                h,
                d,
                ub: d + h * std::f64::consts::SQRT_2,
            }
        };

        let mut queue: std::collections::BinaryHeap<Cell> = std::collections::BinaryHeap::new();
        let mut x = min_x;
        let h0 = cell_size / 2.0;
        while x < max_x {
            let mut y = min_y;
            while y < max_y {
                queue.push(make_cell(x + h0, y + h0, h0));
                y += cell_size;
            }
            x += cell_size;
        }
        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;
        let mut best = make_cell(cx, cy, 0.0);

        while let Some(cell) = queue.pop() {
            if cell.d > best.d {
                best = cell;
            }
            if cell.ub - best.d <= precision {
                continue;
            }
            let nh = cell.h / 2.0;
            queue.push(make_cell(cell.x - nh, cell.y - nh, nh));
            queue.push(make_cell(cell.x + nh, cell.y - nh, nh));
            queue.push(make_cell(cell.x - nh, cell.y + nh, nh));
            queue.push(make_cell(cell.x + nh, cell.y + nh, nh));
        }

        if best.d > 0.0 {
            let pt = Point::new(best.x, best.y);
            if best_overall.map(|(_, d)| best.d > d).unwrap_or(true) {
                best_overall = Some((pt, best.d));
            }
        }
    }
    best_overall
}

impl Default for RegionPolygons {
    fn default() -> Self {
        Self::new()
    }
}

/// Decomposes fitted shapes into exclusive [`RegionPiece`]s, one entry per
/// non-empty set combination.
///
/// # Algorithm
///
/// 1. Convert each shape to a polygon at `n_vertices` resolution.
/// 2. Discover candidate region masks sparsely from the actual fitted
///    geometry (via the shape-boundary intersection points), instead of
///    enumerating the full `2^n - 1` power set. Only regions that can
///    geometrically be non-empty are considered, so the cost scales with
///    the number of geometrically real regions and large `n` stays
///    tractable on sparse layouts.
/// 3. For each candidate, intersect the in-set polygons and subtract the
///    out-of-set polygons via repeated polygon clipping.
/// 4. Run the resulting flat ring list through the topological piece
///    classifier — each piece gets one CCW outer + zero or more CW holes,
///    with orientations normalised.
/// 5. Drop sliver pieces (see "Sliver filtering" below) and store the rest.
///
/// # Output guarantees
///
/// For every retained piece:
///
/// * `outer.area() > 0`, vertex count ≥ 3, winding is CCW.
/// * Each `hole` is strictly inside `outer`, with CW winding.
/// * Net area (`outer` minus `holes`) is positive and at least
///   `1e-3 ×` the largest piece in the diagram.
///
/// Renderers can therefore use SVG / Canvas defaults (`fill-rule: nonzero`)
/// and don't need to do their own ring classification — see [`RegionPiece`].
///
/// # Sliver filtering
///
/// 200-vertex polygonization (the typical default) produces tiny rounding
/// artifacts along seams between two shapes — e.g. a "B-only" region of
/// area ~0.02% of the largest piece when B is geometrically nested in A
/// but their polygonal boundaries don't agree exactly. These slivers
/// would otherwise mislead label anchors, produce duplicate strokes, and
/// expose binding authors to noise. The threshold is **net piece area
/// below `1e-3 ×` the largest single piece in the diagram**; smaller
/// pieces are dropped before insertion. This is scale-invariant (large
/// and small diagrams behave the same) and stable against `n_vertices`.
///
/// Driven by the actual fitted shapes, not the spec, so unexpected
/// overlaps the optimiser produces are still visualised.
///
/// # Arguments
///
/// * `shapes` - The fitted diagram shapes (one per set).
/// * `set_names` - Set names in the same order as `shapes`.
/// * `_spec` - The diagram specification (unused at present; reserved for
///   future use such as spec-driven hole hints).
/// * `n_vertices` - Resolution used when polygonizing each shape. Higher =
///   smoother boundaries at higher clipping cost. The eulerr default is 200.
///
/// # Examples
///
/// ```
/// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::plotting::decompose_regions;
///
/// let spec = DiagramSpecBuilder::new()
///     .set("A", 5.0)
///     .set("B", 3.0)
///     .intersection(&["A", "B"], 1.0)
///     .input_type(InputType::Exclusive)
///     .build()
///     .unwrap();
///
/// let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
/// let shapes: Vec<_> = spec.set_names().iter()
///     .map(|name| *layout.shape_for_set(name).unwrap())
///     .collect();
///
/// let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);
/// ```
pub fn decompose_regions<S: DiagramShape + Polygonize>(
    shapes: &[S],
    set_names: &[String],
    _spec: &DiagramSpec,
    n_vertices: usize,
) -> RegionPolygons {
    /// Pieces whose net area is below this fraction of the largest piece in
    /// the diagram are treated as polygonization artifacts and dropped.
    /// Calibrated against the eulerr-comparison case (B/C/D nested in A,
    /// 200-vertex polygonization) where seam slivers come in at ~0.02% of
    /// the largest piece.
    const SLIVER_RELATIVE_THRESHOLD: f64 = 1e-3;

    if shapes.is_empty() {
        return RegionPolygons::new();
    }

    // Convert all shapes to polygons
    let shape_polygons: Vec<Polygon> = shapes.iter().map(|s| s.polygonize(n_vertices)).collect();

    let mut raw: Vec<(Combination, Vec<RegionPiece>)> = Vec::new();

    let n = shapes.len();

    // Sparse region discovery: walk only candidate masks the actual fitted
    // geometry can populate. Avoids the 2^n power-set scan that the previous
    // implementation required and that put a hard practical ceiling on `n`.
    let intersections = collect_intersections_generic(shapes, n);
    let mut region_masks = discover_regions(shapes, &intersections, n);
    region_masks.sort_unstable();

    for mask in region_masks {
        let set_indices_in_combo: Vec<usize> = (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();

        if set_indices_in_combo.is_empty() {
            continue;
        }

        // Start with the first shape in the combination
        let mut current_polygons = vec![shape_polygons[set_indices_in_combo[0]].clone()];

        // Intersect with remaining shapes that should be present
        for &idx in &set_indices_in_combo[1..] {
            current_polygons = polygon_clip_many(
                &current_polygons,
                &shape_polygons[idx],
                ClipOperation::Intersection,
            );

            if current_polygons.is_empty() {
                break;
            }
        }

        // Skip if intersection is empty
        if current_polygons.is_empty() {
            continue;
        }

        // Subtract all shapes that should NOT be present
        for (idx, _) in shapes.iter().enumerate() {
            if !set_indices_in_combo.contains(&idx) {
                current_polygons = polygon_clip_many(
                    &current_polygons,
                    &shape_polygons[idx],
                    ClipOperation::Difference,
                );

                if current_polygons.is_empty() {
                    break;
                }
            }
        }

        // Classify the clipper output (flat list of rings) into connected
        // pieces with explicit outer + holes structure. Skip empty.
        let pieces = classify_into_pieces(current_polygons);
        if pieces.is_empty() {
            continue;
        }

        let combo_sets: Vec<&str> = set_indices_in_combo
            .iter()
            .map(|&i| set_names[i].as_str())
            .collect();
        let combination = Combination::new(&combo_sets);

        raw.push((combination, pieces));
    }

    // Sliver filtering: drop pieces below `SLIVER_RELATIVE_THRESHOLD` of the
    // largest piece anywhere in the diagram. This eliminates the tiny
    // "B-only" shards produced by 200-vertex polygonization where B is
    // geometrically nested in A (their boundaries don't agree exactly along
    // the seam at B's right edge), which would otherwise mislead label
    // anchors and stroke rendering.
    let max_piece_area = raw
        .iter()
        .flat_map(|(_, pieces)| pieces.iter().map(|p| p.area()))
        .fold(0.0_f64, f64::max);
    let min_keep = max_piece_area * SLIVER_RELATIVE_THRESHOLD;

    let mut result = RegionPolygons::new();
    for (combo, pieces) in raw {
        let kept: Vec<RegionPiece> = pieces
            .into_iter()
            .filter(|p| p.area() >= min_keep)
            .collect();
        if !kept.is_empty() {
            result.insert(combo, kept);
        }
    }

    result
}

/// Collect pairwise boundary intersection points across arbitrary closed
/// shapes. Uses only the `Closed` trait, so it works for any `DiagramShape`
/// implementor — including future shapes that don't have a hand-rolled
/// `collect_intersections_*` helper.
///
/// The resulting `IntersectionPoint` list is what `discover_regions` consumes
/// to figure out which region masks the geometry can actually populate.
fn collect_intersections_generic<S: Closed>(shapes: &[S], n_sets: usize) -> Vec<IntersectionPoint> {
    let mut intersections = Vec::new();

    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let pts = shapes[i].intersection_points(&shapes[j]);
            for point in pts {
                let mut adopters = vec![i, j];
                for (k, shape) in shapes.iter().enumerate() {
                    if k != i && k != j && shape.contains_point(&point) {
                        adopters.push(k);
                    }
                }
                adopters.sort_unstable();

                intersections.push(IntersectionPoint::new(point, (i, j), adopters));
            }
        }
    }

    intersections
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Circle;
    use crate::spec::{DiagramSpecBuilder, InputType};

    #[test]
    fn test_decompose_disjoint_circles_skips_pairwise() {
        // Two disjoint circles. The previous power-set scanner would still
        // walk the {A,B} mask and only discover emptiness via polygon
        // clipping. The sparse path should skip {A,B} entirely (no boundary
        // intersection points, no containment), keeping cost proportional to
        // the number of geometrically real regions.
        let circles = [
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(10.0, 0.0), 1.0),
        ];
        let set_names = vec!["A".to_string(), "B".to_string()];
        let spec = DiagramSpecBuilder::new()
            .set("A", std::f64::consts::PI)
            .set("B", std::f64::consts::PI)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let regions = decompose_regions(&circles, &set_names, &spec, 64);

        // Exactly the two singleton regions, never the pair.
        assert_eq!(regions.len(), 2, "expected only A-only and B-only regions");
        assert!(regions.get(&Combination::new(&["A"])).is_some());
        assert!(regions.get(&Combination::new(&["B"])).is_some());
        assert!(
            regions.get(&Combination::new(&["A", "B"])).is_none(),
            "disjoint pair should not appear in region polygons"
        );
    }

    #[test]
    fn test_decompose_many_disjoint_circles_scales_sparsely() {
        // 20 disjoint circles in a row. Power-set decomposition would walk
        // 2^20 - 1 ≈ 1M masks; the sparse path walks only the 20 singletons.
        // This test asserts both correctness (only singletons appear) and
        // implicitly that the call returns in well under a second.
        let n = 20;
        let circles: Vec<Circle> = (0..n)
            .map(|i| Circle::new(Point::new(10.0 * i as f64, 0.0), 1.0))
            .collect();
        let set_names: Vec<String> = (0..n).map(|i| format!("S{i}")).collect();

        let mut builder = DiagramSpecBuilder::new();
        for name in &set_names {
            builder = builder.set(name.clone(), std::f64::consts::PI);
        }
        let spec = builder.input_type(InputType::Exclusive).build().unwrap();

        let regions = decompose_regions(&circles, &set_names, &spec, 32);
        assert_eq!(regions.len(), n, "expected one region per disjoint circle");
    }

    #[test]
    fn test_decompose_two_squares() {
        // The plotting path is generic over any `DiagramShape + Polygonize`.
        // Square implements both, so polygon decomposition into per-region
        // polygons should work without any plotting-side changes.
        use crate::geometry::shapes::Square;

        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Square>::new(&spec).seed(42).fit().unwrap();
        let shapes: Vec<Square> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        // n_vertices is ignored by Square::polygonize (always 4 corners).
        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);
        assert!(!regions.is_empty(), "no regions decomposed");
        for (combo, polys) in regions.iter() {
            assert!(!polys.is_empty(), "Region {:?} should have polygons", combo);
        }
    }

    #[test]
    fn test_decompose_two_circles() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);

        // Should have regions for A-only, B-only, and A&B
        assert!(regions.len() >= 2); // At least A-only and B-only

        // Check that we have some polygons
        for (combo, polys) in regions.iter() {
            assert!(!polys.is_empty(), "Region {:?} should have polygons", combo);
        }
    }

    #[test]
    fn test_decompose_three_circles() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .set("C", 4.0)
            .intersection(&["A", "B"], 1.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["A", "C"], 1.0)
            .intersection(&["A", "B", "C"], 0.5)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(123).fit().unwrap();

        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);

        // Should have multiple regions
        assert!(regions.len() >= 3);
    }

    #[test]
    fn test_label_points_two_circles() {
        // Two overlapping circles should yield 3 non-empty regions: A-only,
        // B-only, A&B. Each should get a label point inside the region itself.
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);
        let labels = regions.label_points(0.01);

        // Every non-empty region in `regions` should appear in `labels`.
        for combo in regions.iter().map(|(c, _)| c) {
            assert!(
                labels.contains_key(combo),
                "Missing label point for region {:?}",
                combo
            );
        }

        // Each label point must sit inside the axis-aligned bounding box of
        // its region's largest piece's outer ring.
        for (combo, pieces) in regions.iter() {
            let label = labels.get(combo).unwrap();
            let largest = pieces
                .iter()
                .max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap())
                .unwrap();
            let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
            let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for v in largest.outer.vertices() {
                min_x = min_x.min(v.x());
                min_y = min_y.min(v.y());
                max_x = max_x.max(v.x());
                max_y = max_y.max(v.y());
            }
            assert!(
                label.x() >= min_x - 1e-9
                    && label.x() <= max_x + 1e-9
                    && label.y() >= min_y - 1e-9
                    && label.y() <= max_y + 1e-9,
                "Label for {:?} at ({:.3}, {:.3}) is outside its region's bounding box [{:.3}, {:.3}] x [{:.3}, {:.3}]",
                combo,
                label.x(),
                label.y(),
                min_x,
                max_x,
                min_y,
                max_y
            );
        }
    }

    #[test]
    fn test_set_label_points_two_circles() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 128);
        let set_labels = regions.set_label_points(spec.set_names(), 0.01);

        assert_eq!(set_labels.len(), 2);
        for name in ["A", "B"] {
            let label = set_labels.get(name).expect("missing set label");
            // The set label must lie inside the corresponding circle.
            let circle = layout.shape_for_set(name).unwrap();
            let dx = label.x() - circle.center().x();
            let dy = label.y() - circle.center().y();
            let r = circle.radius();
            assert!(
                dx * dx + dy * dy <= r * r + 1e-6,
                "set label for {} at ({:.3}, {:.3}) is outside circle (center=({:.3}, {:.3}), r={:.3})",
                name,
                label.x(),
                label.y(),
                circle.center().x(),
                circle.center().y(),
                r,
            );
        }
    }

    #[test]
    fn test_set_label_points_skips_absent_sets() {
        let regions = RegionPolygons::new();
        let names = vec!["A".to_string()];
        assert!(regions.set_label_points(&names, 0.01).is_empty());
    }

    #[test]
    fn test_label_points_empty() {
        let empty = RegionPolygons::new();
        assert!(empty.label_points(0.01).is_empty());
    }

    #[test]
    fn test_label_points_skips_zero_area_regions() {
        // A region composed only of a degenerate (zero-area) outer should be
        // omitted from the label map.
        let mut regions = RegionPolygons::new();
        let degenerate = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 0.0), // collinear → zero area
        ]);
        let piece = RegionPiece {
            outer: degenerate,
            holes: Vec::new(),
        };
        regions.insert(Combination::new(&["X"]), vec![piece]);

        assert!(regions.label_points(0.01).is_empty());
    }

    #[test]
    fn test_region_areas() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 128);
        let areas = regions.areas();

        // Total area should be close to sum of fitted values
        let total_area: f64 = areas.values().sum();
        let expected_total: f64 = spec.exclusive_areas().values().sum();

        // Allow some tolerance due to polygonization
        assert!(
            (total_area - expected_total).abs() < 0.5,
            "Total area {:.3} should be close to expected {:.3}",
            total_area,
            expected_total
        );
    }

    #[test]
    fn test_decompose_with_zero_spec_area() {
        // Regression test: When spec has zero area for a set (e.g., C),
        // but the fitted layout has non-zero area, we should still generate
        // polygons for all possible regions, even if after subtraction some
        // regions are empty.
        let spec = DiagramSpecBuilder::new()
            .set("A", 3.0)
            .set("B", 5.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // Spec should have C with zero area
        let c_combo = crate::spec::Combination::new(&["C"]);
        assert!(
            spec.exclusive_areas().get(&c_combo).copied().unwrap_or(0.0) < 1e-10,
            "Spec should have zero area for C-only"
        );

        // Fit the layout - this will create a non-zero ellipse for C
        use crate::geometry::shapes::Ellipse;
        let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();

        let shapes: Vec<Ellipse> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        // Decompose regions
        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);

        // For this particular configuration, C is fully contained within
        // A&B&C intersection, so there won't be a C-only region. But we
        // should verify that regions involving C are present.
        let abc_combo = crate::spec::Combination::new(&["A", "B", "C"]);
        let abc_polygons = regions.get(&abc_combo);
        assert!(abc_polygons.is_some(), "Should have polygons for A&B&C");

        // Verify total area is reasonable
        let total_area: f64 = regions.areas().values().sum();

        // The total should be close to sum of individual shapes minus overlaps
        assert!(
            total_area > 5.0,
            "Total area should be substantial, got {:.3}",
            total_area
        );
    }
}
