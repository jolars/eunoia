//! Shape implementations for Euler and Venn diagrams.
//!
//! This module defines composable traits for geometric operations and diagram shapes.
//! The trait system is decomposed to allow code reuse across different geometric types:
//!
//! - Basic geometric properties: `Area`, `Centroid`, `Perimeter`, `BoundingBox`
//! - Spatial relations between same-type objects: `Distance`, `Closed`
//! - Diagram-specific operations: `DiagramShape` (composes all of the above)

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;

/// Trait for objects that have a computable distance to another object of the same type.
///
/// This can be implemented by both shapes (Circle, Point, LineSegment, etc.) to enable
/// generic distance-based algorithms.
pub trait Distance<Rhs = Self> {
    /// Computes the minimum distance between this object and another.
    /// For shapes, this is the distance between boundaries (0.0 if overlapping).
    /// For points, this is the Euclidean distance.
    fn distance(&self, other: &Rhs) -> f64;
}

/// Trait for objects that have a measurable area.
pub trait Area {
    /// Returns the area of the object.
    fn area(&self) -> f64;
}

/// Trait for objects that have a definable centroid (center of mass).
pub trait Centroid {
    /// Returns the centroid (center point) as a Point.
    fn centroid(&self) -> Point;
}

/// Trait for objects that have a measurable perimeter.
pub trait Perimeter {
    /// Computes the perimeter (boundary length) of the object.
    fn perimeter(&self) -> f64;
}

/// Trait for objects that can be bounded by a rectangle.
pub trait BoundingBox {
    /// Computes the axis-aligned bounding box as a Rectangle.
    fn bounding_box(&self) -> Rectangle;
}

/// Trait for spatial relationships between objects of the same type.
///
/// This covers containment, intersection testing, and computing intersection areas/points.
pub trait Closed: Sized + Area + BoundingBox + Perimeter + Centroid {
    /// Checks if this object contains another object entirely within its boundaries.
    fn contains(&self, other: &Self) -> bool;

    /// Checks if a point is inside the object (on the boundary or interior).
    fn contains_point(&self, point: &Point) -> bool;

    /// Checks if this object intersects with another object.
    fn intersects(&self, other: &Self) -> bool;

    /// Computes the area of intersection between this object and another.
    fn intersection_area(&self, other: &Self) -> f64;

    /// Computes the points where this object's boundary intersects another's.
    fn intersection_points(&self, other: &Self) -> Vec<Point>;
}

/// Trait for shapes that can be used in Euler and Venn diagrams.
///
/// This is a supertrait that combines all the geometric capabilities needed for
/// diagram construction and optimization. Types implementing this trait can be used
/// with the diagram fitter.
///
/// # Two parameter encodings
///
/// Each shape exposes its parameters in two encodings:
///
/// - **Geometric** (`to_params` / `from_params`): the human-readable, FFI-friendly
///   encoding. Linear values throughout.
///   - Circle: `[x, y, r]`
///   - Square: `[x, y, side]`
///   - Ellipse: `[x, y, a, b, phi]` (semi-axes in **linear** space)
///
///   This is what almost every external consumer wants. `to_params()` returns
///   the same numbers a human would write down, and `from_params()` accepts
///   the same.
///
/// - **Optimizer** (`to_optimizer_params` / `from_optimizer_params` /
///   `optimizer_params_from_circle`): the encoding the unconstrained LM solver
///   consumes. Identical to the geometric encoding for circles and squares;
///   for ellipses, the semi-axes are stored in **log space**
///   (`[x, y, ln(a), ln(b), phi]`) so the optimizer can range freely without
///   a clamp on the positive-axis manifold.
///
///   Most callers should not need this — it exists for the fitter and for
///   experiments that plug straight into argmin.
///
/// # Type Parameters
///
/// The associated methods for parameter conversion enable the optimization process:
/// 1. Initial layout uses circles (MDS algorithm)
/// 2. Circle parameters are converted to shape-specific optimizer parameters via
///    `optimizer_params_from_circle`
/// 3. Final optimization operates on shape-specific optimizer parameters
/// 4. Shapes are constructed from optimized parameters via `from_optimizer_params`
///    (or, for external callers, `from_params` after converting to geometric)
pub trait DiagramShape: Closed {
    /// Compute all exclusive regions and their areas from a collection of shapes.
    ///
    /// This method should use exact geometric computation for the shape type.
    /// Returns a map from RegionMask (bit representation) to exclusive area.
    ///
    /// This is used during optimization to compute loss functions.
    fn compute_exclusive_regions(
        shapes: &[Self],
    ) -> std::collections::HashMap<crate::geometry::diagram::RegionMask, f64>
    where
        Self: Sized;

    /// Convert initial circle parameters to shape-specific optimizer parameters.
    ///
    /// Takes circle parameters `(x, y, radius)` (always linear) and returns the
    /// optimizer-encoded parameter vector for this shape. See the trait docs
    /// for the per-shape encoding — in particular, ellipse semi-axes are
    /// returned in **log space**.
    ///
    /// - Circle: `[x, y, r]`
    /// - Square: `[x, y, r·√π]` (equal-area mapping)
    /// - Ellipse: `[x, y, ln(r), ln(r), 0.0]` (circle ≡ a=b=r in log space)
    fn optimizer_params_from_circle(x: f64, y: f64, radius: f64) -> Vec<f64>
    where
        Self: Sized;

    /// Center-to-center distance that would yield `target_overlap` between
    /// two shapes whose set areas are `area_i` and `area_j`. Used by the MDS
    /// initial-layout phase, which optimises 2D positions against a precomputed
    /// scalar distance matrix; this method is the shape-specific overlap-to-
    /// distance inversion that fills that matrix.
    ///
    /// Each shape inverts its own pairwise-overlap formula along a canonical
    /// direction (since axis-aligned squares overlap as a 2D function of
    /// `(|dx|, |dy|)`, the "distance" interpretation is shape-dependent):
    /// - Circle: closed-form lens-area inversion.
    /// - Ellipse: same as Circle (warm-start is a circle of the same area).
    /// - Square (axis-aligned): inverts overlap along the diagonal `|dx| = |dy|`,
    ///   giving `d = √2 · ((s_i+s_j)/2 − √target_overlap)`.
    fn mds_target_distance(
        area_i: f64,
        area_j: f64,
        target_overlap: f64,
    ) -> Result<f64, crate::error::DiagramError>
    where
        Self: Sized;

    /// Get the number of parameters needed for this shape type. The same
    /// length applies to both the geometric and optimizer encodings.
    ///
    /// - Circle: 3 (x, y, r)
    /// - Square: 3 (x, y, side)
    /// - Ellipse: 5 (x, y, a, b, phi)
    fn n_params() -> usize
    where
        Self: Sized;

    /// Construct a shape from its **geometric** parameter representation.
    ///
    /// Inverse of [`to_params`](Self::to_params). The encoding is the
    /// human-readable one — linear semi-axes for ellipses, linear radius for
    /// circles, linear side for squares. See the trait docs for the per-shape
    /// layout.
    ///
    /// This is the entry point external callers (FFI, tests, examples) should
    /// reach for. The optimizer uses [`from_optimizer_params`](Self::from_optimizer_params)
    /// instead.
    fn from_params(params: &[f64]) -> Self
    where
        Self: Sized;

    /// Convert the shape to its **geometric** parameter representation.
    ///
    /// Inverse of [`from_params`](Self::from_params). The encoding matches the
    /// per-shape geometric accessors:
    ///
    /// - Circle: `[x, y, r]`
    /// - Square: `[x, y, side]`
    /// - Ellipse: `[x, y, a, b, phi]` (semi-axes in linear space)
    ///
    /// External callers (FFI bindings, tests, anything that wants values it
    /// can show a user) should use this — **not** `to_optimizer_params`.
    fn to_params(&self) -> Vec<f64>;

    /// Construct a shape from optimizer-encoded parameters.
    ///
    /// The optimizer encoding is identical to the geometric encoding for
    /// Circle and Square; for Ellipse the semi-axes are in log space
    /// (`[x, y, ln(a), ln(b), phi]`). External callers should generally use
    /// [`from_params`](Self::from_params) instead — this method is for the
    /// fitter and for callers that have raw optimizer state in hand.
    ///
    /// The default implementation delegates to `from_params`, which is
    /// correct for any shape whose two encodings coincide; shapes with a
    /// distinct optimizer encoding (e.g. Ellipse) override.
    fn from_optimizer_params(params: &[f64]) -> Self
    where
        Self: Sized,
    {
        Self::from_params(params)
    }

    /// Convert the shape to its optimizer parameter representation.
    ///
    /// Inverse of [`from_optimizer_params`](Self::from_optimizer_params); see
    /// that method for encoding details. External callers should generally
    /// use [`to_params`](Self::to_params) instead.
    ///
    /// The default implementation delegates to `to_params`, which is correct
    /// for any shape whose two encodings coincide; shapes with a distinct
    /// optimizer encoding (e.g. Ellipse) override.
    fn to_optimizer_params(&self) -> Vec<f64> {
        self.to_params()
    }

    /// Optional analytical gradient companion to `compute_exclusive_regions`.
    ///
    /// Returns `Some((exclusive_areas, gradients))` where each `gradients[mask]`
    /// is a length-`n_sets · Self::n_params()` vector of `∂A_excl[mask]/∂θ`,
    /// with `θ` ordered to match the flat parameter vector consumed by
    /// `from_optimizer_params`. The default implementation returns `None`,
    /// signalling to the optimiser that finite differences should be used
    /// instead.
    fn compute_exclusive_regions_with_gradient(
        _shapes: &[Self],
    ) -> Option<ExclusiveRegionsAndGradient>
    where
        Self: Sized,
    {
        None
    }

    /// Canonical Venn-diagram layout for `n` sets, or `None` if no canonical
    /// arrangement exists for this shape at this `n`.
    ///
    /// Implementations should return shapes at unit scale (radius/side ~1),
    /// so callers can rescale the layout to match a spec's footprint.
    /// A returned layout must be a true Venn diagram: every one of the
    /// `2ⁿ − 1` non-empty subsets must have positive area.
    ///
    /// Used by [`crate::venn::VennDiagram`] to produce canonical layouts
    /// without invoking the fitter.
    fn canonical_venn_layout(n: usize) -> Option<Vec<Self>>
    where
        Self: Sized,
    {
        let _ = n;
        None
    }
}

/// Pair of (exclusive areas, exclusive gradients) returned by
/// [`DiagramShape::compute_exclusive_regions_with_gradient`]. The gradient
/// vectors are flat, length `n_sets * Self::n_params()`, mirroring the
/// optimiser's parameter layout.
pub type ExclusiveRegionsAndGradient = (
    std::collections::HashMap<crate::geometry::diagram::RegionMask, f64>,
    std::collections::HashMap<crate::geometry::diagram::RegionMask, Vec<f64>>,
);

/// Trait for converting shapes to polygons for visualization.
///
/// This trait allows analytical shapes (circles, ellipses) to be converted
/// to discrete polygon representations for plotting and export.
pub trait Polygonize {
    /// Convert the shape to a polygon with the specified number of vertices.
    ///
    /// # Arguments
    ///
    /// * `n_vertices` - Number of vertices in the resulting polygon (minimum 3)
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Circle;
    /// use eunoia::geometry::primitives::Point;
    /// use eunoia::geometry::traits::Polygonize;
    ///
    /// let circle = Circle::new(Point::new(0.0, 0.0), 5.0);
    /// let polygon = circle.polygonize(64);
    /// assert_eq!(polygon.vertices().len(), 64);
    /// ```
    fn polygonize(&self, n_vertices: usize) -> crate::geometry::shapes::Polygon;
}

/// Compute the bounding box for a collection of shapes.
pub fn bounding_box<S: BoundingBox>(shapes: &[S]) -> Rectangle {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;

    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for shape in shapes {
        let points = shape.bounding_box().to_points();
        min_x = min_x.min(points.0.x());
        min_y = min_y.min(points.0.y());
        max_x = max_x.max(points.1.x());
        max_y = max_y.max(points.1.y());
    }

    Rectangle::from_corners(Point::new(min_x, min_y), Point::new(max_x, max_y))
}
