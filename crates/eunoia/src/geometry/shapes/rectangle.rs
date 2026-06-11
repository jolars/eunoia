//! Axis-aligned rectangle shape implementation.
//!
//! `Rectangle` doubles as a primitive (used by every shape's
//! [`BoundingBox`] impl) and as a fittable [`DiagramShape`]. It is the
//! 4-parameter sibling of [`Square`](crate::geometry::shapes::Square) — a
//! width and a height, no rotation. The n-way intersection of axis-aligned
//! rectangles is itself one axis-aligned rectangle, so
//! [`Rectangle::compute_exclusive_regions`] is exact in closed form with no
//! polygon-clipping library required.
//!
//! Two parameter encodings are exposed:
//!
//! - **Geometric** (`to_params` / `from_params`): `[x, y, w, h]` — what FFI
//!   callers and humans see.
//! - **Optimizer** (`to_optimizer_params` / `from_optimizer_params`):
//!   `[x, y, ln(w·h), ln(w/h)]`. The 45° rotation in log-space decouples
//!   the area direction (pinned by singleton targets) from the aspect-ratio
//!   direction (only constrained by overlap geometry), giving the unbounded
//!   LM/CMA-ES solver a much better-conditioned Hessian than `[ln w, ln h]`
//!   would.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::geometry::diagram::{
    IntersectionPoint, RegionMask, discover_regions, mask_to_indices, to_exclusive_areas,
    to_exclusive_areas_and_gradients,
};
use crate::geometry::primitives::{Bounds, Point};
use crate::geometry::shapes::Polygon;
use crate::geometry::traits::{
    Area, BoundingBox, Centroid, Closed, DiagramShape, Distance, ExclusiveRegionsAndGradient,
    Perimeter, Polygonize,
};

/// An axis-aligned rectangle defined by a center point, width, and height.
///
/// The rectangle's edges are parallel to the x and y axes. This simplifies
/// many geometric computations compared to rotated rectangles.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Rectangle;
/// use eunoia::geometry::traits::Area;
/// use eunoia::geometry::traits::Closed;
/// use eunoia::geometry::primitives::Point;
///
/// let r1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0);
/// let r2 = Rectangle::new(Point::new(3.0, 0.0), 2.0, 3.0);
///
/// let area1 = r1.area();
/// let overlap = r1.intersection_area(&r2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangle {
    center: Point,
    width: f64,
    height: f64,
}

#[allow(dead_code)]
impl Rectangle {
    /// Creates a new axis-aligned rectangle with the specified center, width, and height.
    ///
    /// # Arguments
    ///
    /// * `center` - The center point of the rectangle
    /// * `width` - The width of the rectangle (must be positive)
    /// * `height` - The height of the rectangle (must be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Rectangle;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let rect = Rectangle::new(Point::new(1.0, 2.0), 4.0, 3.0);
    /// ```
    pub fn new(center: Point, width: f64, height: f64) -> Self {
        Rectangle {
            center,
            width,
            height,
        }
    }

    /// Fallible constructor: returns
    /// [`crate::error::DiagramError::InvalidShapeParameter`] when `width <= 0`
    /// or `height <= 0`. Use this when constructing rectangles from untrusted
    /// input (e.g. across an FFI boundary). Unlike [`Rectangle::new`] (which
    /// is also used as a bounding-box primitive and accepts zero-sized boxes
    /// by design), `try_new` enforces the diagram-shape invariant.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Rectangle;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// assert!(Rectangle::try_new(Point::new(0.0, 0.0), 2.0, 1.0).is_ok());
    /// assert!(Rectangle::try_new(Point::new(0.0, 0.0), 0.0, 1.0).is_err());
    /// assert!(Rectangle::try_new(Point::new(0.0, 0.0), 1.0, -1.0).is_err());
    /// ```
    pub fn try_new(
        center: Point,
        width: f64,
        height: f64,
    ) -> Result<Self, crate::error::DiagramError> {
        if width <= 0.0 {
            return Err(crate::error::DiagramError::InvalidShapeParameter {
                shape: "Rectangle",
                param: "width",
                value: width,
            });
        }
        if height <= 0.0 {
            return Err(crate::error::DiagramError::InvalidShapeParameter {
                shape: "Rectangle",
                param: "height",
                value: height,
            });
        }
        Ok(Rectangle {
            center,
            width,
            height,
        })
    }

    /// Create a rectangle from two corner points (bottom-left and top-right).
    pub fn from_corners(bottom_left: Point, top_right: Point) -> Self {
        let center_x = (bottom_left.x() + top_right.x()) / 2.0;
        let center_y = (bottom_left.y() + top_right.y()) / 2.0;

        let width = top_right.x() - bottom_left.x();
        let height = top_right.y() - bottom_left.y();

        Rectangle::new(Point::new(center_x, center_y), width, height)
    }

    /// Returns a reference to the rectangle's center point.
    pub fn center(&self) -> &Point {
        &self.center
    }

    /// Returns the rectangle's width.
    pub fn width(&self) -> f64 {
        self.width
    }

    /// Returns the rectangle's height.
    pub fn height(&self) -> f64 {
        self.height
    }

    /// Sets the center of the rectangle.
    pub fn set_center(&mut self, center: Point) {
        self.center = center;
    }

    /// Returns the bottom-left and top-right corner points of the rectangle.
    pub fn to_points(self) -> (Point, Point) {
        self.bounds().to_points()
    }

    /// Returns the four corner points of the rectangle.
    pub fn corners(&self) -> [Point; 4] {
        let Bounds {
            x_min,
            x_max,
            y_min,
            y_max,
        } = self.bounds();
        [
            Point::new(x_min, y_min),
            Point::new(x_max, y_min),
            Point::new(x_max, y_max),
            Point::new(x_min, y_max),
        ]
    }
}

impl Area for Rectangle {
    /// Computes the area of the rectangle using the formula A = width × height.
    fn area(&self) -> f64 {
        self.width * self.height
    }
}

impl Perimeter for Rectangle {
    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }
}

impl BoundingBox for Rectangle {
    fn bounds(&self) -> Bounds {
        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;
        Bounds::new(
            self.center.x() - half_width,
            self.center.x() + half_width,
            self.center.y() - half_height,
            self.center.y() + half_height,
        )
    }
}

impl Centroid for Rectangle {
    /// Returns the centroid (center point) of the rectangle.
    fn centroid(&self) -> Point {
        self.center
    }
}

impl Distance for Rectangle {
    /// Computes the minimum distance between the boundaries of two rectangles.
    ///
    /// Returns 0.0 if the rectangles overlap or touch.
    fn distance(&self, other: &Self) -> f64 {
        let Bounds {
            x_min: x1_min,
            x_max: x1_max,
            y_min: y1_min,
            y_max: y1_max,
        } = self.bounds();
        let Bounds {
            x_min: x2_min,
            x_max: x2_max,
            y_min: y2_min,
            y_max: y2_max,
        } = other.bounds();

        let dx = if x1_max < x2_min {
            x2_min - x1_max
        } else if x2_max < x1_min {
            x1_min - x2_max
        } else {
            0.0
        };

        let dy = if y1_max < y2_min {
            y2_min - y1_max
        } else if y2_max < y1_min {
            y1_min - y2_max
        } else {
            0.0
        };

        (dx * dx + dy * dy).sqrt()
    }
}

#[allow(dead_code)]
impl Closed for Rectangle {
    fn contains(&self, other: &Self) -> bool {
        let Bounds {
            x_min: x1_min,
            x_max: x1_max,
            y_min: y1_min,
            y_max: y1_max,
        } = self.bounds();
        let Bounds {
            x_min: x2_min,
            x_max: x2_max,
            y_min: y2_min,
            y_max: y2_max,
        } = other.bounds();

        x2_min >= x1_min && x2_max <= x1_max && y2_min >= y1_min && y2_max <= y1_max
    }

    /// Checks if a point is inside the rectangle (including the boundary).
    fn contains_point(&self, point: &Point) -> bool {
        let Bounds {
            x_min,
            x_max,
            y_min,
            y_max,
        } = self.bounds();
        point.x() >= x_min && point.x() <= x_max && point.y() >= y_min && point.y() <= y_max
    }

    fn intersects(&self, other: &Self) -> bool {
        let Bounds {
            x_min: x1_min,
            x_max: x1_max,
            y_min: y1_min,
            y_max: y1_max,
        } = self.bounds();
        let Bounds {
            x_min: x2_min,
            x_max: x2_max,
            y_min: y2_min,
            y_max: y2_max,
        } = other.bounds();

        !(x1_max < x2_min || x2_max < x1_min || y1_max < y2_min || y2_max < y1_min)
    }

    /// Computes the area of intersection between two axis-aligned rectangles.
    ///
    /// Returns 0 if rectangles don't overlap.
    fn intersection_area(&self, other: &Self) -> f64 {
        let Bounds {
            x_min: x1_min,
            x_max: x1_max,
            y_min: y1_min,
            y_max: y1_max,
        } = self.bounds();
        let Bounds {
            x_min: x2_min,
            x_max: x2_max,
            y_min: y2_min,
            y_max: y2_max,
        } = other.bounds();

        let x_overlap = (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0.0);
        let y_overlap = (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0.0);

        x_overlap * y_overlap
    }

    /// Computes the intersection points between two rectangles.
    ///
    /// For axis-aligned rectangles, intersection points are at the corners
    /// of the overlapping region.
    fn intersection_points(&self, other: &Self) -> Vec<Point> {
        if !self.intersects(other) {
            return vec![];
        }

        let Bounds {
            x_min: x1_min,
            x_max: x1_max,
            y_min: y1_min,
            y_max: y1_max,
        } = self.bounds();
        let Bounds {
            x_min: x2_min,
            x_max: x2_max,
            y_min: y2_min,
            y_max: y2_max,
        } = other.bounds();

        let x_min = x1_min.max(x2_min);
        let x_max = x1_max.min(x2_max);
        let y_min = y1_min.max(y2_min);
        let y_max = y1_max.min(y2_max);

        if x_min >= x_max || y_min >= y_max {
            return vec![];
        }

        vec![
            Point::new(x_min, y_min),
            Point::new(x_max, y_min),
            Point::new(x_max, y_max),
            Point::new(x_min, y_max),
        ]
    }
}

/// Pairwise edge-crossings between axis-aligned rectangles for the
/// region-discovery pass. Mirrors
/// [`crate::geometry::shapes::square::collect_intersections_square`] and
/// the circle / ellipse equivalents — same `IntersectionPoint` shape.
fn collect_intersections_rectangle(rects: &[Rectangle], n_sets: usize) -> Vec<IntersectionPoint> {
    let mut intersections = Vec::new();
    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let pts = rects[i].intersection_points(&rects[j]);
            for point in pts {
                let mut adopters = vec![i, j];
                for (k, r) in rects.iter().enumerate().take(n_sets) {
                    if k != i && k != j && r.contains_point(&point) {
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

/// Companion to [`Rectangle::compute_exclusive_regions`] that also returns
/// the analytical gradient of each exclusive area w.r.t. the flat
/// **optimizer** parameter vector
/// `[x₀, y₀, u₀, v₀, x₁, y₁, u₁, v₁, …]` where
/// `u_i = ln(w_i · h_i)` and `v_i = ln(w_i / h_i)`.
///
/// Per-region geometric gradient (in `[x, y, w, h]` per shape):
///
/// ```text
/// ∂A/∂x_{i_L} = −dy   ∂A/∂x_{i_R} = +dy
/// ∂A/∂y_{i_B} = −dx   ∂A/∂y_{i_T} = +dx
/// ∂A/∂w_{i_L} = dy/2  ∂A/∂w_{i_R} = dy/2
/// ∂A/∂h_{i_B} = dx/2  ∂A/∂h_{i_T} = dx/2
/// ```
///
/// At a tie (multiple shapes share the extremum on a side), the side's
/// contribution is split equally among the tied shapes (subgradient at
/// the non-smooth point — same approach as
/// [`Square::compute_exclusive_regions_with_gradient`]). Width sides only
/// contribute to the width derivative; height sides only to the height
/// derivative.
///
/// The geometric `(∂A/∂w_i, ∂A/∂h_i)` pair is then chain-ruled into the
/// optimizer encoding:
///
/// ```text
/// ∂A/∂u_i = (∂A/∂w_i · w_i + ∂A/∂h_i · h_i) / 2
/// ∂A/∂v_i = (∂A/∂w_i · w_i − ∂A/∂h_i · h_i) / 2
/// ```
///
/// (`x` and `y` derivatives pass through unchanged.) When `dx ≤ 0` or
/// `dy ≤ 0` the area is clamped to 0 and the gradient for that region is
/// zero; the shared inclusion-exclusion combiner
/// [`to_exclusive_areas_and_gradients`] zeroes gradients for post-IE
/// clamped-negative areas as well.
fn compute_exclusive_regions_with_gradient_rectangles(
    shapes: &[Rectangle],
) -> ExclusiveRegionsAndGradient {
    let n_sets = shapes.len();
    let n_params_per_shape = 4;
    let n_params = n_sets * n_params_per_shape;

    let intersections = collect_intersections_rectangle(shapes, n_sets);
    let regions = discover_regions(shapes, &intersections, n_sets);

    let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
    let mut overlapping_grads: HashMap<RegionMask, Vec<f64>> = HashMap::new();

    for &mask in &regions {
        let indices = mask_to_indices(mask, n_sets);

        let mut x_min = f64::NEG_INFINITY;
        let mut x_max = f64::INFINITY;
        let mut y_min = f64::NEG_INFINITY;
        let mut y_max = f64::INFINITY;
        for &i in &indices {
            let Bounds {
                x_min: a,
                x_max: b,
                y_min: c,
                y_max: d,
            } = shapes[i].bounds();
            if a > x_min {
                x_min = a;
            }
            if b < x_max {
                x_max = b;
            }
            if c > y_min {
                y_min = c;
            }
            if d < y_max {
                y_max = d;
            }
        }

        let dx_raw = x_max - x_min;
        let dy_raw = y_max - y_min;
        let dx = dx_raw.max(0.0);
        let dy = dy_raw.max(0.0);
        overlapping_areas.insert(mask, dx * dy);

        let mut grad = vec![0.0; n_params];
        if dx_raw > 0.0 && dy_raw > 0.0 {
            let mut tied_l: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_r: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_b: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_t: Vec<usize> = Vec::with_capacity(indices.len());
            for &i in &indices {
                let Bounds {
                    x_min: a,
                    x_max: b,
                    y_min: c,
                    y_max: d,
                } = shapes[i].bounds();
                #[allow(clippy::float_cmp)]
                {
                    if a == x_min {
                        tied_l.push(i);
                    }
                    if b == x_max {
                        tied_r.push(i);
                    }
                    if c == y_min {
                        tied_b.push(i);
                    }
                    if d == y_max {
                        tied_t.push(i);
                    }
                }
            }
            let w_l = 1.0 / tied_l.len() as f64;
            let w_r = 1.0 / tied_r.len() as f64;
            let w_b = 1.0 / tied_b.len() as f64;
            let w_t = 1.0 / tied_t.len() as f64;

            // Accumulate geometric gradient `[dA/dx, dA/dy, dA/dw, dA/dh]`
            // per shape, then chain-rule into optimizer space.
            let mut geom = vec![0.0_f64; n_params]; // ordered [x, y, w, h] per shape

            for &i in &tied_l {
                geom[4 * i] -= dy * w_l;
                geom[4 * i + 2] += dy * 0.5 * w_l;
            }
            for &i in &tied_r {
                geom[4 * i] += dy * w_r;
                geom[4 * i + 2] += dy * 0.5 * w_r;
            }
            for &i in &tied_b {
                geom[4 * i + 1] -= dx * w_b;
                geom[4 * i + 3] += dx * 0.5 * w_b;
            }
            for &i in &tied_t {
                geom[4 * i + 1] += dx * w_t;
                geom[4 * i + 3] += dx * 0.5 * w_t;
            }

            for i in 0..n_sets {
                let w_i = shapes[i].width;
                let h_i = shapes[i].height;
                let da_dw = geom[4 * i + 2];
                let da_dh = geom[4 * i + 3];
                grad[4 * i] = geom[4 * i];
                grad[4 * i + 1] = geom[4 * i + 1];
                // u = ln(w·h): ∂A/∂u = (∂A/∂w · w + ∂A/∂h · h) / 2
                grad[4 * i + 2] = 0.5 * (da_dw * w_i + da_dh * h_i);
                // v = ln(w/h): ∂A/∂v = (∂A/∂w · w − ∂A/∂h · h) / 2
                grad[4 * i + 3] = 0.5 * (da_dw * w_i - da_dh * h_i);
            }
        }
        overlapping_grads.insert(mask, grad);
    }

    to_exclusive_areas_and_gradients(&overlapping_areas, &overlapping_grads, n_params)
}

/// Clipped exclusive areas: each region's intersection with `container` is
/// itself an axis-aligned rectangle (the n-way intersection of axis-aligned
/// rectangles is one axis-aligned rectangle, and adding the container as one
/// more participant preserves that). Mask `0` is seeded with the container
/// area; inclusion-exclusion then produces
/// `complement = container.area − area(⋃ rectangles ∩ container)`.
pub(crate) fn compute_exclusive_regions_clipped_rectangles(
    shapes: &[Rectangle],
    container: &Rectangle,
) -> HashMap<RegionMask, f64> {
    let n_sets = shapes.len();
    let intersections = collect_intersections_rectangle(shapes, n_sets);
    let regions = discover_regions(shapes, &intersections, n_sets);

    let Bounds {
        x_min: cx_min,
        x_max: cx_max,
        y_min: cy_min,
        y_max: cy_max,
    } = container.bounds();
    let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
    overlapping_areas.insert(0, container.area());

    for &mask in &regions {
        let indices = mask_to_indices(mask, n_sets);
        let mut x_min = cx_min;
        let mut x_max = cx_max;
        let mut y_min = cy_min;
        let mut y_max = cy_max;
        for &i in &indices {
            let Bounds {
                x_min: a,
                x_max: b,
                y_min: c,
                y_max: d,
            } = shapes[i].bounds();
            if a > x_min {
                x_min = a;
            }
            if b < x_max {
                x_max = b;
            }
            if c > y_min {
                y_min = c;
            }
            if d < y_max {
                y_max = d;
            }
        }
        let dx = (x_max - x_min).max(0.0);
        let dy = (y_max - y_min).max(0.0);
        overlapping_areas.insert(mask, dx * dy);
    }

    to_exclusive_areas(&overlapping_areas)
}

/// Gradient companion to [`compute_exclusive_regions_clipped_rectangles`].
///
/// Returns `(exclusive_areas, exclusive_grads)` where each gradient vector has
/// length `n_sets · 4 + 4` and is laid out as
/// `[x₀, y₀, u₀, v₀, …, x_c, y_c, u_c, v_c]` — the trailing four entries are
/// the container's optimizer encoding (`u = ln(w·h)`, `v = ln(w/h)`).
///
/// Per region, the clipped intersection is one axis-aligned rectangle whose
/// four sides each bind to either a shape (mask member) or the container.
/// Geometric gradient (in `[x, y, w, h]` per shape and `[x, y, w, h]` for the
/// container, then chain-ruled into the optimizer encoding):
///
/// ```text
/// ∂A/∂x_{L} = −dy   ∂A/∂x_{R} = +dy
/// ∂A/∂y_{B} = −dx   ∂A/∂y_{T} = +dx
/// ∂A/∂w_{L} = dy/2  ∂A/∂w_{R} = dy/2
/// ∂A/∂h_{B} = dx/2  ∂A/∂h_{T} = dx/2
/// ```
///
/// At a tie (multiple shapes and/or the container share a side's extremum),
/// the side's contribution is split equally among the tied participants.
/// Chain rule into the optimizer encoding (same for shapes and container):
///
/// ```text
/// ∂A/∂u = (∂A/∂w · w + ∂A/∂h · h) / 2
/// ∂A/∂v = (∂A/∂w · w − ∂A/∂h · h) / 2
/// ```
///
/// Mask `0` is seeded with `container.area()` and a gradient
/// `∂(w·h)/∂(x_c, y_c, u_c, v_c) = [0, 0, w·h, 0]`. Inclusion-exclusion then
/// produces the complement and its gradient consistently with the per-mask
/// areas.
pub(crate) fn compute_exclusive_regions_clipped_with_gradient_rectangles(
    shapes: &[Rectangle],
    container: &Rectangle,
) -> ExclusiveRegionsAndGradient {
    let n_sets = shapes.len();
    let n_params_per_shape = 4;
    let container_off = n_sets * n_params_per_shape;
    let n_params = container_off + 4;

    let intersections = collect_intersections_rectangle(shapes, n_sets);
    let regions = discover_regions(shapes, &intersections, n_sets);

    let Bounds {
        x_min: cx_min,
        x_max: cx_max,
        y_min: cy_min,
        y_max: cy_max,
    } = container.bounds();
    let container_area = container.area();

    let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
    let mut overlapping_grads: HashMap<RegionMask, Vec<f64>> = HashMap::new();

    // Mask 0: complement seeded with container area + ∂(w·h)/∂u_c = w·h.
    overlapping_areas.insert(0, container_area);
    let mut zero_grad = vec![0.0; n_params];
    zero_grad[container_off + 2] = container_area;
    overlapping_grads.insert(0, zero_grad);

    for &mask in &regions {
        let indices = mask_to_indices(mask, n_sets);

        let mut x_min = cx_min;
        let mut x_max = cx_max;
        let mut y_min = cy_min;
        let mut y_max = cy_max;
        for &i in &indices {
            let Bounds {
                x_min: a,
                x_max: b,
                y_min: c,
                y_max: d,
            } = shapes[i].bounds();
            if a > x_min {
                x_min = a;
            }
            if b < x_max {
                x_max = b;
            }
            if c > y_min {
                y_min = c;
            }
            if d < y_max {
                y_max = d;
            }
        }

        let dx_raw = x_max - x_min;
        let dy_raw = y_max - y_min;
        let dx = dx_raw.max(0.0);
        let dy = dy_raw.max(0.0);
        overlapping_areas.insert(mask, dx * dy);

        let mut grad = vec![0.0; n_params];
        if dx_raw > 0.0 && dy_raw > 0.0 {
            let mut tied_l: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_r: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_b: Vec<usize> = Vec::with_capacity(indices.len());
            let mut tied_t: Vec<usize> = Vec::with_capacity(indices.len());
            for &i in &indices {
                let Bounds {
                    x_min: a,
                    x_max: b,
                    y_min: c,
                    y_max: d,
                } = shapes[i].bounds();
                #[allow(clippy::float_cmp)]
                {
                    if a == x_min {
                        tied_l.push(i);
                    }
                    if b == x_max {
                        tied_r.push(i);
                    }
                    if c == y_min {
                        tied_b.push(i);
                    }
                    if d == y_max {
                        tied_t.push(i);
                    }
                }
            }
            #[allow(clippy::float_cmp)]
            let tied_l_c = cx_min == x_min;
            #[allow(clippy::float_cmp)]
            let tied_r_c = cx_max == x_max;
            #[allow(clippy::float_cmp)]
            let tied_b_c = cy_min == y_min;
            #[allow(clippy::float_cmp)]
            let tied_t_c = cy_max == y_max;

            let n_l = tied_l.len() + tied_l_c as usize;
            let n_r = tied_r.len() + tied_r_c as usize;
            let n_b = tied_b.len() + tied_b_c as usize;
            let n_t = tied_t.len() + tied_t_c as usize;

            let w_l = 1.0 / n_l as f64;
            let w_r = 1.0 / n_r as f64;
            let w_b = 1.0 / n_b as f64;
            let w_t = 1.0 / n_t as f64;

            // Geometric gradient: per-shape blocks `[x, y, w, h]` then the
            // container block `[x_c, y_c, w_c, h_c]`. Chained into the
            // optimizer encoding below.
            let mut geom = vec![0.0_f64; n_params];

            for &i in &tied_l {
                geom[4 * i] -= dy * w_l;
                geom[4 * i + 2] += dy * 0.5 * w_l;
            }
            for &i in &tied_r {
                geom[4 * i] += dy * w_r;
                geom[4 * i + 2] += dy * 0.5 * w_r;
            }
            for &i in &tied_b {
                geom[4 * i + 1] -= dx * w_b;
                geom[4 * i + 3] += dx * 0.5 * w_b;
            }
            for &i in &tied_t {
                geom[4 * i + 1] += dx * w_t;
                geom[4 * i + 3] += dx * 0.5 * w_t;
            }
            if tied_l_c {
                geom[container_off] -= dy * w_l;
                geom[container_off + 2] += dy * 0.5 * w_l;
            }
            if tied_r_c {
                geom[container_off] += dy * w_r;
                geom[container_off + 2] += dy * 0.5 * w_r;
            }
            if tied_b_c {
                geom[container_off + 1] -= dx * w_b;
                geom[container_off + 3] += dx * 0.5 * w_b;
            }
            if tied_t_c {
                geom[container_off + 1] += dx * w_t;
                geom[container_off + 3] += dx * 0.5 * w_t;
            }

            for i in 0..n_sets {
                let w_i = shapes[i].width;
                let h_i = shapes[i].height;
                let da_dw = geom[4 * i + 2];
                let da_dh = geom[4 * i + 3];
                grad[4 * i] = geom[4 * i];
                grad[4 * i + 1] = geom[4 * i + 1];
                grad[4 * i + 2] = 0.5 * (da_dw * w_i + da_dh * h_i);
                grad[4 * i + 3] = 0.5 * (da_dw * w_i - da_dh * h_i);
            }
            let w_c = container.width;
            let h_c = container.height;
            let da_dw_c = geom[container_off + 2];
            let da_dh_c = geom[container_off + 3];
            grad[container_off] = geom[container_off];
            grad[container_off + 1] = geom[container_off + 1];
            grad[container_off + 2] = 0.5 * (da_dw_c * w_c + da_dh_c * h_c);
            grad[container_off + 3] = 0.5 * (da_dw_c * w_c - da_dh_c * h_c);
        }
        overlapping_grads.insert(mask, grad);
    }

    to_exclusive_areas_and_gradients(&overlapping_areas, &overlapping_grads, n_params)
}

impl Polygonize for Rectangle {
    /// Returns the four corners as a CCW polygon. `n_vertices` is ignored —
    /// a rectangle has exactly four vertices.
    fn polygonize(&self, _n_vertices: usize) -> Polygon {
        let Bounds {
            x_min,
            x_max,
            y_min,
            y_max,
        } = self.bounds();
        Polygon::new(vec![
            Point::new(x_min, y_min),
            Point::new(x_max, y_min),
            Point::new(x_max, y_max),
            Point::new(x_min, y_max),
        ])
    }
}

impl DiagramShape for Rectangle {
    fn compute_exclusive_regions(shapes: &[Self]) -> HashMap<RegionMask, f64> {
        let n_sets = shapes.len();
        let intersections = collect_intersections_rectangle(shapes, n_sets);
        let regions = discover_regions(shapes, &intersections, n_sets);

        let mut overlapping_areas: HashMap<RegionMask, f64> = HashMap::new();
        for &mask in &regions {
            let indices = mask_to_indices(mask, n_sets);
            // The n-way intersection of axis-aligned rectangles is one
            // axis-aligned rectangle (or empty).
            let mut x_min = f64::NEG_INFINITY;
            let mut x_max = f64::INFINITY;
            let mut y_min = f64::NEG_INFINITY;
            let mut y_max = f64::INFINITY;
            for &i in &indices {
                let Bounds {
                    x_min: a,
                    x_max: b,
                    y_min: c,
                    y_max: d,
                } = shapes[i].bounds();
                if a > x_min {
                    x_min = a;
                }
                if b < x_max {
                    x_max = b;
                }
                if c > y_min {
                    y_min = c;
                }
                if d < y_max {
                    y_max = d;
                }
            }
            let dx = (x_max - x_min).max(0.0);
            let dy = (y_max - y_min).max(0.0);
            overlapping_areas.insert(mask, dx * dy);
        }

        to_exclusive_areas(&overlapping_areas)
    }

    fn optimizer_params_from_circle(x: f64, y: f64, radius: f64) -> Vec<f64> {
        // Map the circle warm-start (area = π·r²) to a square of equal area
        // (`w = h = r·√π`). In optimizer encoding: u = ln(π·r²), v = 0.
        let u = PI.ln() + 2.0 * radius.ln();
        vec![x, y, u, 0.0]
    }

    fn mds_target_distance(
        area_i: f64,
        area_j: f64,
        target_overlap: f64,
    ) -> Result<f64, crate::error::DiagramError> {
        // The MDS phase only computes 2D positions; aspect ratio is
        // recovered by the final stage. Treat each rectangle as a square of
        // equal area and reuse Square's diagonal-direction inversion.
        let s_i = area_i.sqrt();
        let s_j = area_j.sqrt();
        let half_sum = 0.5 * (s_i + s_j);

        if target_overlap <= 0.0 {
            return Ok(half_sum * 2.0_f64.sqrt());
        }
        let root = target_overlap.sqrt();
        if root > half_sum {
            return Ok(0.0);
        }
        let d = 2.0_f64.sqrt() * (half_sum - root);
        Ok(d.max(0.0))
    }

    fn n_params() -> usize {
        4 // x, y, w, h (geometric); x, y, ln(area), ln(ratio) (optimizer)
    }

    fn from_params(params: &[f64]) -> Self {
        debug_assert_eq!(
            params.len(),
            4,
            "Rectangle requires 4 parameters: x, y, width, height"
        );
        Rectangle::new(
            Point::new(params[0], params[1]),
            params[2].max(f64::MIN_POSITIVE),
            params[3].max(f64::MIN_POSITIVE),
        )
    }

    fn to_params(&self) -> Vec<f64> {
        vec![self.center.x(), self.center.y(), self.width, self.height]
    }

    fn from_optimizer_params(params: &[f64]) -> Self {
        debug_assert_eq!(
            params.len(),
            4,
            "Rectangle optimizer params: x, y, ln(area), ln(ratio)"
        );
        let u = params[2];
        let v = params[3];
        let w = ((u + v) * 0.5).exp();
        let h = ((u - v) * 0.5).exp();
        // exp() is always > 0, but guard against NaN propagating through.
        let w = if w.is_finite() && w > 0.0 {
            w
        } else {
            f64::MIN_POSITIVE
        };
        let h = if h.is_finite() && h > 0.0 {
            h
        } else {
            f64::MIN_POSITIVE
        };
        Rectangle::new(Point::new(params[0], params[1]), w, h)
    }

    fn to_optimizer_params(&self) -> Vec<f64> {
        let u = (self.width * self.height).ln();
        let v = (self.width / self.height).ln();
        vec![self.center.x(), self.center.y(), u, v]
    }

    fn compute_exclusive_regions_with_gradient(
        shapes: &[Self],
    ) -> Option<ExclusiveRegionsAndGradient> {
        Some(compute_exclusive_regions_with_gradient_rectangles(shapes))
    }

    fn compute_exclusive_regions_clipped(
        shapes: &[Self],
        container: &Rectangle,
    ) -> Option<HashMap<RegionMask, f64>> {
        Some(compute_exclusive_regions_clipped_rectangles(
            shapes, container,
        ))
    }

    fn compute_exclusive_regions_clipped_with_gradient(
        shapes: &[Self],
        container: &Rectangle,
    ) -> Option<ExclusiveRegionsAndGradient> {
        Some(compute_exclusive_regions_clipped_with_gradient_rectangles(
            shapes, container,
        ))
    }

    /// Canonical axis-aligned Venn arrangements for `n ∈ {1, 2, 3}`.
    ///
    /// Reuses the [`Square`](crate::geometry::shapes::Square) footprints
    /// with `width = height = side`. `n ≥ 4` returns `None`: there is no
    /// axis-aligned Venn arrangement (square or rectangle) that opens all
    /// `2ⁿ − 1` regions for `n ≥ 4`.
    fn canonical_venn_layout(n: usize) -> Option<Vec<Self>> {
        let centers_and_side: &[((f64, f64), f64)] = match n {
            1 => &[((0.0, 0.0), 2.0)],
            2 => &[((-0.4, 0.0), 1.0), ((0.4, 0.0), 1.0)],
            3 => &[
                ((0.0, 0.36), 1.0),
                ((0.42, -0.36), 1.0),
                ((-0.42, -0.36), 1.0),
            ],
            _ => return None,
        };
        Some(
            centers_and_side
                .iter()
                .map(|&((x, y), s)| Rectangle::new(Point::new(x, y), s, s))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_rectangle_new() {
        let center = Point::new(1.0, 2.0);
        let rect = Rectangle::new(center, 4.0, 3.0);
        assert_eq!(rect.width(), 4.0);
        assert_eq!(rect.height(), 3.0);
        assert_eq!(rect.center().x(), 1.0);
        assert_eq!(rect.center().y(), 2.0);
    }

    #[test]
    fn test_rectangle_area() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        assert!(approx_eq(rect.area(), 12.0));

        let rect2 = Rectangle::new(Point::new(5.0, 5.0), 2.0, 5.0);
        assert!(approx_eq(rect2.area(), 10.0));
    }

    #[test]
    fn test_rectangle_perimeter() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        assert!(approx_eq(rect.perimeter(), 14.0));

        let rect2 = Rectangle::new(Point::new(1.0, 1.0), 2.0, 2.0);
        assert!(approx_eq(rect2.perimeter(), 8.0));
    }

    #[test]
    fn test_rectangle_bounds() {
        let rect = Rectangle::new(Point::new(2.0, 3.0), 4.0, 6.0);
        let Bounds {
            x_min,
            x_max,
            y_min,
            y_max,
        } = rect.bounds();
        assert!(approx_eq(x_min, 0.0));
        assert!(approx_eq(x_max, 4.0));
        assert!(approx_eq(y_min, 0.0));
        assert!(approx_eq(y_max, 6.0));
    }

    #[test]
    fn test_rectangle_corners() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let corners = rect.corners();
        assert_eq!(corners.len(), 4);

        let expected = [
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(-1.0, 1.0),
        ];

        for (corner, &expected_corner) in corners.iter().zip(expected.iter()) {
            assert!(approx_eq(corner.x(), expected_corner.x()));
            assert!(approx_eq(corner.y(), expected_corner.y()));
        }
    }

    #[test]
    fn test_rectangle_centroid() {
        let rect = Rectangle::new(Point::new(3.0, 4.0), 2.0, 2.0);
        let centroid = rect.centroid();
        assert!(approx_eq(centroid.x(), 3.0));
        assert!(approx_eq(centroid.y(), 4.0));
    }

    #[test]
    fn test_rectangle_contains_point() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0);

        assert!(rect.contains_point(&Point::new(0.0, 0.0)));
        assert!(rect.contains_point(&Point::new(1.0, 0.5)));
        assert!(rect.contains_point(&Point::new(-1.0, -0.5)));
        assert!(rect.contains_point(&Point::new(2.0, 1.0)));

        assert!(!rect.contains_point(&Point::new(3.0, 0.0)));
        assert!(!rect.contains_point(&Point::new(0.0, 2.0)));
    }

    #[test]
    fn test_rectangle_distance_no_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (4, 6, -1, 1)
        // Distance between edges: 4 - 1 = 3
        assert!(approx_eq(rect1.distance(&rect2), 3.0));
    }

    #[test]
    fn test_rectangle_distance_touching() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(3.0, 0.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (2, 4, -1, 1)
        // Distance between edges: 2 - 1 = 1
        assert!(approx_eq(rect1.distance(&rect2), 1.0));
    }

    #[test]
    fn test_rectangle_distance_overlapping() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        assert!(approx_eq(rect1.distance(&rect2), 0.0));
    }

    #[test]
    fn test_rectangle_distance_diagonal() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 5.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (4, 6, 4, 6)
        // dx = 4 - 1 = 3, dy = 4 - 1 = 3
        let expected = ((3.0_f64).powi(2) + (3.0_f64).powi(2)).sqrt();
        assert!(approx_eq(rect1.distance(&rect2), expected));
    }

    #[test]
    fn test_rectangle_contains_smaller() {
        let large = Rectangle::new(Point::new(0.0, 0.0), 10.0, 10.0);
        let small = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        assert!(large.contains(&small));
    }

    #[test]
    fn test_rectangle_contains_self() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 3.0, 3.0);
        assert!(rect.contains(&rect));
    }

    #[test]
    fn test_rectangle_not_contains() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);
        assert!(!rect1.contains(&rect2));
    }

    #[test]
    fn test_rectangle_not_contains_partial_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 3.0, 3.0);
        assert!(!rect1.contains(&rect2));
    }

    #[test]
    fn test_rectangle_intersects_separate() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);
        assert!(!rect1.intersects(&rect2));
    }

    #[test]
    fn test_rectangle_intersects_touching() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (1, 3, -1, 1)
        // They touch at x=1, so they intersect
        assert!(rect1.intersects(&rect2));
    }

    #[test]
    fn test_rectangle_intersects_overlapping() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        assert!(rect1.intersects(&rect2));
    }

    #[test]
    fn test_intersection_area_no_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(10.0, 0.0), 2.0, 2.0);
        assert!(approx_eq(rect1.intersection_area(&rect2), 0.0));
    }

    #[test]
    fn test_intersection_area_touching() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(3.0, 0.0), 2.0, 2.0);
        assert!(approx_eq(rect1.intersection_area(&rect2), 0.0));
    }

    #[test]
    fn test_intersection_area_complete_overlap_same_size() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        let rect2 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        assert!(approx_eq(rect1.intersection_area(&rect2), 12.0));
    }

    #[test]
    fn test_intersection_area_one_inside_other() {
        let large = Rectangle::new(Point::new(0.0, 0.0), 10.0, 10.0);
        let small = Rectangle::new(Point::new(1.0, 0.0), 4.0, 4.0);
        let expected = 16.0;
        assert!(approx_eq(large.intersection_area(&small), expected));
        assert!(approx_eq(small.intersection_area(&large), expected));
    }

    #[test]
    fn test_intersection_area_partial_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 4.0, 4.0);

        let x_overlap = 2.0;
        let y_overlap = 4.0;
        let expected = x_overlap * y_overlap;

        assert!(approx_eq(rect1.intersection_area(&rect2), expected));
    }

    #[test]
    fn test_intersection_area_symmetric() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        let rect2 = Rectangle::new(Point::new(1.5, 0.0), 3.0, 2.0);
        let area1 = rect1.intersection_area(&rect2);
        let area2 = rect2.intersection_area(&rect1);
        assert!(approx_eq(area1, area2));
    }

    #[test]
    fn test_intersection_area_different_sizes() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 6.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 2.0, 2.0);
        let area = rect1.intersection_area(&rect2);

        assert!(area > 0.0);
        assert!(area <= 4.0);
    }

    #[test]
    fn test_intersection_points_no_intersection() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);

        let points = rect1.intersection_points(&rect2);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_intersection_points_overlapping() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(1.0, 1.0), 2.0, 2.0);

        let points = rect1.intersection_points(&rect2);
        assert_eq!(points.len(), 4);

        for point in &points {
            assert!(rect1.contains_point(point));
            assert!(rect2.contains_point(point));
        }
    }

    #[test]
    fn test_intersection_points_partial_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 4.0, 4.0);

        let points = rect1.intersection_points(&rect2);
        assert_eq!(points.len(), 4);

        for point in &points {
            assert!(rect1.contains_point(point));
            assert!(rect2.contains_point(point));
        }
    }

    #[test]
    fn test_set_center() {
        let mut rect = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        rect.set_center(Point::new(5.0, 3.0));
        assert_eq!(rect.center().x(), 5.0);
        assert_eq!(rect.center().y(), 3.0);
    }

    // ----------------------------------------------------------------
    // DiagramShape / Polygonize tests
    // ----------------------------------------------------------------

    #[test]
    fn test_try_new_accepts_positive() {
        let r = Rectangle::try_new(Point::new(0.0, 0.0), 2.0, 1.0).unwrap();
        assert_eq!(r.width(), 2.0);
        assert_eq!(r.height(), 1.0);
    }

    #[test]
    fn test_try_new_rejects_zero_or_negative_width() {
        for bad in [0.0, -1.0] {
            let err = Rectangle::try_new(Point::new(0.0, 0.0), bad, 1.0).unwrap_err();
            assert!(matches!(
                err,
                crate::error::DiagramError::InvalidShapeParameter {
                    shape: "Rectangle",
                    param: "width",
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_try_new_rejects_zero_or_negative_height() {
        for bad in [0.0, -2.0] {
            let err = Rectangle::try_new(Point::new(0.0, 0.0), 1.0, bad).unwrap_err();
            assert!(matches!(
                err,
                crate::error::DiagramError::InvalidShapeParameter {
                    shape: "Rectangle",
                    param: "height",
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_to_params_round_trip() {
        let r = Rectangle::new(Point::new(1.5, -2.0), 3.0, 4.0);
        let p = r.to_params();
        assert_eq!(p, vec![1.5, -2.0, 3.0, 4.0]);
        let back = Rectangle::from_params(&p);
        assert_eq!(r, back);
    }

    #[test]
    fn test_optimizer_params_round_trip() {
        let r = Rectangle::new(Point::new(1.5, -2.0), 3.0, 4.0);
        let p = r.to_optimizer_params();
        let back = Rectangle::from_optimizer_params(&p);
        assert!((r.center().x() - back.center().x()).abs() < 1e-12);
        assert!((r.center().y() - back.center().y()).abs() < 1e-12);
        assert!((r.width() - back.width()).abs() < 1e-12);
        assert!((r.height() - back.height()).abs() < 1e-12);
    }

    #[test]
    fn test_optimizer_params_from_circle_equal_area() {
        // optimizer_params_from_circle should produce a square of the same
        // area as the seed circle (πr²).
        let r = 2.0;
        let p = Rectangle::optimizer_params_from_circle(0.0, 0.0, r);
        // u = ln(π·r²), v = 0
        assert!((p[2] - (PI * r * r).ln()).abs() < 1e-12);
        assert_eq!(p[3], 0.0);
        let rect = Rectangle::from_optimizer_params(&p);
        assert!(approx_eq(rect.area(), PI * r * r));
        assert!(approx_eq(rect.width(), rect.height()));
    }

    #[test]
    fn test_compute_exclusive_regions_two_disjoint() {
        let a = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let b = Rectangle::new(Point::new(10.0, 0.0), 2.0, 2.0);
        let regions = Rectangle::compute_exclusive_regions(&[a, b]);
        assert!(approx_eq(regions[&0b01], 4.0));
        assert!(approx_eq(regions[&0b10], 4.0));
        assert_eq!(regions.get(&0b11).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn test_compute_exclusive_regions_two_partial() {
        let a = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let b = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        // a∩b is a 1×2 rectangle of area 2; exclusive areas are each 4 − 2 = 2.
        let regions = Rectangle::compute_exclusive_regions(&[a, b]);
        assert!(approx_eq(regions[&0b01], 2.0));
        assert!(approx_eq(regions[&0b10], 2.0));
        assert!(approx_eq(regions[&0b11], 2.0));
    }

    #[test]
    fn test_compute_exclusive_regions_unequal_sides() {
        // Two 4×2 rectangles offset horizontally by 1: overlap is a
        // 3×2 rectangle of area 6. Exclusive areas: 4·2 − 6 = 2 each.
        let a = Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0);
        let b = Rectangle::new(Point::new(1.0, 0.0), 4.0, 2.0);
        let regions = Rectangle::compute_exclusive_regions(&[a, b]);
        assert!(approx_eq(regions[&0b01], 2.0));
        assert!(approx_eq(regions[&0b10], 2.0));
        assert!(approx_eq(regions[&0b11], 6.0));
    }

    #[test]
    fn test_compute_exclusive_regions_three_way_grid() {
        let a = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let b = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        let c = Rectangle::new(Point::new(0.5, 1.0), 2.0, 2.0);
        let regions = Rectangle::compute_exclusive_regions(&[a, b, c]);
        // Triple-intersection rectangle: x ∈ [0,1], y ∈ [0,1] → area 1.
        assert!(
            approx_eq(regions[&0b111], 1.0),
            "triple ∩ area = {}, expected 1.0",
            regions[&0b111]
        );
    }

    #[test]
    fn test_compute_exclusive_regions_nested() {
        let outer = Rectangle::new(Point::new(0.0, 0.0), 6.0, 4.0);
        let inner = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let regions = Rectangle::compute_exclusive_regions(&[outer, inner]);
        assert!(approx_eq(regions[&0b11], 4.0)); // inner fully inside outer
        assert!(approx_eq(regions[&0b01], 24.0 - 4.0));
        assert_eq!(regions.get(&0b10).copied().unwrap_or(0.0), 0.0);
    }

    /// Central-difference reference for `compute_exclusive_regions`. Returns
    /// a HashMap matching the analytical layout: per region, a length-`4·n_sets`
    /// gradient vector ordered `[x₀, y₀, u₀, v₀, x₁, …]` (optimizer encoding).
    fn fd_exclusive_region_gradients(
        shapes: &[Rectangle],
        h: f64,
    ) -> HashMap<RegionMask, Vec<f64>> {
        let n_sets = shapes.len();
        let n_params = n_sets * 4;
        let base = Rectangle::compute_exclusive_regions(shapes);

        let mut grads: HashMap<RegionMask, Vec<f64>> =
            base.keys().map(|&m| (m, vec![0.0; n_params])).collect();

        for i in 0..n_sets {
            for k in 0..4 {
                let perturb = |delta: f64| -> HashMap<RegionMask, f64> {
                    let mut copy: Vec<Rectangle> = shapes.to_vec();
                    let mut opt = copy[i].to_optimizer_params();
                    opt[k] += delta;
                    copy[i] = Rectangle::from_optimizer_params(&opt);
                    Rectangle::compute_exclusive_regions(&copy)
                };
                let plus = perturb(h);
                let minus = perturb(-h);
                for (&mask, g) in grads.iter_mut() {
                    let p = plus.get(&mask).copied().unwrap_or(0.0);
                    let m = minus.get(&mask).copied().unwrap_or(0.0);
                    g[4 * i + k] = (p - m) / (2.0 * h);
                }
            }
        }
        grads
    }

    fn assert_grad_matches_fd(
        analytical: &HashMap<RegionMask, Vec<f64>>,
        fd: &HashMap<RegionMask, Vec<f64>>,
        tol: f64,
    ) {
        for (&mask, ag) in analytical.iter() {
            let fg = fd
                .get(&mask)
                .expect("FD missing mask present in analytical");
            assert_eq!(ag.len(), fg.len(), "param count mismatch for mask {mask:b}");
            for (k, (&a, &f)) in ag.iter().zip(fg.iter()).enumerate() {
                assert!(
                    (a - f).abs() < tol,
                    "mask {mask:b} param {k}: analytical={a} fd={f} (tol={tol})"
                );
            }
        }
    }

    #[test]
    fn test_gradient_single_rectangle_matches_area_only_on_u() {
        // A = w·h = exp(u). So ∂A/∂u = A, ∂A/∂v = 0, ∂A/∂x = ∂A/∂y = 0.
        let r = Rectangle::new(Point::new(1.5, -2.0), 3.0, 4.0);
        let (areas, grads) = Rectangle::compute_exclusive_regions_with_gradient(&[r]).unwrap();
        let a = 3.0 * 4.0;
        assert!(approx_eq(areas[&0b1], a));
        let g = &grads[&0b1];
        assert_eq!(g.len(), 4);
        assert!(approx_eq(g[0], 0.0));
        assert!(approx_eq(g[1], 0.0));
        assert!(approx_eq(g[2], a)); // ∂A/∂u = A
        assert!(approx_eq(g[3], 0.0)); // ∂A/∂v = 0
    }

    #[test]
    fn test_gradient_two_rectangles_partial_overlap_matches_fd() {
        // Edge-coincident on y = ±1, exercising the tie-split path.
        let a = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let b = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        let (_, grads) = Rectangle::compute_exclusive_regions_with_gradient(&[a, b]).unwrap();
        let fd = fd_exclusive_region_gradients(&[a, b], 1e-6);
        assert_grad_matches_fd(&grads, &fd, 1e-5);
    }

    #[test]
    fn test_gradient_three_rectangles_overlap_matches_fd() {
        let a = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let b = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        let c = Rectangle::new(Point::new(0.5, 1.0), 2.0, 2.0);
        let (_, grads) = Rectangle::compute_exclusive_regions_with_gradient(&[a, b, c]).unwrap();
        let fd = fd_exclusive_region_gradients(&[a, b, c], 1e-6);
        assert_grad_matches_fd(&grads, &fd, 1e-5);
    }

    #[test]
    fn test_gradient_generic_no_ties_matches_fd_tightly() {
        // Generic configuration with all distinct edges so no tie-split
        // applies; central FD is O(h²) accurate and the gradient should
        // match to ~1e-7 with h=1e-5. Mix of widths and heights stresses
        // the chain-rule transformation.
        let a = Rectangle::new(Point::new(0.0, 0.0), 2.3, 1.7);
        let b = Rectangle::new(Point::new(1.1, 0.4), 1.7, 2.1);
        let c = Rectangle::new(Point::new(0.6, 1.2), 2.1, 1.5);
        let (_, grads) = Rectangle::compute_exclusive_regions_with_gradient(&[a, b, c]).unwrap();
        let fd = fd_exclusive_region_gradients(&[a, b, c], 1e-5);
        assert_grad_matches_fd(&grads, &fd, 1e-7);
    }

    #[test]
    fn test_gradient_disjoint_pair_is_zero_on_intersection() {
        let a = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let b = Rectangle::new(Point::new(10.0, 0.0), 2.0, 2.0);
        let (_, grads) = Rectangle::compute_exclusive_regions_with_gradient(&[a, b]).unwrap();
        if let Some(g) = grads.get(&0b11) {
            for &v in g {
                assert!(approx_eq(v, 0.0), "expected zero on disjoint pair, got {v}");
            }
        }
        let fd = fd_exclusive_region_gradients(&[a, b], 1e-5);
        for &mask in &[0b01_usize, 0b10_usize] {
            let ag = grads.get(&mask).expect("singleton missing");
            let fg = fd.get(&mask).expect("FD singleton missing");
            for (k, (&a, &f)) in ag.iter().zip(fg.iter()).enumerate() {
                assert!(
                    (a - f).abs() < 1e-6,
                    "mask {mask:b} param {k}: analytical={a} fd={f}"
                );
            }
        }
    }

    #[test]
    fn test_gradient_nested_matches_fd() {
        let outer = Rectangle::new(Point::new(0.0, 0.0), 6.0, 4.0);
        let inner = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let (areas, grads) =
            Rectangle::compute_exclusive_regions_with_gradient(&[outer, inner]).unwrap();
        let fd = fd_exclusive_region_gradients(&[outer, inner], 1e-5);
        assert!(approx_eq(areas[&0b11], 4.0));
        assert!(approx_eq(areas[&0b01], 20.0));
        assert_grad_matches_fd(&grads, &fd, 1e-7);
    }

    #[test]
    fn test_mds_target_distance_zero_overlap_is_diagonal_tangency() {
        let area = 1.0;
        let d = Rectangle::mds_target_distance(area, area, 0.0).unwrap();
        let s = area.sqrt();
        let expected = 2.0_f64.sqrt() * s;
        assert!(approx_eq(d, expected), "d = {d}, expected {expected}");
    }

    #[test]
    fn test_mds_target_distance_full_overlap_is_zero() {
        let area = 4.0;
        let d = Rectangle::mds_target_distance(area, area, area).unwrap();
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn test_polygonize_returns_4_ccw_vertices_with_correct_area() {
        let r = Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0);
        let p = r.polygonize(0);
        assert_eq!(p.vertices().len(), 4);
        // CCW shoelace gives positive area = 8.
        let v = p.vertices();
        let mut shoelace = 0.0;
        for i in 0..4 {
            let j = (i + 1) % 4;
            shoelace += v[i].x() * v[j].y() - v[j].x() * v[i].y();
        }
        assert!(approx_eq(0.5 * shoelace, 8.0));
    }

    #[test]
    fn test_fitter_end_to_end_two_partial_overlap() {
        use crate::{DiagramSpecBuilder, Fitter, InputType};

        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .intersection(&["A", "B"], 2.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Rectangle>::new(&spec).seed(42).fit().unwrap();
        let fitted = layout.fitted();
        assert!(
            fitted.values().all(|&v| v.is_finite()),
            "non-finite fitted areas in {fitted:?}"
        );
        assert!(
            layout.loss().is_finite(),
            "non-finite loss {}",
            layout.loss()
        );
        assert_eq!(layout.shapes().len(), 2);
    }

    fn assert_rect(actual: &Rectangle, x: f64, y: f64, w: f64, h: f64) {
        assert!(approx_eq(actual.center().x(), x));
        assert!(approx_eq(actual.center().y(), y));
        assert!(approx_eq(actual.width(), w));
        assert!(approx_eq(actual.height(), h));
    }

    #[test]
    fn test_canonical_venn_layout_n1() {
        let shapes = Rectangle::canonical_venn_layout(1).unwrap();
        assert_eq!(shapes.len(), 1);
        assert_rect(&shapes[0], 0.0, 0.0, 2.0, 2.0);
    }

    #[test]
    fn test_canonical_venn_layout_n2() {
        let shapes = Rectangle::canonical_venn_layout(2).unwrap();
        assert_eq!(shapes.len(), 2);
        assert_rect(&shapes[0], -0.4, 0.0, 1.0, 1.0);
        assert_rect(&shapes[1], 0.4, 0.0, 1.0, 1.0);
    }

    #[test]
    fn test_canonical_venn_layout_n3() {
        let shapes = Rectangle::canonical_venn_layout(3).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_rect(&shapes[0], 0.0, 0.36, 1.0, 1.0);
        assert_rect(&shapes[1], 0.42, -0.36, 1.0, 1.0);
        assert_rect(&shapes[2], -0.42, -0.36, 1.0, 1.0);
    }

    #[test]
    fn test_canonical_venn_layout_unsupported() {
        assert!(Rectangle::canonical_venn_layout(0).is_none());
        assert!(Rectangle::canonical_venn_layout(4).is_none());
        assert!(Rectangle::canonical_venn_layout(5).is_none());
    }

    // ----------------------------------------------------------------
    // Container clipping (S5)
    // ----------------------------------------------------------------

    #[test]
    fn test_clipped_no_rects_complement_equals_container_area() {
        let container = Rectangle::new(Point::new(0.0, 0.0), 5.0, 4.0);
        let areas = compute_exclusive_regions_clipped_rectangles(&[], &container);
        assert!(approx_eq(areas[&0], 20.0));
    }

    #[test]
    fn test_clipped_single_rect_inside_container() {
        let r = Rectangle::new(Point::new(0.0, 0.0), 2.0, 1.0);
        let container = Rectangle::new(Point::new(0.0, 0.0), 5.0, 4.0);
        let areas = compute_exclusive_regions_clipped_rectangles(&[r], &container);
        assert!(approx_eq(areas[&0b1], 2.0));
        assert!(approx_eq(areas[&0], 20.0 - 2.0));
    }

    #[test]
    fn test_clipped_single_rect_outside_container() {
        let r = Rectangle::new(Point::new(100.0, 0.0), 2.0, 1.0);
        let container = Rectangle::new(Point::new(0.0, 0.0), 5.0, 4.0);
        let areas = compute_exclusive_regions_clipped_rectangles(&[r], &container);
        assert!(approx_eq(areas.get(&0b1).copied().unwrap_or(0.0), 0.0));
        assert!(approx_eq(areas[&0], 20.0));
    }

    #[test]
    fn test_clipped_single_rect_partial_clip() {
        // Rectangle of size 4x2 centered at (1, 0): bounds x ∈ [-1, 3],
        // y ∈ [-1, 1]. Container of size 4x4 at origin: bounds x ∈ [-2, 2],
        // y ∈ [-2, 2]. Overlap x ∈ [-1, 2] (= 3) × y ∈ [-1, 1] (= 2) → 6.
        let r = Rectangle::new(Point::new(1.0, 0.0), 4.0, 2.0);
        let container = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let areas = compute_exclusive_regions_clipped_rectangles(&[r], &container);
        assert!(approx_eq(areas[&0b1], 6.0));
        assert!(approx_eq(areas[&0], 16.0 - 6.0));
    }

    #[test]
    fn test_clipped_container_fully_inside_rect() {
        // Container is fully inside the (huge) rectangle; the rectangle's
        // clipped exclusive area equals the container's area, complement is 0.
        let r = Rectangle::new(Point::new(0.0, 0.0), 100.0, 100.0);
        let container = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        let areas = compute_exclusive_regions_clipped_rectangles(&[r], &container);
        assert!(approx_eq(areas[&0b1], 12.0));
        assert!(approx_eq(areas[&0], 0.0));
    }

    #[test]
    fn test_clipped_two_rects_inside_sum_to_union() {
        let a = Rectangle::new(Point::new(-0.6, 0.0), 1.6, 1.6);
        let b = Rectangle::new(Point::new(0.6, 0.0), 1.6, 1.6);
        let container = Rectangle::new(Point::new(0.0, 0.0), 5.0, 4.0);

        let unclipped = Rectangle::compute_exclusive_regions(&[a, b]);
        let clipped = compute_exclusive_regions_clipped_rectangles(&[a, b], &container);

        // Both rectangles fully inside container, so per-shape exclusive
        // areas match the unclipped path.
        for &mask in &[0b01_usize, 0b10_usize, 0b11_usize] {
            let lhs = clipped.get(&mask).copied().unwrap_or(0.0);
            let rhs = unclipped.get(&mask).copied().unwrap_or(0.0);
            assert!(
                approx_eq(lhs, rhs),
                "mask {mask:b}: clipped={lhs} unclipped={rhs}"
            );
        }
        // Complement equals container - union.
        let union: f64 = clipped[&0b01] + clipped[&0b10] + clipped[&0b11];
        assert!(approx_eq(clipped[&0], 20.0 - union));
    }

    /// Pack `(rectangles, container)` into the same flat parameter layout the
    /// optimiser uses: per-rect `[x, y, u, v]` blocks (optimizer encoding),
    /// then container `[x_c, y_c, u_c, v_c]`.
    fn pack_clipped_rect_params(rects: &[Rectangle], container: &Rectangle) -> Vec<f64> {
        let mut p = Vec::with_capacity(rects.len() * 4 + 4);
        for r in rects {
            p.extend(r.to_optimizer_params());
        }
        p.extend(container.to_optimizer_params());
        p
    }

    /// Inverse of `pack_clipped_rect_params`: decode a flat parameter slice
    /// back into rectangles plus container.
    fn unpack_clipped_rect_params(p: &[f64], n_sets: usize) -> (Vec<Rectangle>, Rectangle) {
        let rects: Vec<Rectangle> = (0..n_sets)
            .map(|i| Rectangle::from_optimizer_params(&p[4 * i..4 * i + 4]))
            .collect();
        let container = Rectangle::from_optimizer_params(&p[4 * n_sets..4 * n_sets + 4]);
        (rects, container)
    }

    /// FD vs analytical gradient comparison for the clipped per-mask area
    /// helper. `params` is laid out as in `pack_clipped_rect_params`.
    fn assert_clipped_rect_gradient_matches_fd(
        rects: &[Rectangle],
        container: &Rectangle,
        h: f64,
        tol: f64,
        label: &str,
    ) {
        let (areas, grads) =
            compute_exclusive_regions_clipped_with_gradient_rectangles(rects, container);
        let n_sets = rects.len();
        let n_params = n_sets * 4 + 4;
        let p0 = pack_clipped_rect_params(rects, container);

        for (mask, analytic) in &grads {
            assert_eq!(
                analytic.len(),
                n_params,
                "{}: gradient length {} ≠ expected {}",
                label,
                analytic.len(),
                n_params
            );
            let mut fd = vec![0.0; n_params];
            for i in 0..n_params {
                let mut plus = p0.clone();
                let mut minus = p0.clone();
                plus[i] += h;
                minus[i] -= h;
                let (rp, kp) = unpack_clipped_rect_params(&plus, n_sets);
                let (rm, km) = unpack_clipped_rect_params(&minus, n_sets);
                let ap = compute_exclusive_regions_clipped_rectangles(&rp, &kp)
                    .get(mask)
                    .copied()
                    .unwrap_or(0.0);
                let am = compute_exclusive_regions_clipped_rectangles(&rm, &km)
                    .get(mask)
                    .copied()
                    .unwrap_or(0.0);
                fd[i] = (ap - am) / (2.0 * h);
            }
            let direct = compute_exclusive_regions_clipped_rectangles(rects, container)
                .get(mask)
                .copied()
                .unwrap_or(0.0);
            let analytic_area = areas.get(mask).copied().unwrap_or(0.0);
            assert!(
                (analytic_area - direct).abs() < 1e-12,
                "{label}: mask {mask:b} area {analytic_area} vs direct {direct} mismatch"
            );
            let diff_norm: f64 = analytic
                .iter()
                .zip(fd.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
            let rel = if fd_norm > 1e-9 {
                diff_norm / fd_norm
            } else {
                diff_norm
            };
            assert!(
                rel < tol,
                "{label}: mask {mask:b} analytic vs FD mismatch (rel={rel:.3e}, |fd|={fd_norm:.3e})\n  analytic={analytic:?}\n  fd      ={fd:?}"
            );
        }
    }

    #[test]
    fn test_clipped_grad_two_rects_inside_box_matches_fd() {
        let rects = vec![
            Rectangle::new(Point::new(-0.6, 0.05), 1.7, 1.3),
            Rectangle::new(Point::new(0.62, -0.04), 1.4, 1.5),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.0), 6.0, 5.0);
        assert_clipped_rect_gradient_matches_fd(&rects, &container, 1e-5, 1e-7, "two_rects_inside");
    }

    #[test]
    fn test_clipped_grad_two_rects_one_clipped_by_edge_matches_fd() {
        // Right rectangle extends past the container's right edge, exercising
        // the container-x_max binding path. y-extents are deliberately
        // distinct in f64 so no shape-vs-shape ties slip in via floating-point
        // coincidence.
        let rects = vec![
            Rectangle::new(Point::new(0.0, 0.07), 1.7, 1.3),
            Rectangle::new(Point::new(1.4, -0.03), 1.5, 1.5),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.0), 3.6, 3.0);
        assert_clipped_rect_gradient_matches_fd(
            &rects,
            &container,
            1e-5,
            1e-6,
            "two_rects_one_clipped",
        );
    }

    #[test]
    fn test_clipped_grad_three_rects_inside_box_matches_fd() {
        let rects = vec![
            Rectangle::new(Point::new(-0.5, -0.3), 1.8, 1.4),
            Rectangle::new(Point::new(0.5, -0.3), 1.7, 1.5),
            Rectangle::new(Point::new(0.0, 0.55), 1.6, 1.3),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.05), 3.5, 3.5);
        assert_clipped_rect_gradient_matches_fd(
            &rects,
            &container,
            1e-5,
            1e-7,
            "three_rects_inside",
        );
    }

    #[test]
    fn test_clipped_grad_three_rects_one_clipped_matches_fd() {
        let rects = vec![
            Rectangle::new(Point::new(-0.6, -0.2), 1.5, 1.3),
            Rectangle::new(Point::new(0.7, -0.2), 1.5, 1.4),
            Rectangle::new(Point::new(0.05, 0.6), 1.4, 1.6),
        ];
        let container = Rectangle::new(Point::new(0.0, 0.0), 3.5, 2.0);
        assert_clipped_rect_gradient_matches_fd(
            &rects,
            &container,
            1e-5,
            1e-5,
            "three_rects_one_clipped",
        );
    }

    /// End-to-end fitter test with complement: two rectangles that should fit
    /// comfortably inside a jointly-optimised container.
    #[test]
    fn fit_two_rectangles_with_complement_runs_to_finite_loss() {
        use crate::{DiagramSpecBuilder, Fitter, InputType};

        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .intersection(&["A", "B"], 1.0)
            .complement(20.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Rectangle>::new(&spec).seed(42).fit().unwrap();
        let fitted = layout.fitted();
        assert!(
            fitted.values().all(|&v| v.is_finite()),
            "non-finite fitted areas in {fitted:?}"
        );
        assert!(
            layout.loss().is_finite(),
            "non-finite loss {}",
            layout.loss()
        );
        let container = layout
            .container()
            .expect("complement spec carries container");
        // Universe = 4 + 4 + 1 + 20 = 29.
        assert!(
            (container.area() - 29.0).abs() / 29.0 < 0.1,
            "container area {} should be near universe 29",
            container.area()
        );
    }
}
