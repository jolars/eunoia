//! Refine straight leaders without changing their edge type or attachment rule.

use std::f64::consts::TAU;

use super::{
    ExteriorEntry, ExteriorPolicy, InteriorAabb, LeaderStrategy, PlacementStrategy,
    principal_axis_direction, raycast_anchor, raycast_anchor_union, segment_vs_aabb,
    segments_properly_cross, straight_leader,
};
use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::plotting::regions::{RegionPiece, RegionPolygons, signed_clearance};
use crate::spec::Combination;

struct Obstacle<'a> {
    combo: &'a Combination,
    piece: &'a RegionPiece,
    bounds: InteriorAabb,
}

pub(super) fn refine_straight_leaders(
    entries: &mut [ExteriorEntry],
    regions: &RegionPolygons,
    union: &[RegionPiece],
    bbox: &Rectangle,
    interior_boxes: &[InteriorAabb],
    strategy: &PlacementStrategy,
) {
    if entries.is_empty() {
        return;
    }
    let epsilon = 1e-8 * bbox.width().hypot(bbox.height());
    let obstacles: Vec<_> = regions
        .iter()
        .flat_map(|(combo, pieces)| {
            pieces.iter().filter_map(move |piece| {
                let mut bounds = InteriorAabb {
                    xmin: f64::INFINITY,
                    ymin: f64::INFINITY,
                    xmax: f64::NEG_INFINITY,
                    ymax: f64::NEG_INFINITY,
                };
                for p in piece.outer.vertices() {
                    bounds.xmin = bounds.xmin.min(p.x());
                    bounds.ymin = bounds.ymin.min(p.y());
                    bounds.xmax = bounds.xmax.max(p.x());
                    bounds.ymax = bounds.ymax.max(p.y());
                }
                (piece.outer.vertices().len() >= 3).then_some(Obstacle {
                    combo,
                    piece,
                    bounds,
                })
            })
        })
        .collect();
    let matched_margin = match strategy.leader {
        LeaderStrategy::Straight(ExteriorPolicy::Matched { margin }) => {
            Some(margin.unwrap_or_else(|| {
                0.5 * entries.iter().map(|e| e.w.max(e.h)).fold(0.0_f64, f64::max)
            }))
        }
        _ => None,
    };
    let angle = |p: &Point| (p.y() - bbox.center().y()).atan2(p.x() - bbox.center().x());
    let mut order: Vec<_> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| angle(&entries[a].anchor).total_cmp(&angle(&entries[b].anchor)));
    let mut boxes: Vec<_> = entries
        .iter()
        .map(|e| label_box(&e.anchor, e.w, e.h))
        .collect();
    let mut leaders: Vec<_> = entries
        .iter()
        .map(|e| straight_leader(e, &e.anchor, &regions.get(&e.combo).unwrap()[..], strategy))
        .collect();

    // Each accepted move decreases this label's foreign traversal. Later labels
    // see the committed geometry, so no subsequent collision sweep can undo it.
    for i in 0..entries.len() {
        let entry = &entries[i];
        let pieces = &regions.get(&entry.combo).unwrap()[..];
        let original = leaders[i];
        let original_cost = foreign_length(original, &entry.combo, &obstacles, epsilon);
        if original_cost <= epsilon {
            continue;
        }
        let old_angle = (entry.anchor.y() - entry.poi.y()).atan2(entry.anchor.x() - entry.poi.x());
        let mut directions = vec![
            (
                entry.anchor.x() - entry.poi.x(),
                entry.anchor.y() - entry.poi.y(),
            ),
            principal_axis_direction(pieces),
        ];
        let axis = directions[1];
        directions.push((-axis.0, -axis.1));
        // Vertex bearings include narrow exits where several region boundaries
        // meet; a uniform angular grid alone can miss these entirely.
        for piece in pieces {
            directions.extend(
                piece
                    .outer
                    .vertices()
                    .iter()
                    .map(|p| (p.x() - entry.poi.x(), p.y() - entry.poi.y())),
            );
        }
        directions.extend((0..64).map(|k| {
            let a = TAU * k as f64 / 64.0;
            (a.cos(), a.sin())
        }));
        let mut directions: Vec<_> = directions
            .into_iter()
            .filter_map(|(x, y)| {
                let length = x.hypot(y);
                (length > 0.0 && length.is_finite()).then(|| (x / length, y / length))
            })
            .collect();
        directions.sort_by(|a, b| a.1.atan2(a.0).total_cmp(&b.1.atan2(b.0)));
        directions.dedup_by(|a, b| (a.0 - b.0).hypot(a.1 - b.1) < 1e-12);

        let margin = matched_margin.unwrap_or(entry.margin).max(0.0);
        let mut best_anchor = entry.anchor;
        let mut best_leader = original;
        let mut best_score = (original_cost, distance(original.0, original.1), 0.0);
        for dir in directions {
            // The silhouette is usually tighter. Keeping the AABB candidate
            // also permits a safe fallback when polygon discretization leaves
            // its expanded label box too close to an edge.
            let anchors = [
                raycast_anchor_union(&entry.poi, entry.w, entry.h, union, margin, dir),
                Some(raycast_anchor(
                    &entry.poi, entry.w, entry.h, bbox, margin, dir,
                )),
            ];
            for anchor in anchors.into_iter().flatten() {
                if !anchor.x().is_finite() || !anchor.y().is_finite() {
                    continue;
                }
                let leader = straight_leader(entry, &anchor, pieces, strategy);
                let cost = foreign_length(leader, &entry.combo, &obstacles, epsilon);
                let deviation = ((dir.1.atan2(dir.0) - old_angle + TAU / 2.0).rem_euclid(TAU)
                    - TAU / 2.0)
                    .abs();
                let score = (cost, distance(leader.0, leader.1), deviation);
                if cost >= original_cost - epsilon || !better(score, best_score, epsilon) {
                    continue;
                }
                if matched_margin.is_some() && order.len() > 2 {
                    let pos = order.iter().position(|&idx| idx == i).unwrap();
                    let prev = angle(&entries[order[(pos + order.len() - 1) % order.len()]].anchor);
                    let next = angle(&entries[order[(pos + 1) % order.len()]].anchor);
                    if (angle(&anchor) - prev).rem_euclid(TAU) > (next - prev).rem_euclid(TAU) {
                        continue;
                    }
                }
                let label = label_box(&anchor, entry.w, entry.h);
                let padded = label_box(&anchor, entry.w + 2.0 * margin, entry.h + 2.0 * margin);
                if obstacles
                    .iter()
                    .any(|o| box_hits_piece(&padded, o, epsilon))
                    || interior_boxes
                        .iter()
                        .any(|b| boxes_overlap(&label, b, epsilon) || leader_hits_box(leader, b))
                {
                    continue;
                }
                if (0..entries.len()).filter(|&j| j != i).any(|j| {
                    boxes_overlap(&label, &boxes[j], epsilon)
                        || leader_hits_box(leader, &boxes[j])
                        || leader_hits_box(leaders[j], &label)
                        || (segments_properly_cross(
                            &leader.0,
                            &leader.1,
                            &leaders[j].0,
                            &leaders[j].1,
                        ) && !segments_properly_cross(
                            &original.0,
                            &original.1,
                            &leaders[j].0,
                            &leaders[j].1,
                        ))
                }) {
                    continue;
                }
                best_anchor = anchor;
                best_leader = leader;
                best_score = score;
            }
        }
        boxes[i] = label_box(&best_anchor, entry.w, entry.h);
        leaders[i] = best_leader;
        entries[i].anchor = best_anchor;
    }
}

fn distance(a: Point, b: Point) -> f64 {
    (b.x() - a.x()).hypot(b.y() - a.y())
}

fn better(a: (f64, f64, f64), b: (f64, f64, f64), epsilon: f64) -> bool {
    a.0 < b.0 - epsilon
        || ((a.0 - b.0).abs() <= epsilon
            && (a.1 < b.1 - epsilon || ((a.1 - b.1).abs() <= epsilon && a.2 < b.2)))
}

fn label_box(anchor: &Point, w: f64, h: f64) -> InteriorAabb {
    InteriorAabb {
        xmin: anchor.x() - 0.5 * w,
        ymin: anchor.y() - 0.5 * h,
        xmax: anchor.x() + 0.5 * w,
        ymax: anchor.y() + 0.5 * h,
    }
}

fn boxes_overlap(a: &InteriorAabb, b: &InteriorAabb, epsilon: f64) -> bool {
    a.xmax.min(b.xmax) - a.xmin.max(b.xmin) > epsilon
        && a.ymax.min(b.ymax) - a.ymin.max(b.ymin) > epsilon
}

fn leader_hits_box((start, end): (Point, Point), b: &InteriorAabb) -> bool {
    segment_vs_aabb(&start, &end, b.xmin, b.ymin, b.xmax, b.ymax).is_some()
}

fn box_hits_piece(b: &InteriorAabb, obstacle: &Obstacle<'_>, epsilon: f64) -> bool {
    if !boxes_overlap(b, &obstacle.bounds, epsilon) {
        return false;
    }
    let p = obstacle.piece;
    for x in [b.xmin, b.xmax] {
        for y in [b.ymin, b.ymax] {
            if signed_clearance(x, y, p) > epsilon {
                return true;
            }
        }
    }
    for ring in std::iter::once(&p.outer).chain(&p.holes) {
        let vertices = ring.vertices();
        for (a, c) in vertices
            .iter()
            .zip(vertices.iter().cycle().skip(1))
            .take(vertices.len())
        {
            if segment_vs_aabb(
                a,
                c,
                b.xmin + epsilon,
                b.ymin + epsilon,
                b.xmax - epsilon,
                b.ymax - epsilon,
            )
            .is_some()
            {
                return true;
            }
        }
    }
    false
}

fn foreign_length(
    (start, end): (Point, Point),
    own: &Combination,
    obstacles: &[Obstacle<'_>],
    epsilon: f64,
) -> f64 {
    obstacles
        .iter()
        .filter(|o| o.combo != own)
        .map(|o| segment_length_in_piece(start, end, o, epsilon))
        .sum()
}

fn segment_length_in_piece(start: Point, end: Point, obstacle: &Obstacle<'_>, epsilon: f64) -> f64 {
    let length = distance(start, end);
    if length <= epsilon || !leader_hits_box((start, end), &obstacle.bounds) {
        return 0.0;
    }
    let d = (end.x() - start.x(), end.y() - start.y());
    let cross = |a: (f64, f64), b: (f64, f64)| a.0 * b.1 - a.1 * b.0;
    let mut cuts = vec![0.0, 1.0];
    for ring in std::iter::once(&obstacle.piece.outer).chain(&obstacle.piece.holes) {
        let vertices = ring.vertices();
        for (a, b) in vertices
            .iter()
            .zip(vertices.iter().cycle().skip(1))
            .take(vertices.len())
        {
            let edge = (b.x() - a.x(), b.y() - a.y());
            let delta = (a.x() - start.x(), a.y() - start.y());
            let denom = cross(d, edge);
            let edge_length = edge.0.hypot(edge.1);
            if denom.abs() <= 8.0 * f64::EPSILON * length * edge_length {
                // Collinear overlaps need endpoints too, so a run along a
                // boundary cannot be mistaken for a run through its interior.
                if cross(delta, d).abs() <= epsilon * length {
                    for p in [a, b] {
                        let t = ((p.x() - start.x()) * d.0 + (p.y() - start.y()) * d.1)
                            / length.powi(2);
                        cuts.push(t.clamp(0.0, 1.0));
                    }
                }
                continue;
            }
            let t = cross(delta, edge) / denom;
            let u = cross(delta, d) / denom;
            let edge_epsilon = epsilon / edge_length;
            if (0.0..=1.0).contains(&t) && (-edge_epsilon..=1.0 + edge_epsilon).contains(&u) {
                cuts.push(t);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() <= epsilon / length);
    cuts.windows(2)
        .filter(|ts| {
            let t = 0.5 * (ts[0] + ts[1]);
            signed_clearance(start.x() + t * d.0, start.y() + t * d.1, obstacle.piece) > epsilon
        })
        .map(|ts| (ts[1] - ts[0]) * length)
        .sum()
}

#[cfg(test)]
mod tests;
