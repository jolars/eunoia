//! Trajectory recording for the fitter pipeline.
//!
//! This is a diagnostics/visualization path that re-runs a single deterministic
//! fit attempt with `basin` observers attached, capturing the optimizer's
//! parameter vector at every iteration so the journey from random start →
//! MDS init → final optimization can be replayed as an animation (it backs the
//! `fitter-pipeline` docs page via the wasm `record_*_trajectory` entry points).
//!
//! It deliberately stays off the production hot path: [`Fitter::fit_recording`]
//! forces `n_restarts = 1`, pins Levenberg-Marquardt for both phases (so the
//! observer always sees a single-iterate `NllsState<DVector<f64>>`), and uses
//! the `SumSquared` least-squares loss. The recorders live only here and on the
//! dedicated `run_*_recorded` entry points; `fit()` / `fit_initial_only()` never
//! allocate one and pay no overhead.
//!
//! [`Fitter::fit_recording`]: crate::Fitter::fit_recording

use std::cell::RefCell;
use std::rc::Rc;

use nalgebra::DVector;

/// Which stage of the pipeline a recorded frame belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The random initial layout, before any MDS iteration. This is the
    /// `observe_init` frame of the MDS solver — the raw sampled positions.
    Init,
    /// Multidimensional-scaling initial-layout refinement (centers only,
    /// shape sizes held fixed at their circle-equivalent radii).
    Mds,
    /// Final loss-minimizing optimization over the full shape parameters.
    Final,
}

impl Stage {
    /// Lowercase tag used when serializing frames to JS (`"init"` / `"mds"` /
    /// `"final"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Init => "init",
            Stage::Mds => "mds",
            Stage::Final => "final",
        }
    }
}

/// One recorded optimizer iterate, with the shapes already decoded into `S`.
#[derive(Debug, Clone)]
pub struct RecordedFrame<S> {
    /// Which pipeline stage produced this frame.
    pub stage: Stage,
    /// The solver's iteration counter at this frame (resets per stage; the
    /// `Init` frame and the first `Mds` frame are both iteration 0).
    pub iteration: u64,
    /// The final region-area loss (`LossType::SumSquared`) at this frame —
    /// computed from the frame's shapes for *every* stage, so the same
    /// criterion falls consistently from the random start through MDS and the
    /// final optimization rather than switching objectives mid-pipeline.
    pub cost: f64,
    /// The shapes at this frame, in preprocessed-set order.
    pub shapes: Vec<S>,
}

/// The full recorded trajectory of a single deterministic fit attempt.
#[derive(Debug, Clone)]
pub struct FitRecording<S> {
    /// Frames in playback order: one `Init`, then the `Mds` iterates, then the
    /// `Final` iterates.
    pub frames: Vec<RecordedFrame<S>>,
    /// Set names in the order shapes appear within each frame (i.e. the
    /// preprocessed-spec order, with empty sets dropped). `set_names[i]` labels
    /// `frame.shapes[i]`.
    pub set_names: Vec<String>,
}

/// A raw (undecoded) recorded iterate: the optimizer parameter vector plus its
/// iteration index. Decoding into shapes (and scoring the region-area loss) is
/// stage-specific and happens in [`Fitter::fit_recording`]; the solver's own
/// cost is not retained since the displayed loss is recomputed uniformly.
///
/// [`Fitter::fit_recording`]: crate::Fitter::fit_recording
pub(crate) struct RawFrame {
    pub iteration: u64,
    pub params: Vec<f64>,
}

/// basin observer that appends each iterate's parameter vector to a shared sink.
///
/// Capture is via `Rc<RefCell<…>>` (single-threaded — the recording path never
/// touches rayon), so the caller keeps a clone of the `Rc` and drains the
/// frames after the solver returns. A `max_frames` cap bounds the payload on
/// pathologically long runs; the curated docs example stays well under it.
pub(crate) struct FrameRecorder {
    sink: Rc<RefCell<Vec<RawFrame>>>,
    max_frames: usize,
}

impl FrameRecorder {
    /// Create a recorder writing into `sink`, dropping frames past `max_frames`.
    pub(crate) fn new(sink: Rc<RefCell<Vec<RawFrame>>>, max_frames: usize) -> Self {
        Self { sink, max_frames }
    }

    fn push<S>(&self, state: &S)
    where
        S: basin::State<Param = DVector<f64>, Float = f64>,
    {
        let mut sink = self.sink.borrow_mut();
        if sink.len() >= self.max_frames {
            return;
        }
        sink.push(RawFrame {
            iteration: state.iter(),
            params: state.param().as_slice().to_vec(),
        });
    }
}

impl<S> basin::Observe<S> for FrameRecorder
where
    S: basin::State<Param = DVector<f64>, Float = f64>,
{
    fn observe_init(&mut self, state: &S) {
        self.push(state);
    }

    fn observe_iter(&mut self, state: &S) {
        self.push(state);
    }
}

#[cfg(test)]
mod tests {
    use super::Stage;
    use crate::DiagramSpecBuilder;
    use crate::Fitter;
    use crate::geometry::shapes::{Circle, Ellipse};

    fn three_set_spec() -> crate::spec::DiagramSpec {
        DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 9.0)
            .set("C", 8.0)
            .intersection(&["A", "B"], 3.0)
            .intersection(&["A", "C"], 2.5)
            .intersection(&["B", "C"], 2.0)
            .intersection(&["A", "B", "C"], 1.0)
            .build()
            .unwrap()
    }

    #[test]
    fn recording_walks_init_mds_final_in_order() {
        let spec = three_set_spec();
        let rec = Fitter::<Circle>::new(&spec)
            .seed(42)
            .fit_recording()
            .unwrap();

        assert!(!rec.frames.is_empty(), "expected recorded frames");
        // Exactly one Init frame, and it comes first.
        assert_eq!(rec.frames[0].stage, Stage::Init);
        assert_eq!(
            rec.frames.iter().filter(|f| f.stage == Stage::Init).count(),
            1
        );
        // All three stages present.
        assert!(rec.frames.iter().any(|f| f.stage == Stage::Mds));
        assert!(rec.frames.iter().any(|f| f.stage == Stage::Final));

        // Stage order is non-decreasing as Init(0) → Mds(1) → Final(2): once we
        // leave a stage we never return to it.
        let rank = |s: Stage| match s {
            Stage::Init => 0,
            Stage::Mds => 1,
            Stage::Final => 2,
        };
        let mut prev = 0;
        for f in &rec.frames {
            let r = rank(f.stage);
            assert!(r >= prev, "stages out of order");
            prev = r;
            // Every frame carries one shape per set.
            assert_eq!(f.shapes.len(), 3);
        }

        // Init/Mds/Final minimize different objectives, so costs aren't
        // comparable across stages. Check progress within the final stage: its
        // last cost should not exceed its first.
        let final_costs: Vec<f64> = rec
            .frames
            .iter()
            .filter(|f| f.stage == Stage::Final)
            .map(|f| f.cost)
            .collect();
        assert!(final_costs.len() >= 2);
        assert!(
            *final_costs.last().unwrap() <= final_costs[0] + 1e-12,
            "final stage cost should not increase overall"
        );
    }

    #[test]
    fn recording_is_deterministic_for_a_seed() {
        let spec = three_set_spec();
        let a = Fitter::<Circle>::new(&spec)
            .seed(7)
            .fit_recording()
            .unwrap();
        let b = Fitter::<Circle>::new(&spec)
            .seed(7)
            .fit_recording()
            .unwrap();

        assert_eq!(a.frames.len(), b.frames.len());
        let fa = a.frames.last().unwrap();
        let fb = b.frames.last().unwrap();
        for (ca, cb) in fa.shapes.iter().zip(fb.shapes.iter()) {
            assert_eq!(ca.center().x(), cb.center().x());
            assert_eq!(ca.center().y(), cb.center().y());
            assert_eq!(ca.radius(), cb.radius());
        }
    }

    #[test]
    fn recording_works_for_ellipses() {
        let spec = three_set_spec();
        let rec = Fitter::<Ellipse>::new(&spec)
            .seed(1)
            .fit_recording()
            .unwrap();
        assert!(rec.frames.iter().any(|f| f.stage == Stage::Final));
        // Init/Mds frames are circle-equivalent ellipses (semi-axes equal).
        let init = &rec.frames[0];
        for e in &init.shapes {
            assert!((e.semi_major() - e.semi_minor()).abs() < 1e-9);
        }
    }
}
