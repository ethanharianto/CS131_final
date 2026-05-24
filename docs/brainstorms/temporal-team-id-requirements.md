# Temporal Team-ID Denoising — Requirements

**Date:** 2026-05-21
**Course:** Stanford CS131 — Spring 2026, Final Project
**Status:** Mid-project pivot in response to proposal feedback. Milestone (due 5/22) updated; this document scopes 5/22 – 6/6.

## Problem

The proposal pitched a fully classical detect-then-track pipeline for SportsMOT basketball, with per-frame LAB k-means as the team-labeling mechanism. Proposal reviewer raised three concerns:

1. Frame differencing assumes a stable background; SportsMOT pans and tilts.
2. GrabCut is too expensive for 30–120 s evaluation segments.
3. Foreground/temporal resolution is under-specified — consider an RNN with convolutional kernels.

The pivot accepts (1) and (2) by replacing the classical detector with a pretrained off-the-shelf YOLO and dropping GrabCut entirely. It accepts (3) by reframing the project's contribution around temporal modeling — but applied to **team labeling**, not detection, since YOLO already solves per-frame detection well. The interesting remaining classical problem is producing a single stable team identity per track from a sequence of noisy per-frame color-based assignments.

## Beneficiary and success criteria

**Beneficiary.** CS131 graders evaluating a final project on classical computer vision that responds to proposal feedback and produces a defensible contribution.

**Success criteria.**

- The pipeline runs end-to-end on a real SportsMOT basketball clip (30–120 s) on CPU.
- A measurable improvement in **team purity** from the learned temporal model over the per-frame LAB k-means + temporal-majority-vote baseline, on the same tracks, same evaluation segment.
- Report includes the requested response-to-feedback figure: MOG2 (or frame-differencing) failing on a panning subsequence vs. YOLO succeeding.
- Fallback: if the learned temporal model is unstable, the report ships as a comparison-study framing in which the majority-vote variant is the headline and the learned model is an attempted-but-unstable extension. The pipeline still ships.

## Scope

### In scope (contribution)

- **Detector:** pretrained YOLO (off-the-shelf, black-box). Treated as a fixed input.
- **Tracker:** greedy/Hungarian IoU association on YOLO boxes, optional Kalman motion model. Already prototyped in `scripts/run_tracking.py`.
- **Embedding:** LAB color histogram of each track's per-frame torso crop. Classical, no learned features.
- **Contribution:** small temporal sequence model (1-layer BiLSTM or temporal 1D CNN) mapping a per-track sequence of LAB-histogram embeddings to one team label per track ∈ {A, B, other}.
- **Training regime:** self-supervised denoising. Weak per-frame labels come from running LAB k-means on the same embeddings; the model is trained to predict the temporally-consistent team identity from the noisy sequence. No new hand-labeling.
- **Baselines:**
    - Per-frame LAB k-means + temporal majority vote (isolates the contribution of the learned temporal model).
    - MOG2 detection on a panning subsequence (one figure, deliberately broken baseline answering reviewer critique #1).
- **Metrics:** team purity (primary), ID switches per minute, track fragmentation.
- **Evaluation:** one 30–120 s SportsMOT basketball sequence with hand-labeled team IDs for ~5–10 tracks across ~10-frame intervals (as proposed).

### Deferred for later

- Multiple sequences / cross-sequence generalization.
- Refs/coaches detection beyond a single "other" class.
- Demo overlays and CVPR-template final report polish (last sprint, 6/4–6/6).

### Outside scope (explicit non-goals)

- Motion-compensated MOG2 (homography warping then differencing). Cost > value for this timeline.
- GrabCut, or any per-frame segmentation step.
- Frame differencing as a serious detector.
- Hand-labeled supervised training of the temporal model.
- Unsupervised contrastive team clustering across tracks.
- Detection refinement, learned association, or court-homography variants of the temporal module (alternatives a, b, d explored and rejected during brainstorm).
- Pretrained CNN / ResNet / CLIP embeddings (LAB histogram keeps the "one learned component" framing).
- Court-homography occupancy heatmap stretch from the original proposal.
- Re-identification or pose estimation (unchanged from proposal — explicitly out).

## Constraints

- **Compute:** CPU only (per proposal). Model must be small enough to train in minutes on a single basketball clip.
- **Classical-CV identity:** exactly one off-the-shelf learned component (YOLO). The temporal model is small, the embedding is LAB-histogram, the tracker is classical. The project remains presentable as primarily-classical CV.
- **Time:** 16 days (5/22 – 6/6). Hard deadline 6/6 with demo and final report.
- **Solo project.**

## Key decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Framing | Hybrid (path C) | YOLO directly resolves panning-camera and GrabCut critiques |
| Project center of gravity | Temporal modeling (path 4) | Leans into reviewer hint; cleanest contribution given YOLO does detection |
| Temporal module attachment | Team-ID smoothing (option c) | Smallest model, lines up with proposal's "team purity" metric, lowest training risk |
| Training regime | Self-supervised denoising of weak labels (option iii) | No new labels needed; baseline is built into training signal; CPU-trainable in minutes |
| Embedding | LAB color histogram (option L) | Preserves classical identity; matches what k-means clusters on |
| Classical detection baseline | Naive MOG2 as broken baseline (option β) | One figure answering reviewer critique #1 without eating a week on motion compensation |

## Assumptions (unverified or to-confirm during implementation)

- A pretrained YOLO checkpoint (e.g., YOLOv8n weights) runs at acceptable speed on CPU for a 30–120 s clip; preliminary check in week 1 of the new timeline.
- LAB k-means on torso crops produces noisy-but-mostly-correct per-frame labels — i.e., enough signal for self-supervised denoising. If per-frame labels are near-random on the chosen clip, the training signal collapses and the fallback path becomes the headline.
- SportsMOT sequence chosen has visually distinct jersey colors (ablation on white-jersey or similar-color sequences is documented as a known failure mode, not a primary evaluation).
- Hand-labeled team IDs for ~5–10 tracks at ~10-frame intervals (as in the proposal) is sufficient to compute team purity reliably.

## Dependencies

- SportsMOT basketball sequence (download in progress per milestone).
- Pretrained YOLO weights (e.g., ultralytics YOLOv8n, CPU inference).
- Existing prototype tracker in `scripts/run_tracking.py`.

## Timeline (5/22 – 6/6)

Mirrors the milestone document's revised timeline. Reproduced here for planning continuity.

- **5/22 – 5/24:** Finish SportsMOT unzip. Integrate pretrained YOLO. Rerun detection on a real basketball sequence. Capture the MOG2-vs-YOLO panning-failure figure.
- **5/25 – 5/27:** Hungarian association (+ optional Kalman) on YOLO boxes. ID-switches/min on evaluation segment.
- **5/28 – 5/30:** LAB-histogram torso embeddings. Per-frame k-means baseline + temporal majority vote. Team-purity metric implemented.
- **5/31 – 6/3:** Train temporal sequence model on weak labels. Compare against majority-vote baseline. Ablations.
- **6/4 – 6/6:** Failure analysis. Demo overlays. CVPR-template final report. Fallback path (no learned temporal model) ready by 6/3.

## Risks

- **Temporal model training instability.** Self-supervised denoising can collapse to "predict the majority of the input" with no real gain over majority-vote, or fail to converge on a small clip. Mitigation: the majority-vote baseline is the same pipeline minus the model — fallback is free.
- **YOLO CPU latency.** If a 30–120 s clip takes hours of inference, pre-extract detections once and cache. Mitigation already accounted for in pipeline structure.
- **Weak-label signal too weak.** If LAB k-means is near-random on the chosen sequence (e.g., similar jersey colors), pick a different sequence; the proposal already names this as a known failure mode.
- **Scope creep into the temporal model architecture.** Cap at: 1 hidden layer, ≤128 hidden units, ≤1k parameters worth of attention. Anything fancier is post-final.

## Open questions for planning

- Exact architecture (1-layer BiLSTM vs. 1D temporal CNN vs. mean-pool-then-MLP).
- Loss formulation: cross-entropy on majority-vote pseudo-label, or a consistency loss across the sequence.
- Whether "other" is a third output class or a confidence-threshold reject.
- Histogram dimensionality and quantization scheme.
- Whether Kalman is worth the integration effort vs. raw IoU association.

All of the above are implementation decisions deferred to ce-plan.
