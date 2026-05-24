---
title: "feat: Temporal Team-ID Denoising on YOLO+IoU Tracks"
type: feat
status: active
date: 2026-05-21
origin: docs/brainstorms/temporal-team-id-requirements.md
---

# feat: Temporal Team-ID Denoising on YOLO+IoU Tracks

## Summary

Replace the proposal's MOG2/GrabCut detection path with pretrained YOLOv8n, keep the existing greedy IoU tracker as the association backbone, and add a small BiLSTM that denoises per-frame LAB color-histogram k-means team labels into one stable team identity per track. Trained self-supervised against per-track majority-vote pseudo-labels; compared against the same pipeline without the BiLSTM. One deliberately broken MOG2 figure on a panning subsequence answers the proposal reviewer's first critique.

---

## Problem Frame

CS131 final project mid-flight pivot in response to proposal feedback. Brief framing only — full context in origin doc.

---

## Requirements

- R1. Pipeline runs end-to-end on one real SportsMOT basketball clip (30–120 s) on CPU.
- R2. Learned temporal model achieves measurably higher **team purity** than the per-frame LAB k-means + temporal-majority-vote baseline on the same tracks.
- R3. Report contains a MOG2-vs-YOLO panning-failure figure (reviewer critique #1).
- R4. Pipeline retains the classical-CV identity: exactly one off-the-shelf learned component (YOLO); embedding stays LAB-histogram; tracker stays classical.
- R5. Fallback: if BiLSTM training is unstable, the same pipeline minus the BiLSTM ships as the headline. Hard gate at 6/3.
- R6. Secondary metrics reported: ID switches per minute, track fragmentation.

---

## Scope Boundaries

- No motion-compensated MOG2, GrabCut, or frame-differencing as a serious detector.
- No hand-labeled supervised training of the temporal model.
- No unsupervised contrastive clustering across tracks.
- No detection-refinement, learned-association, or court-homography variants.
- No pretrained CNN / ResNet / CLIP embeddings.
- No court-homography occupancy heatmap stretch.
- No re-identification or pose estimation.
- No cross-sequence generalization — single clip is sufficient.

### Deferred to Follow-Up Work

- Kalman filtering on tracker: included only if time permits before 6/3; otherwise greedy IoU stays.
- Multiple evaluation sequences: post-final if at all.

---

## Context & Research

### Relevant Code and Patterns

- `src/detect.py` — MOG2 detector + IoU helper. Keep as-is; reused for the broken-baseline figure.
- `src/track.py` — `GreedyTracker` with IoU threshold + max-missed. The tracking backbone; consumes any box source (MOG2 or YOLO).
- `src/mot_io.py` — `Box` dataclass and MOT-format GT reader. New code reuses `Box`.
- `src/visualize.py` — overlay rendering. Extend for team-colored overlays.
- `scripts/run_detection.py` / `scripts/run_tracking.py` — entry points. Add `scripts/run_yolo.py`, `scripts/run_team_id.py`, `scripts/train_team_model.py`.
- `config.py` — central path/param config. Extend with YOLO and team-model params; do not scatter constants.
- `outputs/<sequence>/figures/` — established convention for generated figures.

### Institutional Learnings

- None applicable (first project in this codebase).

### External References

- `ultralytics` YOLOv8 docs for CPU inference and person-class filtering.
- PyTorch BiLSTM standard idioms (no novel architecture; well-trodden).

---

## Key Technical Decisions

- **Detector = YOLOv8n CPU, person class only, detections cached to disk.** Decouples slow inference from fast iteration on tracker/embedder/model. One YOLO run per sequence; everything downstream reads the cache.
- **Tracker stays greedy IoU.** Hungarian assignment offers marginal gains for short basketball clips with dense detections; not worth the 2-day diff in a 16-day project. Kalman deferred.
- **Embedding = LAB 4×8×8 = 256-bin histogram on torso ROI** (top 60% height, middle 60% width of player box). Preserves classical identity; matches what k-means clusters on; small enough to feed an LSTM efficiently.
- **Three-class output {A, B, other}.** Avoids a confidence-threshold tuning step and lets the model learn ref/coach embedding signatures.
- **Pseudo-label = per-track majority vote of per-frame k-means cluster assignments**, with k-means run *once globally* on all frame-embeddings in the clip (not per-frame). Global k-means gives stable cluster identities; per-frame would have label-swap problems.
- **Architecture = 1-layer BiLSTM(64) → mean-pool → MLP → 3-class softmax.** Smallest model that exploits temporal structure. ~10k params. Trains in minutes on CPU.
- **Loss = cross-entropy against majority-vote pseudo-label, weighted by track length** (longer tracks have more reliable pseudo-labels).
- **Train/eval split = same clip.** Acceptable because the model's job is denoising the input it's given, not generalization. Report this explicitly as a limitation.
- **PyTorch added to `requirements.txt`** — accepted cost for one learned component.

---

## Open Questions

### Resolved During Planning

- Architecture, loss, pseudo-label source, embedding dims, class count, Kalman scope: all resolved above.

### Deferred to Implementation

- Exact YOLO confidence/NMS thresholds — tune empirically against GT IoU once on the chosen sequence.
- Whether the BiLSTM benefits from temporal subsampling of long tracks (>500 frames) — decide after seeing first training curves.
- Whether to L2-normalize histograms in addition to L1 — try both during U6.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
SportsMOT clip (img1/*.jpg)
        │
        ▼
[U1] YOLOv8n CPU  ──►  detections.parquet   (cached, per-sequence)
        │
        ▼
[U2] GreedyTracker  ──►  tracks.parquet     (track_id, frame, box)
        │
        ▼
[U3] LAB torso histogram per (track, frame)  ──►  embeddings.npz
        │
        ├──►  [U4]  Global LAB k-means (k=3) ──►  per-frame pseudo-labels
        │                                              │
        │                                              ▼
        │                                    Per-track majority vote
        │                                       (= baseline team-id)
        │
        └──►  [U6]  BiLSTM(64) → meanpool → MLP → softmax
                     trained on majority-vote pseudo-labels
                     (= contribution team-id)
        │
        ▼
[U5] team_purity, id_switches/min, track_fragmentation   (both methods)
        │
        ▼
[U7] Figures: MOG2 panning-failure, team-colored overlays,
             confusion-matrix-style purity comparison
```

---

## Implementation Units

- U1. **YOLOv8n detection backend with on-disk cache**

**Goal:** Replace MOG2 as the primary detector. Produce a cached detections file per sequence so downstream units iterate fast.

**Requirements:** R1, R3, R4

**Dependencies:** None.

**Files:**
- Create: `src/yolo_detect.py`
- Create: `scripts/run_yolo.py`
- Modify: `requirements.txt` (add `ultralytics`, `pandas`, `pyarrow`, `torch`)
- Modify: `config.py` (add `YOLO_MODEL = "yolov8n.pt"`, `YOLO_CONF = 0.25`, `YOLO_IOU = 0.5`, `YOLO_PERSON_CLASS = 0`)
- Test: `tests/test_yolo_detect.py`

**Approach:**
- `src/yolo_detect.py` exposes `detect_sequence(sequence_dir, out_path)` that iterates frames, runs YOLO with person-class filter, writes a parquet file with columns `(frame, left, top, width, height, conf)`.
- `scripts/run_yolo.py` is a thin CLI wrapper reading `config.SEQUENCE_NAME`.
- Cache format = parquet (small, fast random access by frame).

**Patterns to follow:**
- `scripts/run_detection.py` for CLI structure and output-dir conventions.
- `src/mot_io.py::Box` for box representation downstream.

**Test scenarios:**
- Happy path: running on the dev synth sequence produces a non-empty parquet with strictly positive widths/heights and frame ids in `[FRAME_START, FRAME_END]`.
- Edge case: empty frame (no detections) is written as zero rows for that frame, not a missing key — downstream code must handle "frame present, no boxes."
- Integration: re-running `run_yolo.py` with an existing cache file is idempotent (overwrites cleanly, same output bytes given same model + clip).

**Verification:**
- `detections.parquet` exists at `outputs/<sequence>/detections.parquet`.
- Eyeball-check via existing overlay tool: detections on real SportsMOT frames visibly cover on-court players.

---

- U2. **Tracker on YOLO detections + ID-switch metric**

**Goal:** Run the existing greedy IoU tracker on cached YOLO detections; compute ID switches per minute against SportsMOT GT.

**Requirements:** R1, R6

**Dependencies:** U1.

**Files:**
- Create: `src/track_metrics.py`
- Create: `scripts/run_tracking_yolo.py`
- Modify: `scripts/run_tracking.py` (optional: add `--source yolo|mog2` flag; or leave alone and ship a separate script — implementer's call)
- Test: `tests/test_track_metrics.py`

**Approach:**
- Reuse `src/track.py::GreedyTracker` unchanged. Input: list of `Box` per frame, loaded from `detections.parquet`.
- `src/track_metrics.py::id_switches_per_minute(predicted_tracks, gt_tracks, fps)` — match predicted tracks to GT via majority-IoU per track, count GT-ID changes within each predicted track.
- Also compute track fragmentation = number of predicted tracks assigned to each GT identity.
- Output: `tracks.parquet` with columns `(frame, track_id, left, top, width, height)` plus `metrics.json`.

**Patterns to follow:**
- `src/detect.py::iou` for box overlap.
- `src/mot_io.py::read_gt` for GT loading.

**Test scenarios:**
- Happy path: synthetic 2-track, 10-frame fixture with no swaps → `id_switches == 0`.
- Edge case: same fixture with a deliberate GT-ID swap inside one predicted track → `id_switches == 1`.
- Edge case: predicted track with no GT overlap is excluded from the metric, not counted as a switch.
- Integration: full pipeline on dev sequence produces a finite, non-negative `id_switches/min` value written to `metrics.json`.

**Verification:**
- `tracks.parquet` and `metrics.json` materialized; ID-switch count is a plain integer in the JSON.

---

- U3. **LAB torso-histogram embedding per (track, frame)**

**Goal:** For every (track, frame) pair, compute a 256-dim LAB histogram of the torso ROI. Store as a single `.npz` keyed by (track_id, frame).

**Requirements:** R2, R4

**Dependencies:** U2.

**Files:**
- Create: `src/embed.py`
- Create: `scripts/run_embed.py`
- Modify: `config.py` (add `HIST_BINS = (4, 8, 8)`, `TORSO_TOP = 0.0`, `TORSO_BOT = 0.6`, `TORSO_LEFT = 0.2`, `TORSO_RIGHT = 0.8`)
- Test: `tests/test_embed.py`

**Approach:**
- `src/embed.py::torso_crop(frame_bgr, box)` returns the cropped torso region using the config ratios.
- `src/embed.py::lab_histogram(crop_bgr, bins)` converts to LAB via `cv2.cvtColor`, computes a joint 3D histogram with `cv2.calcHist`, flattens, L1-normalizes, returns a (256,) float32 vector.
- `scripts/run_embed.py` iterates `tracks.parquet`, loads frames, crops, embeds. Output: `embeddings.npz` with arrays `track_ids`, `frame_ids`, `embeddings` (N×256).

**Patterns to follow:**
- `src/mot_io.py::read_frame` for image loading.
- Reuse `Box` dataclass.

**Test scenarios:**
- Happy path: known solid-red and solid-blue synthetic crops produce histograms whose L1 distance is large (>1.0 with both L1-normalized to sum=1).
- Edge case: zero-area or out-of-frame box → returns a uniform histogram and is logged; does not crash.
- Edge case: torso crop that lands partially off-image is clamped to image bounds.
- Integration: `embeddings.npz` row count equals number of (track, frame) entries in `tracks.parquet`.

**Verification:**
- `embeddings.npz` materialized; spot-check a handful of crops visually via a debug overlay script (no need to commit).

---

- U4. **Global k-means baseline + per-track majority vote**

**Goal:** Produce the comparison baseline: assign every (track, frame) a cluster from global k-means(k=3), then majority-vote within each track to produce one team label per track.

**Requirements:** R2, R5

**Dependencies:** U3.

**Files:**
- Create: `src/baseline.py`
- Create: `scripts/run_baseline.py`
- Modify: `config.py` (add `KMEANS_K = 3`, `KMEANS_SEED = 0`)
- Test: `tests/test_baseline.py`

**Approach:**
- `src/baseline.py::cluster_embeddings(embeddings, k, seed)` runs `sklearn.cluster.KMeans` (add `scikit-learn` to requirements) on the full N×256 matrix; returns cluster ids and centroids.
- `src/baseline.py::majority_vote(track_ids, cluster_ids)` returns a dict `track_id -> team_label`.
- Cluster labels are re-mapped post-hoc so that the two largest clusters become {A, B} and the smallest becomes "other".
- Output: `baseline_labels.json` mapping `track_id -> {"team": "A"|"B"|"other", "per_frame_labels": [...], "vote_confidence": float}`.

**Patterns to follow:**
- `src/mot_io.py` for JSON output style (compact, repo-relative-friendly).

**Test scenarios:**
- Happy path: synthetic embeddings drawn from three well-separated Gaussians → k=3 recovers the three groups; majority vote on a track sampled mostly from one Gaussian assigns that Gaussian's label.
- Edge case: track with exactly tied votes between two clusters → deterministic tiebreak (lowest cluster id); recorded `vote_confidence < 0.5` so it can be filtered downstream.
- Edge case: track of length 1 → vote equals the single frame's cluster; confidence = 1.0.

**Verification:**
- `baseline_labels.json` materialized; visual overlay (boxes colored by team) on the dev sequence looks plausible.

---

- U5. **Team-purity metric + hand-label ground truth**

**Goal:** Implement the primary evaluation metric and capture a small hand-labeled GT to evaluate against.

**Requirements:** R2, R6

**Dependencies:** U4.

**Files:**
- Create: `src/eval_team.py`
- Create: `scripts/label_teams.py` (interactive labeling helper — shows torso crops every N frames, user types A/B/O)
- Create: `data/labels/<sequence>_team_gt.json` (hand-labeled — created during execution, not by this plan)
- Test: `tests/test_eval_team.py`

**Approach:**
- `src/eval_team.py::team_purity(predicted_labels, gt_labels)` — for each predicted track, look up GT team label; purity = (correctly-labeled predicted tracks) / (total predicted tracks with a GT). Report per-class purity too.
- `scripts/label_teams.py` is the labeling tool: for each predicted track, sample 3 evenly-spaced frames, display the torso crops with `cv2.imshow`, prompt for `A` / `B` / `O` / `?`. Skip tracks marked `?`. Write `<sequence>_team_gt.json`.
- Target ~5–10 labeled tracks per the proposal; this is enough for stable purity since each track contributes one label.

**Patterns to follow:**
- Existing config-driven script structure.

**Test scenarios:**
- Happy path: 5 predicted tracks, GT labels match 4 of 5 → purity = 0.8.
- Edge case: predicted track with no GT entry is excluded from denominator, not counted against purity.
- Edge case: GT label "other" against predicted "A" counts as incorrect (not silently dropped).

**Verification:**
- `<sequence>_team_gt.json` exists with at least 5 labeled tracks.
- `team_purity()` on baseline labels returns a finite value in `[0, 1]`.

---

- U6. **Temporal BiLSTM team-ID model: define, train, evaluate**

**Goal:** Implement the contribution — a small BiLSTM trained to denoise majority-vote pseudo-labels into per-track team identities. Save checkpoint and predicted labels.

**Requirements:** R2, R4, R5

**Dependencies:** U4 (baseline produces training labels), U5 (eval harness).

**Execution note:** Start with the smallest viable architecture and verify it can overfit a single track before scaling up. If at any point by 6/3 the model is not beating the majority-vote baseline by a meaningful margin, **stop** and document the negative result for the fallback story (R5).

**Files:**
- Create: `src/team_model.py` (model definition only — `TeamBiLSTM(nn.Module)`)
- Create: `src/team_train.py` (training loop, dataset, checkpoint I/O)
- Create: `scripts/train_team_model.py`
- Create: `scripts/run_team_id.py` (loads checkpoint, predicts per-track labels, writes `learned_labels.json`)
- Modify: `config.py` (add `LSTM_HIDDEN = 64`, `LSTM_EPOCHS = 50`, `LSTM_LR = 1e-3`, `LSTM_MAX_SEQ_LEN = 500`)
- Test: `tests/test_team_model.py`

**Approach:**
- Model: `nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, batch_first=True)` → mean-pool over time → `nn.Linear(128, 64) + ReLU` → `nn.Linear(64, 3)`.
- Dataset: each item is one track's embedding sequence (T×256) plus its majority-vote label. Tracks longer than `LSTM_MAX_SEQ_LEN` are sub-sampled uniformly.
- Loss: cross-entropy weighted by track length (longer = higher weight).
- No train/val split — model trains on the same clip it predicts on. Document as a limitation; the BiLSTM cannot "memorize" the label because the label is *derived* from the same embeddings via a different mechanism, so improvement-over-majority-vote is a meaningful signal.
- Checkpoint to `outputs/<sequence>/team_model.pt`. Predictions to `outputs/<sequence>/learned_labels.json` in the same schema as `baseline_labels.json`.

**Technical design:** *(directional, not implementation specification)*

```
Input  : (B, T, 256) — batch of variable-length track embedding sequences
LSTM   : (B, T, 128) bidirectional, 64 hidden each direction
Pool   : (B, 128)    — masked mean over valid timesteps
MLP    : (B, 3)      — team logits {A, B, other}
Loss   : CE(logits, majority_vote_label), weight = log(1 + track_len)
```

**Patterns to follow:**
- Standard PyTorch idioms; no project precedent.

**Test scenarios:**
- Happy path: tiny synthetic dataset of 3 tracks with cleanly separable embedding sequences → model achieves 100% train accuracy in <100 epochs.
- Edge case: track of length 1 is handled by the dataset (returns a (1, 256) tensor, mean-pool degenerates correctly).
- Edge case: dataset with one track per class produces a valid training step (no NaN, no crash on tiny batch).
- Integration: end-to-end `train_team_model.py` produces a checkpoint file; `run_team_id.py` loads it and emits `learned_labels.json` matching the schema of `baseline_labels.json` (so the eval code from U5 works unchanged on both).

**Verification:**
- `team_model.pt` and `learned_labels.json` materialized.
- `team_purity(learned_labels, gt_labels) > team_purity(baseline_labels, gt_labels)`, or the negative result is documented in the final report per R5.

---

- U7. **Figures and qualitative outputs**

**Goal:** Produce the report-grade figures: MOG2 panning-failure, team-colored track overlays for baseline vs. learned, and a small purity comparison plot.

**Requirements:** R3, R2

**Dependencies:** U1, U4, U6.

**Files:**
- Create: `scripts/make_figures.py`
- Modify: `src/visualize.py` (add `team_colored_overlay(frame, track_boxes, team_labels)`)
- Output: `outputs/<sequence>/figures/mog2_vs_yolo_panning.png`, `team_overlay_baseline.png`, `team_overlay_learned.png`, `purity_comparison.png`

**Approach:**
- MOG2 figure: re-run `scripts/run_detection.py` on a deliberately-panning subsequence (or use the dev synth if SportsMOT panning clip is not available — annotate as such) and stitch with YOLO output on the same frame.
- Overlay figures: pick one representative mid-clip frame; color boxes red/blue/grey for A/B/other.
- Purity bar plot: matplotlib bar of baseline vs. learned purity, plus per-class breakdown.

**Patterns to follow:**
- `src/visualize.py` existing overlay helpers.

**Test scenarios:**
- Test expectation: none — this unit produces figures for human review only. Pure visualization code with no behavioral contract to assert; visual inspection is the verification.

**Verification:**
- All four PNGs land under `outputs/<sequence>/figures/`.
- Figures are readable when inserted into the final report (no font-too-small, no clipped axes).

---

## System-Wide Impact

- **Interaction graph:** `config.py` becomes the single source of truth for new params (YOLO, embedding, k-means, LSTM). All new scripts read from it; nothing hard-codes paths.
- **Error propagation:** YOLO inference failure → cache write fails → downstream scripts fail with a clear "missing detections.parquet" error rather than silently producing empty tracks.
- **State lifecycle risks:** Detections cache must be invalidated when YOLO config changes. Simplest mitigation: cache file path encodes `YOLO_MODEL` + `YOLO_CONF` (e.g., `detections_yolov8n_conf0.25.parquet`) so config tweaks don't silently reuse stale results.
- **API surface parity:** Baseline and learned label files share the same JSON schema so the eval code in U5 works unchanged on both. Do not let these schemas drift.
- **Integration coverage:** End-to-end smoke test (`scripts/run_yolo.py → run_tracking_yolo.py → run_embed.py → run_baseline.py → run_team_id.py`) on the dev synth sequence should produce non-empty outputs at every stage. Run this once after U6 lands.
- **Unchanged invariants:** Existing milestone artifacts (`outputs/dev_basketball_synth/figures/panel_mog2_vs_gt.png`, `timeline_mog2.png`) are not touched — milestone PDF must remain reproducible.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| YOLOv8n CPU inference too slow on a 120 s clip | Cache once; downsample to 10 fps if needed; in worst case, evaluate on a 30 s subsequence (still within proposal's 30–120 s window). |
| Global k-means produces useless clusters (similar jersey colors on chosen sequence) | Pick a sequence with visually distinct jerseys; documented in origin as known failure mode. Pre-screen 3 candidate sequences before committing. |
| BiLSTM does not beat majority-vote baseline | R5 fallback: ship the comparison study without the learned model as the headline. Hard gate at 6/3. |
| Hand-labeling 5–10 tracks takes too long | The labeling tool (U5) shows only 3 sampled frames per track; total labeling time budgeted at <30 min. |
| PyTorch install / CPU-only issues | Pin `torch` to a known CPU wheel; document install command in README. |
| Self-supervised label collapse (model predicts majority class for all tracks) | Class-weighted loss; check confusion matrix during training; track-length weighting reduces bias toward short noisy tracks. |

---

## Documentation / Operational Notes

- Update `README.md` once after U1 lands (new YOLO step, new scripts, new pip deps).
- Final-report figures live in `outputs/<sequence>/figures/`; CVPR-template report assembled in last sprint (6/4–6/6), out of scope for this plan.
- Pre-commit hook not needed; this is research code.

---

## Sources & References

- **Origin document:** `docs/brainstorms/temporal-team-id-requirements.md`
- **Milestone (reflects pivot):** `milestone.tex` § *Response to proposal feedback*, *Revised methodology*
- Related code: `src/detect.py`, `src/track.py`, `src/mot_io.py`, `src/visualize.py`, `config.py`
- External docs: ultralytics YOLOv8 (CPU inference, person class), PyTorch BiLSTM standard usage
