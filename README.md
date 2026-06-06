# CS131 Final: Team-Labeled MOT on SportsMOT (Basketball)

Classical CV pipeline for the Spring 2026 final project (proposal + milestone + final report).

## While SportsMOT is downloading

Use the built-in **dev sequence** (no network):

```bash
source .venv/bin/activate
python scripts/make_dev_sequence.py
python scripts/run_detection.py
python scripts/run_tracking.py
pdflatex milestone.tex   # uses dev figures under outputs/dev_basketball_synth/
```

Optional: extract any local basketball MP4 you already have:

```bash
python scripts/mp4_to_mot.py ~/Downloads/some_clip.mp4 --name dev_local_mp4
python scripts/run_detection.py --sequence dev_local_mp4
```

When the real zip finishes, set `SEQUENCE_NAME` in `config.py` and re-run the same scripts.

## SportsMOT download (for final experiments)

1. Register and join the [SportsMOT Codalab competition](https://codalab.lisn.upsaclay.fr/competitions/12424#participate).
2. Under **Participate → Get Data**, download the **train** split (CC BY-NC 4.0).
3. Unzip so frames live at:

   ```
   data/dataset/train/<SEQUENCE_NAME>/img1/000001.jpg
   data/dataset/train/<SEQUENCE_NAME>/gt/gt.txt
   ```

4. Pick a basketball sequence from `splits_txt/basketball.txt` in the zip (or use the default in `config.py`).
5. Set `SEQUENCE_NAME` in `config.py` if needed.

Smaller sample: the [OneDrive example](https://1drv.ms/u/s!AtjeLq7YnYGRgQRrmqGr4B-k-xsC?e=7PndU8) from the SportsMOT README is enough for the milestone if the full train zip is slow to download.

## Setup

```bash
cd /path/to/CS131_final
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Milestone: generate figures + metrics

```bash
python scripts/run_detection.py
```

Outputs:

- `outputs/<sequence>/figures/panel_mog2_vs_gt.png` — 2×2 panel (RGB, mask, detections, detections+GT)
- `outputs/<sequence>/figures/timeline_mog2.png` — sampled frames with boxes
- `outputs/<sequence>/metrics.json` — MOG2 vs frame-diff precision @ IoU 0.3 on GT frames

## Build reports (LaTeX; local only, not in git)

CVPR template files live in `latex/` (from [cvpr-org/author-kit](https://github.com/cvpr-org/author-kit)).

```bash
pdflatex milestone.tex
bash scripts/build_final_report.sh   # ablation plots + report_paths.tex, then pdflatex (CVPR format)
```

The final report is a **4-page main body** (Introduction, Related Work, Methodology, Results, Conclusion) plus references and a **Supplementary Figures** appendix. Non-essential visuals (MOG2 comparison, training curve, overlay strips) are in the appendix.

## M3 ablations (final-report experiments)

```bash
python scripts/sweep_e2e_ablations.py   # ~2–3 h CPU: k-means k + 8 e2e configs
python scripts/eval_learned_e2e.py      # includes bootstrap 95% CI on current checkpoint
python scripts/update_report_ablations.py   # writes ablation_kmeans_k.png, ablation_e2e.png
```

Results land in `outputs/<sequence>/ablations/{kmeans_k.json,e2e_ablations.json}`; bar plots are written to `outputs/<sequence>/figures/`.

## Timeline (solo)

| Date | Target |
|------|--------|
| 5/22 | Milestone: detection v0 + visuals + updated plan |
| 5/23–5/28 | IoU/Hungarian tracking, ID-switch metric |
| 5/29–6/2 | LAB k-means team labels, ablations, demo slides |
| 6/6 | CVPR-format final report |
