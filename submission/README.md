# Final Submission — Read Me First

This folder is the cleaned, submission-ready version of the project. Two tasks are covered:
**Task 1 — symbolic unconditioned generation** and **Task 2 — symbolic conditioned generation**,
both on the Nottingham folk corpus (ABC → MIDI).

---

## ✅ WHAT YOU ACTUALLY SUBMIT (4 files, uploaded individually to Gradescope — NOT a zip)

Everything you submit is already collected in **`SUBMIT_THESE/`**:

| File | What it is | Status |
|---|---|---|
| `workbook.html` | The documented analysis notebook, exported to HTML. Opens in a browser, readable without running. Starts with `<!DOCTYPE html>` (passes the autograder). | ✅ Done |
| `symbolic_unconditioned.mid` | Best Task 1 generated tune. | ✅ Done |
| `symbolic_conditioned.mid` | Best Task 2 conditioned continuation. | ✅ Done |
| `video_url.txt` | One line: your Google Drive / YouTube link to the ~20-min video. | ⚠️ **You must edit this** after recording (see below) |

> The big project tree (`code/`, `results/`) is **not** submitted — it's kept here for your own
> provenance and in case you want to re-run. Gradescope only needs the four files above.

---

## ⚠️ THE TWO THINGS LEFT FOR YOU TO DO

1. **Record the ~20-minute video.**
   - Use the slide-by-slide script in **`PRESENTATION_NOTES.md`** (one slide per rubric subsection,
     timed to ~20 min, covering all 4 sections × 2 tasks).
   - Narrate while scrolling the matching part of `workbook.html`.
   - Play your music at the very end (this does **not** count toward the 20 minutes).
   - Export as **mp4**.

2. **Upload the video and paste the link.**
   - Google Drive: upload the mp4 → Share → "Anyone with the link" → copy link, **or** use a
     YouTube link.
   - Open `SUBMIT_THESE/video_url.txt` and replace the placeholder with that single link.
   - The autograder checks the file is downloadable, >1 MB, and `video/mp4`.

That's it — then upload the four files in `SUBMIT_THESE/` to Gradescope.

---

## Folder layout

```
SUBMIT_THESE/                 <- the 4 files that go to Gradescope
  workbook.html
  symbolic_unconditioned.mid
  symbolic_conditioned.mid
  video_url.txt               (edit this)

PRESENTATION_NOTES.md         <- slide-by-slide script for the video
workbook.ipynb                <- source notebook for workbook.html (provenance)
build_workbook.py             <- script that regenerates workbook.html
REPORT_REFERENCES_AND_METRICS.md  <- references, baselines, metric equations

code/
  task1_unconditional_generation/  run_experiments.py, config_modes.yaml, requirements.txt
  task2_conditional_generation/    run_task2.py, config_task2.yaml, requirements.txt

results/
  task1/  metrics CSVs + plots (real_pch, lstm_training, task1_comparison)
  task2/  metrics CSVs + plots (training_loss_curves, metrics_comparison_heatmap,
          classifier_confusion_matrix) + showcase ABC + narrated audio reels
```

---

## How `workbook.html` was produced (so you can regenerate it)

`workbook.html` is `workbook.ipynb` exported to HTML with outputs baked in:

```bash
cd <this folder>
python build_workbook.py                       # writes workbook.ipynb
jupyter nbconvert --to html --execute workbook.ipynb
cp workbook.html SUBMIT_THESE/workbook.html
```

The executed cells read the real result CSVs and plots from `results/`, so the tables and figures
in the HTML are the actual medium-run outputs.

---

## How to reproduce the model results (optional, needs Kaggle T4 + internet)

```bash
cd code/task1_unconditional_generation
python run_experiments.py --mode medium --models markov,lstm,gpt2,tunesformer,melodyt5

cd ../task2_conditional_generation
python run_task2.py --mode medium --models markov,transformer,gpt2,vae --skip_melodyt5
```

---

## Note for later: peer grading
Separate deliverable, due ~1 week after submission (grade 4 assignments, worth 4 marks). Nothing
to prepare now.
