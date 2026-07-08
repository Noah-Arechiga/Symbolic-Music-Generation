# Symbolic Music Generation
 
Symbolic (ABC notation / MIDI) folk-melody generation on the Nottingham dataset, built around two tasks:
 
1. **Unconditional generation** — train/evaluate several models to generate new folk melodies from scratch.
2. **Conditional generation** — style-aware continuation: detect a seed melody's style (jig, reel, hornpipe, waltz) and generate a continuation that respects it.
Each task trains and compares multiple model families side by side (Markov chains, an LSTM/scratch Transformer, a fine-tuned GPT-2, and pretrained ABC-native models such as TunesFormer / MelodyT5), scores them with a shared set of objective metrics, and writes out CSVs, plots, and sample MIDI/ABC outputs.
 
## Repository Structure
 
```
submission/
├── code/
│   ├── task1_unconditional_generation/
│   │   ├── run_experiments.py      # main entry point for Task 1
│   │   ├── config_modes.yaml       # smoke / medium / full run configs
│   │   └── requirements.txt
│   └── task2_conditional_generation/
│       ├── run_task2.py            # main entry point for Task 2
│       ├── config_task2.yaml       # smoke / medium / full run configs
│       └── requirements.txt
├── results/
│   ├── task1/                      # metrics CSVs, plots, sample outputs
│   └── task2/                      # metrics CSVs, plots, showcase ABC/audio
├── SUBMIT_THESE/                   # final deliverables (MIDI samples, workbook export)
├── workbook.ipynb                  # notebook that ties results together for the report
├── build_workbook.py               # script that (re)builds workbook.ipynb
└── REPORT_REFERENCES_AND_METRICS.md  # write-up: references, baselines, metric definitions
```
 
## Setup
 
Each task has its own `requirements.txt` since Task 1 and Task 2 use slightly different dependency sets.
 
```bash
# Task 1
cd submission/code/task1_unconditional_generation
pip install -r requirements.txt
 
# Task 2
cd submission/code/task2_conditional_generation
pip install -r requirements.txt
```
 
A CUDA GPU is recommended for the `medium`/`full` run modes (GPT-2 fine-tuning and Transformer training), but the `smoke` mode is designed to run on CPU for a quick pipeline check.
 
## Task 1 — Unconditional Generation
 
Trains/evaluates models on Nottingham ABC tunes and generates new, unconditioned melodies.
 
```bash
cd submission/code/task1_unconditional_generation
python run_experiments.py --mode smoke
python run_experiments.py --mode medium
python run_experiments.py --mode full
python run_experiments.py --mode smoke --models markov,lstm
```
 
**Arguments**
 
| Flag | Description | Default |
|---|---|---|
| `--mode` | `smoke`, `medium`, or `full` — controls dataset size, epochs, and generation counts (see `config_modes.yaml`) | `smoke` |
| `--models` | Comma-separated subset of `markov,lstm,gpt2,tunesformer,melodyt5` | all five |
| `--seed` | Random seed | `42` |
| `--run-dir` | Output directory override | auto-generated |
 
**Models compared:** n-gram/Markov chain baselines, a character-level LSTM, a fine-tuned GPT-2, and two pretrained ABC-native Transformers (TunesFormer, MelodyT5).
 
**Metrics:** NLL / perplexity, grammar validity, pitch count, pitch-class entropy, pitch range, average pitch interval, note count, average inter-onset interval, scale consistency, and Overlapping Area (OA) against the real test-set distribution. Full definitions are in [`submission/REPORT_REFERENCES_AND_METRICS.md`](submission/REPORT_REFERENCES_AND_METRICS.md).
 
## Task 2 — Conditional Generation
 
Given a seed melody, a classifier detects its style, then each generator produces a continuation conditioned on that style plus the seed bars.
 
```bash
cd submission/code/task2_conditional_generation
python run_task2.py --mode smoke --models all --skip_melodyt5
python run_task2.py --mode medium --models markov,transformer,gpt2,vae --skip_melodyt5
```
 
**Arguments**
 
| Flag | Description | Default |
|---|---|---|
| `--mode` | `smoke`, `medium`, or `full` | `smoke` |
| `--models` | Comma-separated subset, or `all` | `all` |
| `--skip_melodyt5` | Skip the MelodyT5 model (it's slow to load) | off |
| `--seed` | Random seed | `42` |
| `--output_dir` | Output directory override | auto-generated |
 
**Styles:** jig (6/8), reel (4/4), hornpipe (4/4), waltz (3/4).
 
**Models compared:** Markov baseline, a from-scratch Transformer, a fine-tuned GPT-2, and a VAE.
 
**Metrics:** style accuracy, meter consistency, seed-key consistency, structural validity, intra-style and cross-style diversity, pitch-range naturalness, MIDI validity, and perplexity (where applicable). Full definitions are in [`submission/REPORT_REFERENCES_AND_METRICS.md`](submission/REPORT_REFERENCES_AND_METRICS.md).
 
## Results Summary
 
**Task 1** (medium run, `submission/results/task1/task1_medium_metrics_final_merged.csv`):
 
| Model | Test NLL | Test PPL | OA Overall | Valid MIDI |
|---|---:|---:|---:|---:|
| Markov (k=3, α=0.1) | 1.490 | 4.436 | 0.257 | 20/20 |
| Markov (k=2, α=0.2) | 1.590 | 4.903 | 0.271 | 20/20 |
| CharLSTM | 0.996 | 2.708 | 0.210 | 20/20 |
| GPT-2 fine-tuned | — | — | 0.124 | 19/20 |
| TunesFormer | — | — | 0.102 | 20/20 |
| MelodyT5 | — | — | 0.302 | 19/20 |
 
**Task 2** (medium run, `submission/results/task2/task2_medium_model_summary.csv`):
 
| Model | Valid MIDI | Style Accuracy | Composite (no PPL) | Perplexity |
|---|---:|---:|---:|---:|
| GPT-2 fine-tuned | 51/51 | 0.958 | 0.879 | — |
| VAE | 51/51 | 0.908 | 0.868 | — |
| Markov baseline | 51/51 | 0.850 | 0.862 | 6.924 |
| Scratch Transformer | 51/51 | 0.813 | 0.851 | 5.441 |
 
See [`submission/REPORT_REFERENCES_AND_METRICS.md`](submission/REPORT_REFERENCES_AND_METRICS.md) for the important caveat on comparing these numbers to published baselines (representation/tokenization differences mean scores aren't strictly apples-to-apples across papers).
 
## Data
 
Both tasks use the **Nottingham folk tune collection** in ABC notation (the cleaned JukeDeck GitHub version), a standard symbolic-music benchmark introduced in Boulanger-Lewandowski et al. (2012).
 
## Sample Outputs
 
Generated audio/MIDI/ABC samples live under `submission/results/task1/` and `submission/results/task2/`, and the final selected deliverables are in `submission/SUBMIT_THESE/` (unconditioned and conditioned MIDI samples plus an exported HTML workbook).
 
## References
 
- Boulanger-Lewandowski, Bengio & Vincent (2012), *Modeling Temporal Dependencies in High-Dimensional Sequences* — [arXiv:1206.6392](https://arxiv.org/abs/1206.6392)
- Yang & Lerch (2018), *On the Evaluation of Generative Models in Music* (MGEval)
- Sturm et al. (2016), *FolkRNN* — [arXiv:1604.08723](https://arxiv.org/abs/1604.08723)
- Huang et al. (2018/2019), *Music Transformer* — [arXiv:1809.04281](https://arxiv.org/abs/1809.04281)
- Geerlings & Meroño-Peñuela (2020), *Interacting with GPT-2 to Generate Controlled and Believable Musical Sequences in ABC Notation*
- Wu et al. (2023), *TunesFormer* — [arXiv:2301.02884](https://arxiv.org/abs/2301.02884)
- Wu et al. (2024), *MelodyT5* — [arXiv:2407.02277](https://arxiv.org/abs/2407.02277)
Full reference list and metric derivations: [`submission/REPORT_REFERENCES_AND_METRICS.md`](submission/REPORT_REFERENCES_AND_METRICS.md).
