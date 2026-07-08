#!/usr/bin/env python3
"""Builds workbook.ipynb: a clean, documented analysis notebook for both tasks.

Structure mirrors the assignment rubric exactly:
  For each task -> Intro, then four graded sections
  (1) Data exploration/collection/preprocessing  (2) Modeling
  (3) Evaluation                                  (4) Related work
Each of (1)-(3) is split into Context / Discussion / Code subsections.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))

# ----------------------------------------------------------------------------- TITLE
md(r"""
# Beautiful Music: Symbolic Generation on the Nottingham Folk Corpus

**Tasks attempted (2 of 4):**

1. **Task 1 — Symbolic, *unconditioned* generation.** Learn an unconditional distribution
   $p(x)$ over folk melodies in ABC notation and sample new tunes from it.
2. **Task 2 — Symbolic, *conditioned* generation.** Given an input melody, detect its
   *style* with a trained classifier and generate a brand-new continuation conditioned on
   that detected style **and** the opening bars of the input.

Both tasks share one dataset (Nottingham, in ABC notation) and one output representation
(ABC text → MIDI), which lets us reuse data processing and evaluation machinery across them.

**How to read this notebook.** Each task is laid out in the four graded sections — *Data*,
*Modeling*, *Evaluation*, *Related work* — and the first three are each broken into
**Context / Discussion / Code** so every rubric item is easy to locate. All numbers and plots
below are the *actual* artifacts produced by our `medium` Kaggle (T4 GPU) runs; the heavy
training code lives in `code/task1.../run_experiments.py` and `code/task2.../run_task2.py`
and is excerpted inline so the analysis can be followed without executing anything.
""")

code(r"""
# Lightweight setup -- only used to render the result tables and plots that follow.
import pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
""")

# ============================================================================= TASK 1
md(r"""
---
# TASK 1 — Symbolic, Unconditioned Generation

**Goal.** Train a model of $p(x)$ over Nottingham folk melodies represented as ABC-notation
text, then sample novel melodies and render them to MIDI. We train several models spanning a
complexity spectrum — a **character N-gram Markov chain**, a **character-level LSTM**, and a
**fine-tuned GPT-2** — and additionally run two **pretrained ABC-native Transformers**
(*TunesFormer*, *MelodyT5*) as strong reference points. The real held-out test set is treated
as the *oracle* distribution.
""")

# ---- 1. DATA ----
md(r"""
## 1.1  Data — Context

**Source.** The **Nottingham Music Database** is a long-standing collection of ~1,000+ British
and American folk tunes (jigs, reels, hornpipes, waltzes, etc.). We use the cleaned
**JukeDeck** GitHub release (`jukedeck/nottingham-dataset`), which stores each tune in **ABC
notation** — a compact text format where a header (`X:` index, `T:` title, `M:` meter,
`L:` unit note length, `K:` key) precedes a body of note tokens, bar lines `|`, and repeats.

**Why this dataset.** It is *small, clean, monophonic-friendly, and text-native*, which makes
it ideal for comparing model families on a single GPU: an ABC tune is literally a short string,
so the same corpus supports an N-gram, an RNN, and a Transformer language model with no change
of representation. It is also a classic symbolic-music benchmark (used by
Boulanger-Lewandowski et al., 2012), giving us historical baselines to discuss.
""")

md(r"""
## 1.2  Data — Discussion (pre-processing)

Our pre-processing pipeline (`ensure_nottingham_data`, `make_melody_only`) does the following:

- **Parse** each `.abc` file into individual tunes by splitting on the `X:` index header.
- **Melody-only view.** We strip accompaniment chord symbols (text in quotes, e.g. `"Gm"`)
  with `strip_chords`, collapse whitespace, and **normalize the header** so only musically
  meaningful fields (`X, T, M, L, K`) are kept (`normalize_headers`). This focuses the language
  model on melodic content rather than chord annotations.
- **Document separator.** Tunes are concatenated with an `<|endoftext|>` delimiter so the
  models learn where tunes begin and end (important for sampling complete tunes).
- **Stratified split.** We split **per subset** (jigs, reels, …) into **80/10/10
  train/val/test** with a fixed seed (42), so every style is represented in all three splits
  and the test set is a fair oracle.

The same text is consumed three ways: as a raw character stream (Markov, LSTM), and as
GPT-2 sub-word tokens (GPT-2 fine-tuning).
""")

md(r"""
## 1.3  Data — Code & statistics

Below is the real header-normalisation step that defines our "melody-only" representation,
followed by the **pitch-class histogram of the real test set** (our oracle), which we later use
as the reference distribution for the *Overlapping Area* metric.
""")

md(r"""
```python
# code/task1_unconditional_generation/run_experiments.py
def strip_chords(abc_str: str) -> str:
    return re.sub(r'"[^"]*"', "", abc_str)            # remove "Gm", "D7", ... chord symbols

def normalize_headers(abc_str, keep=("M", "L", "K", "X", "T")):
    # keep only musically meaningful header fields; everything after K: is the tune body
    ...

def make_melody_only(abc_str: str) -> str:
    cleaned = strip_chords(abc_str)
    cleaned = re.sub(r"  +", " ", cleaned)
    return normalize_headers(cleaned)
```
""")

code(r"""
# Real test-set (oracle) pitch-class usage -- the reference distribution for our OA metric.
display(Image("real_pch.png"))

# Summary statistics of the real test set (the oracle row of our Task 1 metrics).
t1 = pd.read_csv("t1_metrics.csv")
oracle = t1[t1["model"].str.contains("Oracle")][
    ["pitch_count_mean", "pch_entropy_mean", "pitch_range_mean",
     "note_count_mean", "scale_consistency_mean"]
].round(3)
oracle.index = ["Real test set (oracle)"]
print("Oracle descriptive statistics (per-tune means):")
oracle.T
""")

# ---- 2. MODELING ----
md(r"""
## 2.1  Modeling — Context

**ML formulation.** We treat a tune as a sequence of tokens $x_{1:T}$ and train an
**autoregressive language model** that factorises
$p(x_{1:T}) = \prod_t p(x_t \mid x_{<t})$.

- **Inputs / outputs:** input is the prefix $x_{<t}$; output is a categorical distribution over
  the next token $x_t$ (characters for Markov/LSTM, BPE sub-words for GPT-2).
- **Objective:** minimise the average **negative log-likelihood (cross-entropy)** of held-out
  tokens — equivalently, minimise **perplexity**.
- **Sampling:** new tunes are generated by temperature-controlled ancestral sampling, then the
  ABC string is cleaned and converted to MIDI.

**Models appropriate for the task,** in order of capacity: a **Markov chain** (counts of which
character follows a fixed-length context) is the trivial-but-real probabilistic baseline; a
**character LSTM** adds unbounded context through a recurrent hidden state; **GPT-2 fine-tuning**
brings a pretrained sub-word Transformer with long-range self-attention. We add **TunesFormer**
and **MelodyT5** as pretrained ABC-native references.
""")

md(r"""
## 2.2  Modeling — Discussion (advantages / disadvantages)

| Model | Pros | Cons |
|---|---|---|
| **Markov (char N-gram)** | Trivial to fit (just counts); exact NLL; surprisingly strong locally | Fixed context window `k`; no long-range structure; can get stuck / repeat |
| **CharLSTM (from scratch)** | Unbounded context via hidden state; cheap to train; **best held-out NLL/PPL** here | Sequential (slow); weaker at very long-range form than attention |
| **GPT-2 fine-tuned** | Pretrained language prior; long-range attention; high grammar validity | Sub-word tokeniser not music-aware; we don't get a clean per-character NLL; heavier to train |
| **TunesFormer / MelodyT5 (pretrained)** | ABC-native, bar-patching/control codes; strong musical structure | **Not trained by us** — used only as references/oracles, *not* our contribution |

**Honesty note on "training our own weights":** Markov, CharLSTM and the GPT-2 fine-tune are
fully fit by us on the Nottingham training split. TunesFormer and MelodyT5 are run from
*published* weights and are clearly labelled as pretrained references — they are not claimed as
models we trained.
""")

md(r"""
## 2.3  Modeling — Code (architectures)

The probabilistic baseline uses add-$\alpha$ (Laplace) smoothing so that unseen context/character
pairs still receive non-zero probability, which is what makes its NLL well-defined:

```python
class MarkovChainABC:
    def _log_prob(self, ctx, ch):                       # add-alpha smoothing
        c = self.transitions.get(ctx, Counter())
        return math.log((c.get(ch,0)+self.alpha) /
                        (sum(c.values()) + self.alpha*self.vocab_size + 1e-12))
    def compute_nll(self, text):                        # mean NLL over held-out chars
        total = sum(-self._log_prob(text[i-self.order:i], text[i])
                    for i in range(self.order, len(text)))
        return total / (len(text) - self.order)
```

The neural baseline is a 2-layer character LSTM with dropout and a tied output projection:

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=256, layers=2, drop=0.3):
        self.emb  = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=drop)
        self.fc   = nn.Linear(hidden, vocab_size)
    def forward(self, x, h=None):
        out, h = self.lstm(self.drop(self.emb(x)), h)
        return self.fc(self.drop(out)), h
```

GPT-2 is fine-tuned on the melody-only corpus chunked into overlapping windows (sub-word
tokens, `<|endoftext|>` between tunes), then sampled with top-k / top-p decoding.
""")

code(r"""
# CharLSTM training vs. validation NLL across epochs (medium run): smooth convergence,
# no over-fitting -- validation tracks training and the gap stays small.
display(Image("lstm_training.png"))
""")

# ---- 3. EVALUATION ----
md(r"""
## 3.1  Evaluation — Context

There is **no single number** that captures "good music", so we evaluate on two axes:

1. **Model-objective metrics — does the model fit the data?**
   - **NLL** $= -\tfrac1N\sum_i \log p(x_i\mid x_{<i})$ and **Perplexity** $=\exp(\text{NLL})$
     (lower = better). These measure held-out likelihood but say nothing *directly* about
     whether the music sounds good.
2. **Musical / distributional metrics — does the *output* look like real folk music?**
   - **Grammar validity:** fraction of generated ABC that parses (via `music21`).
   - **Pitch-class entropy, pitch range, pitch-count, avg. interval, note-count, IOI,**
     and **scale consistency** (fraction of notes in the implied key).
   - **Overlapping Area (OA)** $=\int \min(p_{\text{real}}(x), p_{\text{gen}}(x))\,dx \in [0,1]$:
     distributional overlap between generated and *real* feature distributions
     (higher = better; the real test set scores 1.0 by definition).

**Objective vs. subjective.** A low perplexity model can still produce dull, repetitive
melodies, and a high-OA model can be locally awkward — so we deliberately report likelihood
**and** distributional overlap **and** keep generated audio for a human listen. This mismatch
between the optimised objective (perplexity) and perceived musicality is exactly the tension the
rubric asks us to discuss.
""")

md(r"""
## 3.2  Evaluation — Discussion (baselines & beating the trivial method)

- **Trivial / oracle baselines.** The **real test set** is the oracle (OA = 1.0): no generator
  can beat it, and it bounds every distributional metric. The **Markov chain** is the trivial
  *generative* baseline — anything learned should at least be coherent relative to it.
- **What "better" means.** On **held-out likelihood**, the **CharLSTM clearly beats Markov**
  (NLL 0.996 vs 1.49; PPL 2.71 vs 4.44) — i.e. the LSTM's unbounded context genuinely predicts
  folk melodies better than a fixed-order count model. On **distributional realism (OA)**, the
  pretrained **MelodyT5** is closest to the oracle (0.302), with Markov second — a useful, honest
  finding that *the model with the best likelihood is not the most distribution-matching one*,
  which is precisely the objective-vs-musicality gap.
- All from-scratch models reach **100% (or near) valid MIDI**, so improvements are not an
  artifact of broken outputs.
""")

md(r"""
## 3.3  Evaluation — Code & results

The OA metric is a kernel-density overlap, computed per feature against the oracle and averaged:

```python
def overlapping_area(real_vals, gen_vals, n=1000):
    x = np.linspace(lo, hi, n); step = (hi-lo)/n
    return np.clip(np.sum(np.minimum(gaussian_kde(real)(x),
                                     gaussian_kde(gen)(x))) * step, 0, 1)
```
""")

code(r"""
# Full Task 1 results table (medium run). NLL/PPL are blank for models where we do not
# compute a clean per-token likelihood (GPT-2 sub-words; pretrained TunesFormer/MelodyT5).
cols = ["model", "grammar_validity", "scale_consistency_mean",
        "OA_overall", "test_nll", "test_ppl", "n_midi_valid"]
t1[cols].round(3)
""")

code(r"""
# Visual comparison of generated models against the real-test-set oracle across feature metrics.
display(Image("task1_comparison.png"))
""")

# ---- 4. RELATED WORK ----
md(r"""
## 4  Task 1 — Related Work

- **Boulanger-Lewandowski, Bengio & Vincent (2012)** established Nottingham as a symbolic-music
  benchmark, reporting log-likelihood for N-grams, RNNs, and RNN-RBM/NADE models on an
  88-dimensional **piano-roll** representation. Our likelihoods are *not directly comparable*
  (we model **ABC characters**, not piano-roll frames), so we use their numbers as **historical
  context** and as justification for reporting NLL/perplexity — not as a leaderboard.
- **Sturm et al. (2016), "FolkRNN"** showed that ABC folk tunes can be modelled as text with a
  char-level LSTM — direct precedent for our CharLSTM.
- **Geerlings & Meroño-Peñuela (2020)** fine-tuned GPT-2 on ABC notation, motivating our GPT-2
  baseline. **Wu et al. (2023, TunesFormer; 2024, MelodyT5)** are ABC-native Transformers with
  bar-patching/control codes; we run their published models as strong references.
- **Yang & Lerch (2018)** motivate objective symbolic-music metrics (pitch/rhythm distributions,
  overlap-style comparison), which our pitch/rhythm/OA metrics follow.

**How our results compare.** Consistent with FolkRNN, a learned recurrent model beats a fixed
N-gram on held-out likelihood; consistent with the metric literature, no generator matches the
oracle distribution, and likelihood ranking ≠ distribution-matching ranking.
""")

# ============================================================================= TASK 2
md(r"""
---
# TASK 2 — Symbolic, Conditioned Generation

**Goal.** Generate music *conditionally on an input*. Our pipeline:
`input melody → detect style (classifier) → generate a new continuation conditioned on
(detected style + seed bars)`. This satisfies the "conditioned generation given some input"
requirement: the input supplies **two** conditions — a **style label** and an **opening seed** —
and the model produces *new* notes rather than copying the input.
""")

# ---- 1. DATA ----
md(r"""
## 1.1  Data — Context

Same **Nottingham ABC** corpus as Task 1, but here we exploit a free supervision signal: in the
JukeDeck release the **file name is the style label** (`jigs.abc`, `reels.abc`,
`hornpipes.abc`, `waltzes.abc`). We restrict to the **four styles with enough data** — *jig*
(6/8), *reel* (4/4), *hornpipe* (4/4, dotted feel), *waltz* (3/4) — and skip rare classes
(slip jigs, morris) that are too small to train/evaluate on.
""")

md(r"""
## 1.2  Data — Discussion (pre-processing)

- **Tune parsing → `Tune` objects** with parsed header (`meter`, `unit`, `key`) and a list of
  **bars** (`split_bars` on `|`). Chords are stripped as in Task 1.
- **Conditional examples.** For each tune we form `header + seed_bars + <SEED_END> +
  continuation_bars`. The first few bars are the **seed**; the rest is the **target
  continuation** the generators must learn to produce.
- **Classifier features.** Each bar is turned into a **35-dim feature vector** —
  pitch-class distribution (12), key one-hot (12), meter one-hot (4), plus note-count,
  pitch-range, unique-notes, sinusoidal **bar-position**, rhythmic density, and interval
  variance — so the classifier sees melodic + metric + rhythmic cues, not raw text.
- **Class imbalance is real and reported:** jigs/reels are plentiful while hornpipes/waltzes are
  scarce (reflected in the per-style test counts below), which we account for when reading
  per-style accuracy.
""")

md(r"""
## 1.3  Data — Code & statistics

```python
# code/task2_conditional_generation/run_task2.py  -- per-bar feature vector (35-dim)
def extract_bar_features(bar, key, meter, bar_idx=0, total_bars=1):
    return np.concatenate([
        pc_dist,                       # 12  pitch-class distribution (duration-weighted)
        key_vec,                       # 12  key root one-hot
        meter_vec,                     #  4  meter one-hot (6/8, 4/4, 3/4, other)
        [note_count, pitch_range, unique_notes,
         sin_pos, cos_pos,             #     cyclic bar position within the tune
         rhythm_density, interval_var]
    ])
```
""")

code(r"""
# Per-style evaluation counts (medium run) -- note the jig/reel vs hornpipe/waltz imbalance.
t2 = pd.read_csv("t2_metrics.csv")
counts = t2[t2["model"] == "markov_baseline"][["style", "n_generated"]].set_index("style")
counts.columns = ["# evaluated tunes"]
counts.T
""")

# ---- 2. MODELING ----
md(r"""
## 2.1  Modeling — Context

**ML formulation (two stages).**

1. **Style detector** $g$: maps an input tune to one of 4 styles. Trained with cross-entropy on
   the bar-feature sequences; inputs = features, output = style label. We train **two**
   detectors (an **MLP** on song-level mean/std features and a **bidirectional GRU** on the bar
   sequence) and keep the better one on the held-out test set.
2. **Conditional generator** $p(\text{continuation}\mid \text{style}, \text{seed})$: an
   autoregressive model whose prompt is `header + style token + seed + <SEED_END>`; it then
   decodes only the **new** continuation tokens. Objective = NLL of the continuation; we report
   perplexity where it is well-defined.

We implement four generators of increasing structure: **per-style Markov**, **scratch
Transformer**, **GPT-2 fine-tuned**, and a **style-conditioned GRU-VAE**.
""")

md(r"""
## 2.2  Modeling — Discussion (advantages / disadvantages)

| Generator | Pros | Cons |
|---|---|---|
| **Per-style Markov** | Trivial, fast; competitive on style control; gives a real PPL | Local only; cannot plan phrase structure |
| **Scratch Transformer** | Causal self-attention; **lowest perplexity**; trained fully by us | Needs more data than we have for the rare styles |
| **GPT-2 fine-tuned** | Pretrained prior → **best style accuracy & composite**; very fluent | No clean PPL here; sub-word tokens not music-aware |
| **Style-conditioned GRU-VAE** | Latent `z` + style embedding → **best cross-style diversity**; controllable | Reconstruction blur; weaker exact likelihood |

Trade-off summary: GPT-2 wins on *obeying the condition*, the Transformer wins on *likelihood*,
and the VAE wins on *diversity* — there is no universal winner, which is the point.
""")

md(r"""
## 2.3  Modeling — Code (architectures & the conditioning mechanism)

The **conditioning** is explicit: the prompt carries a style token + the seed, and only tokens
*after* the seed are returned as the generated continuation:

```python
class TransformerGenerator:
    def generate_continuation(self, tune, style):
        ids = prompt_ids(tune, style, self.vocab, max_len)   # header + STYLE + seed + <SEED_END>
        generated = ids.clone()
        for _ in range(max_new_tokens):                      # causal top-k sampling
            logits = self.model(generated)[0, -1] / 0.95
            ...
        return self.vocab.decode_chars(generated[0, ids.shape[1]:])   # ONLY the new tokens
```

Scratch Transformer (decoder-style, causal mask) and the style-VAE (the style id is embedded and
concatenated to the decoder input at every step, alongside the latent `z`):

```python
class ScratchTransformer(nn.Module):           # token+pos embed -> causal TransformerEncoder -> head
    def forward(self, ids):
        x = self.tok(ids) + self.pos(pos)
        mask = torch.triu(torch.ones(L, L, dtype=bool), 1)     # causal
        return self.head(self.ln(self.blocks(x, mask=mask)))

class StyleVAE(nn.Module):                      # GRU encoder -> (mu, logvar) -> z; GRU decoder
    def forward(self, ids):
        mu, logvar = self.encode(ids); z = mu + randn_like(std)*std
        s = self.style_emb(ids[:,1])            # <-- style conditioning
        out,_ = self.dec(cat([emb, z_rep, s_rep], -1))
        return self.head(out), mu, logvar
```
""")

code(r"""
# Generator + classifier training loss curves (medium run).
display(Image("training_loss_curves.png"))
""")

# ---- 3. EVALUATION ----
md(r"""
## 3.1  Evaluation — Context

A *good* conditioned output must (a) **obey the condition** and (b) **be a valid, varied tune**.
We therefore evaluate condition-following directly:

- **Style accuracy** — fraction of generations the **classifier** labels as the *target* style
  (the classifier acts as an automatic judge of whether the condition was honoured).
- **Meter consistency / seed-key consistency / structural validity** — does the output keep the
  style's expected meter (jig 6/8, reel/hornpipe 4/4, waltz 3/4), preserve the seed's key, and
  contain bar/repeat structure?
- **Intra- & cross-style diversity** — normalised edit distance among outputs (anti-copying) and
  between styles (does the model actually differentiate styles?).
- **Pitch-range naturalness** and **valid-MIDI rate** as sanity floors.
- **Perplexity** where well-defined (Markov, scratch Transformer).

**Objective vs. subjective:** perplexity rewards predictable continuations, but a perfectly
predictable tune is boring — so we pair it with diversity and with style accuracy (a *semantic*
target), and keep narrated audio reels for human listening.
""")

md(r"""
## 3.2  Evaluation — Discussion (baselines & beating the trivial method)

- **Trivial baseline = per-style Markov.** Every learned generator must justify its cost against
  it. Result: GPT-2 (0.958) and the VAE (0.908) clearly beat Markov (0.850) on **style
  accuracy**; the scratch Transformer **beats Markov on perplexity** (5.44 vs 6.92). Markov
  remains a respectable control on style — an honest, non-trivial baseline.
- **The classifier as judge.** Because style accuracy is measured by our trained classifier, we
  also show its **confusion matrix** so graders can see the judge is itself reliable and where it
  confuses styles (mostly reel↔hornpipe, which share 4/4).
- **Across-the-board sanity:** meter consistency, seed-key consistency and structural validity
  are **1.0** for all models, and **51/51** samples produced valid MIDI — so differences in style
  accuracy are about *musical conditioning*, not broken output.
""")

md(r"""
## 3.3  Evaluation — Code & results
""")

code(r"""
# Model-level Task 2 summary (medium run): one row per generator.
s = pd.read_csv("t2_summary.csv")
show = ["model", "style_accuracy", "structural_validity", "pitch_range_naturalness",
        "intra_style_diversity", "cross_style_diversity", "perplexity_mean",
        "composite_no_perplexity"]
s[show].round(3)
""")

code(r"""
# Left: metric heatmap across models.  Right: confusion matrix of the style classifier (judge).
display(Image("metrics_comparison_heatmap.png"))
display(Image("classifier_confusion_matrix.png"))
""")

# ---- 4. RELATED WORK ----
md(r"""
## 4  Task 2 — Related Work

- **Huang et al. (2018/2019), "Music Transformer"** introduced self-attention for long-term
  musical structure and conditional continuation/accompaniment — the architectural motivation for
  our scratch Transformer.
- **Geerlings & Meroño-Peñuela (2020)** fine-tuned GPT-2 to produce *controlled, believable* ABC
  sequences — direct precedent for our GPT-2 conditional continuation, where the control is the
  detected style token.
- **Wu et al. (2024), "MelodyT5"** frame symbolic music as score-to-score tasks (including
  harmonisation-style conditioning) in ABC — related to our style-conditioned generation; data-prep
  hooks for it remain in the code, though the final run uses the four models above.
- Style/structure-following metrics follow the **Yang & Lerch (2018)** objective-evaluation
  tradition, extended here with a *classifier-as-judge* for the condition itself.

**How our results compare.** As in the GPT-2-for-ABC work, a pretrained language prior gives the
best condition-following; as in the broader literature, no single model dominates — likelihood,
condition-accuracy and diversity trade off against each other.
""")

# ============================================================================= CLOSING
md(r"""
---
# Summary, Honest Limitations & Generated Samples

**Headline findings.**

- *Task 1:* CharLSTM gives the best held-out likelihood (PPL 2.71) and beats the Markov baseline;
  MelodyT5 best matches the real distribution (OA 0.302). Likelihood ranking ≠ realism ranking.
- *Task 2:* GPT-2 fine-tuned best obeys the style condition (style acc 0.958, composite 0.879);
  the scratch Transformer has the lowest perplexity; the VAE is the most diverse. All four beat or
  match the Markov baseline on the dimension they target.

**Limitations we are explicit about.**

- Published Nottingham scores use different representations/splits, so we treat them as historical
  context, **not** a direct comparison (see `REPORT_REFERENCES_AND_METRICS.md`).
- We do not compute a clean per-token perplexity for GPT-2 (sub-word tokens) or the VAE.
- TunesFormer / MelodyT5 are **pretrained references**, not models we trained.
- Hornpipe/waltz test sets are small, so their per-style numbers are higher-variance.

**Generated music (submitted files).**

- `symbolic_unconditioned.mid` — best Task 1 sample.
- `symbolic_conditioned.mid` — best Task 2 conditioned continuation.
- Task 2 narrated audio reels (`results/task2/audio_reels/`) play the seed and each model's
  continuation back-to-back for the human-listening part of the presentation.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open("workbook.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"workbook.ipynb written with {len(cells)} cells.")
