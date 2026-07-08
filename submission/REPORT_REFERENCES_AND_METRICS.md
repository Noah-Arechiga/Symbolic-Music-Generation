# Report References, Baselines, and Metrics

This note is meant to support the assignment writeup. It explains which prior work we reference, which scores are fair to compare against, and what each metric in the output CSVs means.

## Important Comparison Caveat

The safest claim is:

> We compare our models primarily within our own Nottingham/ABC pipeline against a real-test-set oracle and Markov baselines. We cite prior papers for dataset history, model design, and metric choice. Direct score-to-score comparison with every paper is limited because prior papers often use different representations, tokenizations, splits, and evaluation units.

This matters especially for NLL/perplexity. A piano-roll NLL from an RNN-RBM paper is not directly equivalent to our ABC-character NLL. It is still useful as a historical baseline, but not as a strict leaderboard comparison.

## Dataset Reference

We use the Nottingham folk tune collection in ABC notation, using the cleaned JukeDeck GitHub version in code. The original Nottingham dataset is a common symbolic music benchmark and is used in Boulanger-Lewandowski et al. 2012.

The code strips or parses ABC notation depending on the task:

- Task 1: unconditional melody generation from ABC.
- Task 2: conditional continuation generation using seed bars plus automatically detected style.

## Publications We Reference

| Paper / Work | Why it matters for this project | How we use it |
|---|---|---|
| Boulanger-Lewandowski, Bengio, Vincent, 2012, *Modeling Temporal Dependencies in High-Dimensional Sequences* | Classic Nottingham benchmark using piano-roll symbolic music; reports log-likelihood and accuracy for models including N-grams, RNN, RTRBM, RNN-RBM, RNN-NADE. | Historical Task 1 baseline and justification for NLL/perplexity. |
| Yang & Lerch, 2018, *On the Evaluation of Generative Models in Music* / MGEval | Motivates objective symbolic-music metrics such as pitch/rhythm distributions and overlap-style comparison. | Our Task 1 pitch/rhythm/OA metrics are inspired by this family of metrics. |
| Sturm et al., 2016, *Music transcription modelling and composition using deep learning* / FolkRNN | LSTM folk generation in ABC notation; shows ABC folk music can be modeled as text-like sequences. | Baseline motivation for CharLSTM and ABC generation. |
| Huang et al., 2018/2019, *Music Transformer* | Transformer self-attention for long-term musical structure and conditional continuation/accompaniment. | Architecture motivation for Task 2 scratch Transformer. |
| Geerlings & Meroño-Peñuela, 2020, *Interacting with GPT-2 to Generate Controlled and Believable Musical Sequences in ABC Notation* | Direct precedent for fine-tuning GPT-2 on ABC notation and using language modeling for music continuation. | Motivation for GPT-2 fine-tuning in both tasks. |
| Wu et al., 2023, *TunesFormer* | ABC-native Transformer with bar patching and control codes; trained on a large Irish tune corpus. | Task 1 pretrained/transfer model and related work for controllable symbolic generation. |
| Wu et al., 2024, *MelodyT5* | Unified score-to-score Transformer in ABC notation with generation and harmonization-style tasks. | Task 1 pretrained model; related work for Task 2 score-to-score generation. |

## Publication Scores / Baselines To Mention

### Boulanger-Lewandowski et al. 2012

Their Table 1 reports Nottingham log-likelihood (LL) and accuracy for piano-roll models. Since they report LL and our tables often report NLL, use:

```text
NLL = -LL
```

Relevant Nottingham numbers from their table:

| Model | Nottingham LL | Equivalent NLL | Accuracy |
|---|---:|---:|---:|
| 1-Gram Add-p | -5.94 | 5.94 | 22.76% |
| N-Gram Gaussian | -3.16 | 3.16 | 65.97% |
| MLP | -4.38 | 4.38 | 63.46% |
| RNN | -4.46 | 4.46 | 62.93% |
| RNN-RBM | -2.39 | 2.39 | 75.40% |
| RNN-NADE HF | -2.31 | 2.31 | 71.50% |

Use these as historical context, not a direct apples-to-apples comparison, because their representation is 88-dimensional piano-roll with chord instantiations and ours is ABC text/converted MIDI.

### Our Task 1 Internal Baselines

From `task1_unconditional_generation/outputs/task1_medium_metrics_final_merged.csv`:

| Model | Test NLL | Test PPL | OA Overall | Valid MIDI |
|---|---:|---:|---:|---:|
| Markov k=3 alpha=0.1 | 1.4897 | 4.4360 | 0.2574 | 20/20 |
| Markov k=2 alpha=0.2 | 1.5898 | 4.9029 | 0.2712 | 20/20 |
| CharLSTM | 0.9960 | 2.7075 | 0.2098 | 20/20 |
| GPT2_FT | n/a | n/a | 0.1236 | 19/20 |
| TunesFormer | n/a | n/a | 0.1024 | 20/20 |
| MelodyT5 fixed rerun | n/a | n/a | 0.3021 | 19/20 |

Interpretation:

- For NLL/PPL, CharLSTM is strongest among models where we compute held-out likelihood.
- For distributional OA, MelodyT5 is best among generated models in the merged medium results.
- Real test set is the oracle with OA = 1.0.

### Our Task 2 Internal Baselines

From `task2_conditional_generation/outputs/medium/task2_medium_model_summary.csv`:

| Model | Valid MIDI | Mean Style Accuracy | Composite without PPL | Perplexity Mean |
|---|---:|---:|---:|---:|
| GPT-2 fine-tuned | 51/51 | 0.9583 | 0.8794 | n/a |
| VAE | 51/51 | 0.9083 | 0.8675 | n/a |
| Markov baseline | 51/51 | 0.8500 | 0.8617 | 6.9237 |
| Scratch Transformer | 51/51 | 0.8125 | 0.8512 | 5.4405 |

Interpretation:

- GPT-2 fine-tuned is the best conditional generator by style accuracy and composite score.
- VAE has the best cross-style diversity.
- Scratch Transformer beats Markov on perplexity, but Markov is competitive on style control.
- All evaluated Task 2 samples converted to valid MIDI.

## Task 1 Metrics

Let generated set be `G` and real test set be `R`.

### Negative Log-Likelihood

Measures how surprised the model is by held-out test tokens:

```text
NLL = -(1/N) * sum_i log p(x_i | x_<i)
```

Lower is better.

### Perplexity

Interpretable version of NLL:

```text
PPL = exp(NLL)
```

Lower is better.

### Grammar Validity

Fraction of generated ABC strings that parse successfully:

```text
grammar_validity = valid_ABC_count / generated_count
```

Higher is better.

### Pitch Count Mean

Average number of distinct MIDI pitches used per tune:

```text
pitch_count = |unique(pitches)|
```

Too low means repetitive/flat melodies; too high can mean noisy wandering.

### Pitch Class Histogram Entropy

Pitch class histogram maps notes to 12 pitch classes, ignoring octave. Entropy is:

```text
H(PCH) = - sum_c p_c log(p_c)
```

Higher means wider pitch-class usage, but extremely high can be less tonal.

### Pitch Range

Span of melody in semitones:

```text
pitch_range = max(pitches) - min(pitches)
```

Folk melodies should have a natural vocal/instrumental range.

### Average Pitch Interval

Average absolute melodic jump:

```text
avg_pitch_interval = mean(|pitch_{i+1} - pitch_i|)
```

Lower values usually mean stepwise motion; very high values can sound erratic.

### Note Count

Total notes per generated tune:

```text
note_count = number_of_notes
```

Used to catch outputs that are too short or too sparse.

### Average IOI

Mean inter-onset interval:

```text
avg_ioi = mean(onset_{i+1} - onset_i)
```

Captures rhythmic pacing.

### Scale Consistency

Fraction of notes belonging to the key implied by the ABC header:

```text
scale_consistency = notes_in_key / total_notes
```

Higher generally means the melody stays tonally coherent.

### Overlapping Area

Distributional similarity between generated and real feature distributions:

```text
OA = integral min(p_real(x), p_generated(x)) dx
```

Range is 0 to 1. Higher is better. Real test set gets OA = 1.0 by definition.

In our Task 1 CSV, `OA_overall` is the mean of:

- `OA_pitch_count`
- `OA_pch_entropy`
- `OA_pitch_range`
- `OA_scale_cons`

## Task 2 Metrics

Task 2 evaluates conditional generation. The requested condition is detected style plus seed melody.

### Style Accuracy

Fraction of generated tunes classified as the target style:

```text
style_accuracy = count(classifier(generated) == target_style) / n_generated
```

Higher means the model obeyed the style condition.

### Meter Consistency

Fraction of generated tunes using the expected meter for the style:

```text
meter_consistency = count(generated_meter == expected_meter(style)) / n_generated
```

Expected meters:

- jig: 6/8
- reel: 4/4
- hornpipe: 4/4
- waltz: 3/4

### Seed Key Consistency

Fraction of generated tunes that preserve the input seed key:

```text
seed_key_consistency = count(generated_key == seed_key) / n_generated
```

Higher means the continuation respects the seed.

### Structural Validity

Fraction of generated tunes with repeat/bar structure:

```text
structural_validity = count(has_repeat_or_bar_structure) / n_generated
```

Higher means the generated ABC looks more like a folk tune.

### Intra-Style Diversity

Average normalized Levenshtein distance among outputs for the same model/style:

```text
diversity = mean(levenshtein(a, b) / max(len(a), len(b)))
```

Higher means less repetition/copying among outputs.

### Pitch Range Naturalness

Fraction of generated tunes whose pitch range is within a natural folk range:

```text
pitch_range_naturalness = count(3 <= pitch_range <= 24) / n_generated
```

Higher means the melody is neither collapsed nor wildly spread out.

### Cross-Style Diversity

Average normalized edit distance between representative outputs across styles for a model:

```text
cross_style_diversity = mean(distance(style_i_output, style_j_output))
```

Higher means the model produces distinguishable styles.

### Perplexity

For models that implement it, Task 2 perplexity is computed on held-out conditional ABC continuation examples:

```text
PPL = exp(cross_entropy)
```

In the current implementation, perplexity is available for Markov and scratch Transformer, but not for GPT-2 and VAE rows.

## What To Claim In The Assignment

Recommended wording:

> For Task 1, we evaluate unconditional Nottingham ABC generation using language-model metrics (NLL, perplexity), grammar validity, pitch/rhythm statistics, scale consistency, and distributional overlap against the real test set. The Markov chain and CharLSTM serve as train-from-scratch baselines; TunesFormer and MelodyT5 provide pretrained ABC-music references. The real test set is the oracle distribution.

> For Task 2, we implement style-aware conditional generation: a classifier first detects the style of the seed melody, then generators continue the melody conditioned on the detected style and seed bars. We evaluate condition-following with style accuracy, meter consistency, seed-key consistency, structure validity, diversity, natural pitch range, and MIDI validity.

Also include this limitation:

> Published Nottingham scores are not directly comparable to our ABC-character results unless the representation, split, and likelihood unit are matched. Therefore, we use published results as historical baselines and metric motivation, while our quantitative claims are based on controlled comparisons inside our own pipeline.

## Links For References

- Boulanger-Lewandowski et al. 2012, ICML: https://arxiv.org/abs/1206.6392
- Yang & Lerch 2018, MGEval-style objective metrics: https://researchr.org/publication/yangevaluation2018
- Sturm et al. 2016, FolkRNN / ABC LSTM: https://arxiv.org/abs/1604.08723
- Huang et al. 2018/2019, Music Transformer: https://arxiv.org/abs/1809.04281
- Geerlings & Meroño-Peñuela 2020, GPT-2 for ABC: https://aclanthology.org/2020.nlp4musa-1.10/
- Wu et al. 2023, TunesFormer: https://arxiv.org/abs/2301.02884
- Wu et al. 2024, MelodyT5: https://arxiv.org/abs/2407.02277

