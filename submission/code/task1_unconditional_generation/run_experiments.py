#!/usr/bin/env python3
"""GPU-ready Nottingham Task 1 experiment runner.

Run examples:
  python run_experiments.py --mode smoke
  python run_experiments.py --mode medium
  python run_experiments.py --mode full
  python run_experiments.py --mode smoke --models markov,lstm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
import pretty_midi
import torch
import torch.nn as nn
import yaml
from huggingface_hub import hf_hub_download
from midiutil import MIDIFile
from scipy.stats import entropy as scipy_entropy
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    # Kaggle can assign GPUs whose cuDNN RNN kernels do not match the installed
    # PyTorch/CUDA build. The native PyTorch LSTM path is slower but portable.
    torch.backends.cudnn.enabled = False


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


DEFAULT_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "n_generate": 2,
        "max_test_tunes": 20,
        "run_pretrained": True,
        "quality_filter": {
            "enabled": False,
            "candidate_multiplier": 1,
            "min_note_count": 40,
            "min_pitch_range": 12,
            "min_unique_pitches": 6,
        },
        "markov": {
            "generate_max_len": 500,
            "temperature": 0.9,
        },
        "lstm": {
            "epochs": 1,
            "hidden": 128,
            "layers": 1,
            "batch_size": 64,
            "seq_len": 128,
            "train_stride": 128,
            "eval_stride": 16,
            "lr": 0.002,
            "generate_max_len": 600,
            "temperature": 0.9,
        },
        "gpt2": {
            "finetune_n": 5,
            "epochs": 2,
            "max_len": 128,
            "batch_size": 1,
            "grad_accum": 4,
            "lr": 5e-5,
            "generate": True,
            "generate_max_new_tokens": 300,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 50,
        },
        "tunesformer": {"n_generate": 2, "max_patch": 64, "top_p": 0.8, "top_k": 8, "temperature": 1.0},
        "melodyt5": {"n_generate": 2, "max_patch": 48, "top_p": 0.8, "top_k": 8, "temperature": 2.0},
    },
    "medium": {
        "n_generate": 20,
        "max_test_tunes": 100,
        "run_pretrained": True,
        "quality_filter": {
            "enabled": True,
            "candidate_multiplier": 3,
            "min_note_count": 80,
            "min_pitch_range": 18,
            "min_unique_pitches": 8,
        },
        "markov": {
            "generate_max_len": 750,
            "temperature": 0.9,
        },
        "lstm": {
            "epochs": 10,
            "hidden": 256,
            "layers": 2,
            "batch_size": 128,
            "seq_len": 128,
            "train_stride": 8,
            "eval_stride": 4,
            "lr": 0.002,
            "generate_max_len": 900,
            "temperature": 1.0,
        },
        "gpt2": {
            "finetune_n": 100,
            "epochs": 3,
            "max_len": 256,
            "batch_size": 2,
            "grad_accum": 4,
            "lr": 5e-5,
            "generate": True,
            "generate_max_new_tokens": 500,
            "temperature": 0.95,
            "top_p": 0.95,
            "top_k": 50,
        },
        "tunesformer": {"n_generate": 20, "max_patch": 192, "top_p": 0.8, "top_k": 8, "temperature": 1.1},
        "melodyt5": {"n_generate": 20, "max_patch": 192, "top_p": 0.8, "top_k": 8, "temperature": 1.5},
    },
    "full": {
        "n_generate": 50,
        "max_test_tunes": None,
        "run_pretrained": True,
        "quality_filter": {
            "enabled": True,
            "candidate_multiplier": 2,
            "min_note_count": 100,
            "min_pitch_range": 20,
            "min_unique_pitches": 8,
        },
        "markov": {
            "generate_max_len": 900,
            "temperature": 0.9,
        },
        "lstm": {
            "epochs": 30,
            "hidden": 512,
            "layers": 3,
            "batch_size": 128,
            "seq_len": 128,
            "train_stride": 1,
            "eval_stride": 1,
            "lr": 0.002,
            "generate_max_len": 1200,
            "temperature": 1.0,
        },
        "gpt2": {
            "finetune_n": "all",
            "epochs": 10,
            "max_len": 512,
            "batch_size": 4,
            "grad_accum": 2,
            "lr": 5e-5,
            "generate": True,
            "generate_max_new_tokens": 700,
            "temperature": 0.95,
            "top_p": 0.95,
            "top_k": 50,
        },
        "tunesformer": {"n_generate": 50, "max_patch": 192, "top_p": 0.8, "top_k": 8, "temperature": 1.1},
        "melodyt5": {"n_generate": 50, "max_patch": 192, "top_p": 0.8, "top_k": 8, "temperature": 1.5},
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_mode_config(mode: str) -> dict[str, Any]:
    cfg_path = ROOT / "config_modes.yaml"
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text())
        if mode in data.get("modes", {}):
            return data["modes"][mode]
    return DEFAULT_MODE_CONFIGS[mode]


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "generated": run_dir / "generated",
        "plots": run_dir / "plots",
        "models": run_dir / "models",
        "cache": run_dir / "cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parse_abc_file(filepath: Path) -> list[dict[str, str]]:
    content = filepath.read_text(encoding="utf-8", errors="replace")
    tune_blocks = re.split(r"(?=^X:\s*\d+)", content, flags=re.MULTILINE)
    tunes: list[dict[str, str]] = []
    for block in tune_blocks:
        block = block.strip()
        if not block or not block.startswith("X:"):
            continue
        x_m = re.search(r"^X:\s*(\d+)", block, re.MULTILINE)
        t_m = re.search(r"^T:\s*(.+)", block, re.MULTILINE)
        tunes.append(
            {
                "tune_id": x_m.group(1) if x_m else "UNK",
                "title": t_m.group(1).strip() if t_m else "Untitled",
                "raw_abc": block,
            }
        )
    return tunes


def strip_chords(abc_str: str) -> str:
    return re.sub(r'"[^"]*"', "", abc_str)


def normalize_headers(abc_str: str, keep: tuple[str, ...] = ("M", "L", "K", "X", "T")) -> str:
    lines = abc_str.split("\n")
    out, in_body = [], False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        hm = re.match(r"^([A-Za-z]):", s)
        if hm and not in_body:
            key = hm.group(1).upper()
            if key == "K":
                in_body = True
                out.append(line)
            elif key in keep:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out).strip()


def make_melody_only(abc_str: str) -> str:
    cleaned = strip_chords(abc_str)
    cleaned = re.sub(r"  +", " ", cleaned)
    return normalize_headers(cleaned)


def clean_generated_abc(abc_str: str, title: str = "Generated") -> str:
    abc = abc_str.replace("\r\n", "\n").replace("\r", "\n")
    abc = abc.split("<|endoftext|>", 1)[0].split("<EOS>", 1)[0]
    start = abc.find("X:")
    if start >= 0:
        abc = abc[start:]

    headers: dict[str, str] = {}
    body: list[str] = []
    in_body = False
    for raw in abc.splitlines():
        line = raw.split("%", 1)[0].strip()
        if not line:
            if in_body and body:
                break
            continue
        hm = re.match(r"^([A-Za-z]):\s*(.*)$", line)
        if hm and not in_body:
            key, val = hm.group(1).upper(), hm.group(2).strip()
            if key in {"X", "T", "M", "L", "K"}:
                headers[key] = val
                if key == "K":
                    in_body = True
            continue
        if hm and in_body:
            continue
        if not in_body:
            continue
        no_strings = re.sub(r'"[^"]*"', "", line)
        no_decor = re.sub(r"![^!]*!", "", no_strings)
        if re.search(r"[HIJKLMNOPQRSTUVWYhijklmnopqrstuvwy]", no_decor):
            continue
        if re.search(r"[A-Ga-gzZxX]", no_decor):
            body.append(line)

    out = [
        f"X:{headers.get('X', '1') or '1'}",
        f"T:{headers.get('T', title) or title}",
        f"M:{headers.get('M', '6/8') or '6/8'}",
        f"L:{headers.get('L', '1/8') or '1/8'}",
        f"K:{headers.get('K', 'G') or 'G'}",
    ]
    out.extend(body[:64] if body else ["GAB d2B|cBA G3|"])
    return "\n".join(out).strip() + "\n"


def make_full_leadsheet(abc_str: str) -> str:
    return normalize_headers(abc_str)


def ensure_nottingham_data(seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    DATA.mkdir(exist_ok=True)
    repo = DATA / "nottingham_raw"
    abc_cleaned_dir = repo / "ABC_cleaned"
    if not abc_cleaned_dir.exists():
        if repo.exists():
            shutil.rmtree(repo)
        log("Cloning JukeDeck Nottingham dataset.")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/jukedeck/nottingham-dataset.git",
                str(repo),
            ],
            check=True,
        )

    all_tunes: list[dict[str, str]] = []
    for abc_file in sorted(abc_cleaned_dir.glob("*.abc")):
        subset = abc_file.stem
        for tune in parse_abc_file(abc_file):
            tune["subset"] = subset
            tune["global_id"] = f"{subset}_{tune['tune_id']}"
            all_tunes.append(tune)

    random.seed(seed)
    by_subset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for tune in all_tunes:
        by_subset[tune["subset"]].append(tune)

    train_tunes, val_tunes, test_tunes = [], [], []
    for _, tunes in by_subset.items():
        random.shuffle(tunes)
        n = len(tunes)
        n_val = max(1, int(n * 0.10))
        n_test = max(1, int(n * 0.10))
        train_tunes.extend(tunes[: n - n_val - n_test])
        val_tunes.extend(tunes[n - n_val - n_test : n - n_test])
        test_tunes.extend(tunes[n - n_test :])

    melody_only_dir = DATA / "melody_only"
    full_leadsheet_dir = DATA / "full_leadsheet"
    melody_only_dir.mkdir(exist_ok=True)
    full_leadsheet_dir.mkdir(exist_ok=True)

    for split_name, split_data in [("train", train_tunes), ("val", val_tunes), ("test", test_tunes)]:
        melody_texts, leadsheet_texts = [], []
        for tune in split_data:
            melody = make_melody_only(tune["raw_abc"])
            lead = make_full_leadsheet(tune["raw_abc"])
            melody_texts.append(melody)
            leadsheet_texts.append(lead)
            gid = tune["global_id"]
            (melody_only_dir / f"{gid}.abc").write_text(melody, encoding="utf-8")
            (full_leadsheet_dir / f"{gid}.abc").write_text(lead, encoding="utf-8")

        (DATA / f"{split_name}.txt").write_text("\n".join(t["global_id"] for t in split_data), encoding="utf-8")
        (DATA / f"{split_name}_melody.txt").write_text("\n<|endoftext|>\n".join(melody_texts), encoding="utf-8")
        (DATA / f"{split_name}_leadsheet.txt").write_text("\n<|endoftext|>\n".join(leadsheet_texts), encoding="utf-8")

    log(f"Dataset ready: train={len(train_tunes)} val={len(val_tunes)} test={len(test_tunes)}.")
    return all_tunes, train_tunes, val_tunes, test_tunes


def abc_duration_multiplier(token: str) -> float:
    if not token:
        return 1.0
    if set(token) == {"/"}:
        return 0.5 ** len(token)
    if token.startswith("/"):
        denom = token[1:] or "2"
        try:
            return 1.0 / float(denom)
        except ValueError:
            return 0.5
    if "/" in token:
        left, right = token.split("/", 1)
        num = float(left) if left else 1.0
        den = float(right) if right else 2.0
        return num / den if den else num
    try:
        return float(token)
    except ValueError:
        return 1.0


def abc_key_accidentals(key: str) -> dict[str, int]:
    root = re.sub(r"(maj|min|m|dor|mix|aeo|ion|phr|lyd|loc)$", "", key.strip(), flags=re.IGNORECASE)
    root = root or "C"
    sharp_order = ["F", "C", "G", "D", "A", "E", "B"]
    flat_order = ["B", "E", "A", "D", "G", "C", "F"]
    n_map = {
        "C": 0,
        "G": 1,
        "D": 2,
        "A": 3,
        "E": 4,
        "B": 5,
        "F#": 6,
        "C#": 7,
        "F": -1,
        "Bb": -2,
        "Eb": -3,
        "Ab": -4,
        "Db": -5,
        "Gb": -6,
        "Cb": -7,
    }
    n = n_map.get(root, 0)
    if n > 0:
        return {note: 1 for note in sharp_order[:n]}
    if n < 0:
        return {note: -1 for note in flat_order[: -n]}
    return {}


def abc_body_for_fallback(abc_str: str) -> tuple[str, str, float]:
    key = "C"
    base_beats = 0.5
    body_lines: list[str] = []
    in_body = False
    for raw in abc_str.splitlines():
        line = raw.split("%", 1)[0].strip()
        if not line:
            continue
        if re.match(r"^L:\s*", line, flags=re.IGNORECASE):
            m = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if m:
                base_beats = 4.0 * int(m.group(1)) / int(m.group(2))
        if re.match(r"^K:\s*", line, flags=re.IGNORECASE):
            key = line.split(":", 1)[1].strip().split()[0] or "C"
            in_body = True
            continue
        if in_body:
            if re.match(r"^[A-Za-z]:", line):
                continue
            body_lines.append(line)
    return "\n".join(body_lines), key, base_beats


def abc_to_midi_fallback(abc_str: str, midi_path: Path, min_notes: int = 8) -> bool:
    body, key, base_beats = abc_body_for_fallback(abc_str)
    if not body:
        return False
    body = re.sub(r'"[^"]*"', "", body)
    body = re.sub(r"![^!]*!", "", body)
    body = re.sub(r"\[[^\]]*?([A-Ga-g])[^]]*?\]", r"\1", body)
    key_acc = abc_key_accidentals(key)
    base_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    token_re = re.compile(r"(?P<acc>[\^_=]*)(?P<note>[A-Ga-gzZxX])(?P<oct>[',]*)(?P<dur>\d*/*\d*)")
    mf = MIDIFile(1)
    mf.addTempo(0, 0, 120)
    time_pos = 0.0
    note_count = 0
    for m in token_re.finditer(body):
        note = m.group("note")
        dur = max(0.125, min(8.0, base_beats * abc_duration_multiplier(m.group("dur"))))
        if note in "zZxX":
            time_pos += dur
            continue
        letter = note.upper()
        octave = 5 if note.islower() else 4
        octave += m.group("oct").count("'")
        octave -= m.group("oct").count(",")
        acc_token = m.group("acc")
        if "^" in acc_token:
            accidental = acc_token.count("^")
        elif "_" in acc_token:
            accidental = -acc_token.count("_")
        elif "=" in acc_token:
            accidental = 0
        else:
            accidental = key_acc.get(letter, 0)
        pitch = 12 * (octave + 1) + base_pc[letter] + accidental
        pitch = int(max(24, min(96, pitch)))
        mf.addNote(0, 0, pitch, time_pos, dur, 90)
        time_pos += dur
        note_count += 1
    if note_count < min_notes:
        return False
    with midi_path.open("wb") as fh:
        mf.writeFile(fh)
    return midi_path.exists() and midi_path.stat().st_size > 100


def abc_to_midi(abc_str: str, abc_path: Path, midi_path: Path) -> str | None:
    abc_path.write_text(abc_str, encoding="utf-8")
    try:
        subprocess.run(["abc2midi", str(abc_path), "-o", str(midi_path)], capture_output=True, text=True, timeout=30)
        if midi_path.exists() and midi_path.stat().st_size > 100:
            return "abc2midi"
    except Exception:
        pass
    try:
        music21.converter.parse(abc_str, format="abc").write("midi", fp=str(midi_path))
        if midi_path.exists() and midi_path.stat().st_size > 100:
            return "music21"
    except Exception:
        pass
    try:
        if abc_to_midi_fallback(abc_str, midi_path):
            return "fallback_note_stream"
    except Exception:
        pass
    return None


def prepare_real_test_midi(test_tunes: list[dict[str, str]], max_test_tunes: int | None = None) -> tuple[Path, dict[str, str]]:
    midi_dir = DATA / "midi" / "test_real"
    midi_dir.mkdir(parents=True, exist_ok=True)
    for old in midi_dir.glob("*.mid"):
        old.unlink()

    key_re = re.compile(r"^K:\s*(\S+)", re.MULTILINE)
    test_key_map = {}
    selected = test_tunes[:max_test_tunes] if max_test_tunes else test_tunes
    ok = 0
    for tune in tqdm(selected, desc="Real test MIDI"):
        gid = tune["global_id"]
        m = key_re.search(tune["raw_abc"])
        test_key_map[gid] = m.group(1).split("%")[0].strip() if m else "C"
        abc_path = DATA / "melody_only" / f"{gid}.abc"
        mid_path = midi_dir / f"{gid}.mid"
        if abc_to_midi(tune["raw_abc"], abc_path, mid_path):
            ok += 1
    log(f"Real MIDI converted: {ok}/{len(selected)}.")
    return midi_dir, test_key_map


def compute_pitch_metrics(midi_path: Path) -> dict[str, Any] | None:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return None
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if len(notes) < 4:
        return None
    pitches = [n.pitch for n in notes]
    pcs = [p % 12 for p in pitches]
    pch = np.zeros(12)
    for pc in pcs:
        pch[pc] += 1
    pch_n = pch / (pch.sum() + 1e-8)
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    return {
        "pitch_count": len(set(pitches)),
        "pch": pch_n.tolist(),
        "pch_entropy": float(scipy_entropy(pch_n + 1e-8)),
        "pitch_range": max(pitches) - min(pitches),
        "avg_pitch_interval": float(np.mean(intervals)) if intervals else 0.0,
    }


def compute_rhythm_metrics(midi_path: Path) -> dict[str, Any] | None:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return None
    notes = sorted([n for inst in pm.instruments if not inst.is_drum for n in inst.notes], key=lambda n: n.start)
    if len(notes) < 4:
        return None
    onsets = [n.start for n in notes]
    durations = [n.end - n.start for n in notes]
    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    _, tempi = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) > 0 else 120.0
    beat_dur = 60.0 / tempo if tempo > 0 else 0.5
    beat_classes = [1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8, 1.0, 5 / 4, 3 / 2, 7 / 4, 2.0]
    nlh = np.zeros(12)
    for dur in durations:
        beats = dur / beat_dur
        idx = int(np.argmin([abs(beats - bc) for bc in beat_classes]))
        nlh[idx] += 1
    return {
        "note_count": len(notes),
        "avg_ioi": float(np.mean(iois)) if iois else 0.0,
        "avg_duration": float(np.mean(durations)),
        "nlh": (nlh / (nlh.sum() + 1e-8)).tolist(),
    }


KEY_CHROMATIC = {"C": 0, "G": 7, "D": 2, "A": 9, "E": 4, "B": 11, "F#": 6, "Gb": 6, "Db": 1, "Ab": 8, "Eb": 3, "Bb": 10, "F": 5}
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


def compute_scale_consistency(midi_path: Path, key_str: str = "C") -> float | None:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    except Exception:
        return None
    if not notes:
        return None
    pcs = [n.pitch % 12 for n in notes]
    ks = key_str.strip()
    is_minor = "min" in ks.lower() or (len(ks) > 1 and ks.endswith("m") and ks[-2].isalpha())
    root_str = re.sub(r"(maj|min|m)$", "", ks, flags=re.IGNORECASE).strip()
    root = KEY_CHROMATIC.get(root_str, 0)
    scale = MINOR_SCALE if is_minor else MAJOR_SCALE
    return sum(1 for pc in pcs if (pc - root) % 12 in scale) / len(pcs)


def check_abc_grammar(abc_str: str) -> bool:
    try:
        music21.converter.parse(abc_str, format="abc")
        return True
    except Exception:
        return False


def overlapping_area(real_vals: list[float], gen_vals: list[float], n: int = 1000) -> float:
    r = np.array(real_vals, dtype=float)
    g = np.array(gen_vals, dtype=float)
    r, g = r[np.isfinite(r)], g[np.isfinite(g)]
    if len(r) < 2 or len(g) < 2:
        return 0.0
    lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
    if lo == hi:
        return 1.0 if abs(r.mean() - g.mean()) < 1e-6 else 0.0
    x = np.linspace(lo, hi, n)
    step = (hi - lo) / n
    try:
        return float(np.clip(np.sum(np.minimum(gaussian_kde(r)(x), gaussian_kde(g)(x))) * step, 0, 1))
    except Exception:
        return 0.0


def compute_oracle(midi_dir: Path, test_key_map: dict[str, str], run_paths: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[float], np.ndarray]:
    real_pm_list, real_rm_list, real_sc_list = [], [], []
    for midi_file in tqdm(sorted(midi_dir.glob("*.mid")), desc="Oracle metrics"):
        gid = midi_file.stem
        key = test_key_map.get(gid, "C")
        pm = compute_pitch_metrics(midi_file)
        rm = compute_rhythm_metrics(midi_file)
        sc = compute_scale_consistency(midi_file, key)
        if pm:
            real_pm_list.append(pm)
        if rm:
            real_rm_list.append(rm)
        if sc is not None:
            real_sc_list.append(sc)
    real_pch_avg = np.mean([m["pch"] for m in real_pm_list], axis=0)
    np.save(run_paths["cache"] / "real_pch_avg.npy", real_pch_avg)
    oracle = {
        "grammar_validity": 1.0,
        "pitch_count_mean": np.mean([m["pitch_count"] for m in real_pm_list]),
        "pch_entropy_mean": np.mean([m["pch_entropy"] for m in real_pm_list]),
        "pitch_range_mean": np.mean([m["pitch_range"] for m in real_pm_list]),
        "avg_pitch_interval_mean": np.mean([m["avg_pitch_interval"] for m in real_pm_list]),
        "note_count_mean": np.mean([m["note_count"] for m in real_rm_list]),
        "avg_ioi_mean": np.mean([m["avg_ioi"] for m in real_rm_list]),
        "scale_consistency_mean": np.mean(real_sc_list),
        "OA_overall": 1.0,
    }
    return oracle, real_pm_list, real_rm_list, real_sc_list, real_pch_avg


def mean_metric(items: list[dict[str, Any]], key: str, default: float = float("nan")) -> float:
    vals = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
    return float(np.mean(vals)) if vals else default


def real_quality_targets(
    real_pm_list: list[dict[str, Any]],
    real_rm_list: list[dict[str, Any]],
    real_sc_list: list[float],
) -> dict[str, float]:
    sc_vals = [float(v) for v in real_sc_list if np.isfinite(float(v))]
    return {
        "pitch_count": mean_metric(real_pm_list, "pitch_count", 20.0),
        "pch_entropy": mean_metric(real_pm_list, "pch_entropy", 1.8),
        "pitch_range": mean_metric(real_pm_list, "pitch_range", 32.0),
        "avg_pitch_interval": mean_metric(real_pm_list, "avg_pitch_interval", 6.0),
        "note_count": mean_metric(real_rm_list, "note_count", 300.0),
        "avg_ioi": mean_metric(real_rm_list, "avg_ioi", 0.2),
        "scale_consistency": float(np.mean(sc_vals)) if sc_vals else 0.95,
    }


def relative_distance(value: float | None, target: float, floor: float = 1.0) -> float:
    if value is None or not np.isfinite(float(value)) or not np.isfinite(float(target)):
        return 10.0
    return abs(float(value) - float(target)) / max(abs(float(target)), floor)


def candidate_quality_score(
    pm: dict[str, Any] | None,
    rm: dict[str, Any] | None,
    sc: float | None,
    targets: dict[str, float],
) -> float:
    if not pm or not rm:
        return float("-inf")
    score = 0.0
    score -= 1.00 * relative_distance(pm.get("pitch_count"), targets["pitch_count"])
    score -= 1.00 * relative_distance(pm.get("pitch_range"), targets["pitch_range"])
    score -= 0.80 * relative_distance(rm.get("note_count"), targets["note_count"])
    score -= 0.55 * relative_distance(pm.get("pch_entropy"), targets["pch_entropy"], floor=0.5)
    score -= 0.35 * relative_distance(pm.get("avg_pitch_interval"), targets["avg_pitch_interval"], floor=0.5)
    score -= 0.25 * relative_distance(rm.get("avg_ioi"), targets["avg_ioi"], floor=0.05)
    if sc is not None and np.isfinite(sc):
        score -= 0.75 * abs(float(sc) - targets["scale_consistency"])
    else:
        score -= 0.75
    return float(score)


def quality_passes(pm: dict[str, Any] | None, rm: dict[str, Any] | None, cfg: dict[str, Any]) -> bool:
    if not pm or not rm:
        return False
    return (
        rm.get("note_count", 0) >= int(cfg.get("min_note_count", 80))
        and pm.get("pitch_range", 0) >= int(cfg.get("min_pitch_range", 18))
        and pm.get("pitch_count", 0) >= int(cfg.get("min_unique_pitches", 8))
    )


def select_quality_candidates(
    abc_list: list[str],
    model_name: str,
    run_paths: dict[str, Path],
    real_pm_list: list[dict[str, Any]],
    real_rm_list: list[dict[str, Any]],
    real_sc_list: list[float],
    cfg: dict[str, Any],
    target_n: int,
) -> tuple[list[str], dict[str, Any]]:
    info: dict[str, Any] = {
        "quality_filter_enabled": bool(cfg.get("enabled", False)),
        "n_candidates": len(abc_list),
        "n_selected": min(len(abc_list), target_n),
        "n_quality_passed": float("nan"),
        "best_quality_score": float("nan"),
    }
    if not cfg.get("enabled", False) or len(abc_list) <= target_n:
        return abc_list[:target_n], info

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    work_dir = run_paths["cache"] / "quality_filter" / safe_name
    reset_dir(work_dir)
    targets = real_quality_targets(real_pm_list, real_rm_list, real_sc_list)
    key_re = re.compile(r"^K:\s*(\S+)", re.MULTILINE)
    records: list[dict[str, Any]] = []

    for i, abc in enumerate(tqdm(abc_list, desc=f"Quality filter {model_name}")):
        abc_path = work_dir / f"candidate_{i:03d}.abc"
        midi_path = work_dir / f"candidate_{i:03d}.mid"
        method = abc_to_midi(abc, abc_path, midi_path)
        pm = compute_pitch_metrics(midi_path) if method else None
        rm = compute_rhythm_metrics(midi_path) if method else None
        key_match = key_re.search(abc)
        key = key_match.group(1).split("%")[0].strip() if key_match else "G"
        sc = compute_scale_consistency(midi_path, key) if method else None
        score = candidate_quality_score(pm, rm, sc, targets)
        passed = quality_passes(pm, rm, cfg)
        records.append(
            {
                "index": i,
                "converted": bool(method),
                "method": method or "",
                "passed": passed,
                "quality_score": score,
                "pitch_count": pm.get("pitch_count") if pm else float("nan"),
                "pitch_range": pm.get("pitch_range") if pm else float("nan"),
                "note_count": rm.get("note_count") if rm else float("nan"),
                "scale_consistency": sc if sc is not None else float("nan"),
            }
        )

    passed_records = [r for r in records if r["passed"] and np.isfinite(r["quality_score"])]
    ordered = sorted(passed_records, key=lambda r: r["quality_score"], reverse=True)
    if len(ordered) < target_n:
        used = {r["index"] for r in ordered}
        fill = [
            r
            for r in sorted(records, key=lambda r: r["quality_score"], reverse=True)
            if r["index"] not in used and np.isfinite(r["quality_score"])
        ]
        ordered.extend(fill)

    chosen = ordered[:target_n]
    selected = [abc_list[int(r["index"])] for r in chosen] if chosen else abc_list[:target_n]
    info.update(
        {
            "n_selected": len(selected),
            "n_quality_passed": len(passed_records),
            "best_quality_score": float(chosen[0]["quality_score"]) if chosen else float("nan"),
        }
    )
    summary = {"model": model_name, "targets": targets, "config": cfg, "info": info, "records": records}
    (run_paths["models"] / f"{safe_name}_quality_filter.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    selected_dir = run_paths["generated"] / f"{model_name}_selected"
    reset_dir(selected_dir)
    for j, abc in enumerate(selected):
        (selected_dir / f"tune_{j:02d}.abc").write_text(abc, encoding="utf-8")
    return selected, info


class MarkovChainABC:
    def __init__(self, order: int = 2, alpha: float = 0.2):
        self.order = order
        self.alpha = alpha
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab_size = 0

    def train(self, text: str) -> None:
        self.vocab_size = len(set(text))
        for i in range(len(text) - self.order):
            self.transitions[text[i : i + self.order]][text[i + self.order]] += 1

    def _log_prob(self, ctx: str, ch: str) -> float:
        c = self.transitions.get(ctx, Counter())
        return math.log((c.get(ch, 0) + self.alpha) / (sum(c.values()) + self.alpha * self.vocab_size + 1e-12))

    def compute_nll(self, text: str) -> float:
        total, n = 0.0, 0
        for i in range(self.order, len(text)):
            total -= self._log_prob(text[i - self.order : i], text[i])
            n += 1
        return total / n if n else float("inf")

    def generate(self, seed: str = "X:1\nT:Gen\nM:6/8\nL:1/8\nK:G\n", max_len: int = 500, temp: float = 0.9) -> str:
        result = list(seed)
        for _ in range(max_len - len(seed)):
            ctx = "".join(result[-self.order :])
            cnts = self.transitions.get(ctx, Counter())
            if not cnts:
                break
            chars = list(cnts.keys())
            probs = np.array([cnts[c] for c in chars], dtype=float) ** (1 / temp)
            probs /= probs.sum()
            result.append(np.random.choice(chars, p=probs))
            if "".join(result[-5:]) == "\n\n":
                break
        return "".join(result)


def run_markov(train_text: str, val_text: str, test_text: str, run_paths: dict[str, Path], cfg: dict[str, Any], n_generate: int) -> tuple[list[tuple[str, list[str]]], dict[str, float]]:
    model_rows: list[tuple[str, list[str]]] = []
    nll_map: dict[str, float] = {}
    results = {}
    for order in [1, 2, 3]:
        for alpha in [0.1, 0.2, 0.5]:
            model = MarkovChainABC(order, alpha)
            model.train(train_text)
            key = f"Markov_k{order}_a{alpha}"
            results[key] = {"model": model, "val": model.compute_nll(val_text), "test": model.compute_nll(test_text)}
            log(f"{key}: val_nll={results[key]['val']:.4f} test_nll={results[key]['test']:.4f}")

    best_key = min(results, key=lambda k: results[k]["val"])
    selected = [(f"Markov_{best_key}", best_key), ("Markov_k2_a0.2", "Markov_k2_a0.2")]
    for out_name, key in selected:
        model = results[key]["model"]
        out_dir = run_paths["generated"] / out_name
        reset_dir(out_dir)
        abc_list = []
        for i in range(n_generate):
            raw_abc = model.generate(max_len=int(cfg.get("generate_max_len", 500)), temp=float(cfg.get("temperature", 0.9)))
            abc = clean_generated_abc(raw_abc, title=f"{out_name}_{i:02d}")
            abc_list.append(abc)
            (out_dir / f"tune_{i:02d}.abc").write_text(abc, encoding="utf-8")
        with open(run_paths["models"] / f"{out_name}.pkl", "wb") as fh:
            pickle.dump(model, fh)
        model_rows.append((out_name, abc_list))
        nll_map[out_name] = results[key]["test"]
    return model_rows, nll_map


class ABCDataset(Dataset):
    def __init__(self, text: str, char2idx: dict[str, int], seq_len: int = 128, stride: int = 1):
        self.seq_len = seq_len
        self.data = torch.tensor([char2idx.get(c, char2idx.get("<UNK>", 0)) for c in text], dtype=torch.long)
        self.starts = list(range(0, max(0, len(self.data) - self.seq_len - 1), stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        return self.data[start : start + self.seq_len], self.data[start + 1 : start + self.seq_len + 1]


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed: int = 128, hidden: int = 256, layers: int = 2, drop: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.emb = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, h: tuple[torch.Tensor, torch.Tensor] | None = None):
        out, h = self.lstm(self.drop(self.emb(x)), h)
        return self.fc(self.drop(out)), h

    def init_h(self, bs: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        z = lambda: torch.zeros(self.layers, bs, self.hidden, device=dev)
        return z(), z()

    def generate(self, seed: str, c2i: dict[str, int], i2c: dict[int, str], max_len: int, temp: float, dev: torch.device) -> str:
        self.eval()
        res = list(seed)
        x = torch.tensor([c2i.get(c, 0) for c in seed], dtype=torch.long, device=dev).unsqueeze(0)
        h = self.init_h(1, dev)
        with torch.no_grad():
            for i in range(x.shape[1] - 1):
                _, h = self(x[:, i : i + 1], h)
            last = x[:, -1:]
            for _ in range(max_len):
                logits, h = self(last, h)
                probs = torch.softmax(logits.squeeze() / temp, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                res.append(i2c[nxt])
                last = torch.tensor([[nxt]], dtype=torch.long, device=dev)
                if "".join(res[-5:]) == "\n\n":
                    break
        return "".join(res)


def run_lstm(train_text: str, val_text: str, test_text: str, cfg: dict[str, Any], run_paths: dict[str, Path], n_generate: int, device: torch.device) -> tuple[list[str], float, list[float], list[float]]:
    char_vocab = ["<PAD>", "<UNK>", "<EOS>"] + sorted(set(train_text))
    char2idx = {c: i for i, c in enumerate(char_vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(char_vocab)

    train_ds = ABCDataset(train_text, char2idx, cfg["seq_len"], cfg["train_stride"])
    val_ds = ABCDataset(val_text, char2idx, cfg["seq_len"], cfg["eval_stride"])
    test_ds = ABCDataset(test_text, char2idx, cfg["seq_len"], cfg["eval_stride"])
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, drop_last=True, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, cfg["batch_size"], shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
    test_dl = DataLoader(test_ds, cfg["batch_size"], shuffle=False, drop_last=False, pin_memory=device.type == "cuda")

    model = CharLSTM(vocab_size, hidden=cfg["hidden"], layers=cfg["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg["epochs"] // 3), gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])
    train_losses, val_losses = [], []
    log(f"CharLSTM params={sum(p.numel() for p in model.parameters()):,} train_windows={len(train_ds)}.")

    for epoch in range(cfg["epochs"]):
        model.train()
        total, batches = 0.0, 0
        for bx, by in tqdm(train_dl, desc=f"LSTM epoch {epoch + 1}/{cfg['epochs']}"):
            bx, by = bx.to(device), by.to(device)
            h = tuple(t.detach() for t in model.init_h(bx.shape[0], device))
            opt.zero_grad(set_to_none=True)
            logits, _ = model(bx, h)
            loss = criterion(logits.reshape(-1, vocab_size), by.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            batches += 1
        train_loss = total / max(batches, 1)
        val_loss = evaluate_lstm_nll(model, val_dl, criterion, vocab_size, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        sched.step()
        log(f"LSTM epoch {epoch + 1}: train_nll={train_loss:.4f} val_nll={val_loss:.4f}.")

    test_nll = evaluate_lstm_nll(model, test_dl, criterion, vocab_size, device)
    torch.save(
        {
            "state": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "vocab_size": vocab_size,
            "cfg": cfg,
        },
        run_paths["models"] / "lstm_model.pt",
    )

    seeds = [
        "X:1\nT:Gen Jig\nM:6/8\nL:1/8\nK:Gmaj\n",
        "X:1\nT:Gen Reel\nM:4/4\nL:1/8\nK:Dmaj\n",
        "X:1\nT:Gen Hornpipe\nM:4/4\nL:1/8\nK:Amaj\n",
        "X:1\nT:Gen Waltz\nM:3/4\nL:1/8\nK:Cmaj\n",
    ]
    out_dir = run_paths["generated"] / "CharLSTM"
    reset_dir(out_dir)
    abc_list = []
    for i in range(n_generate):
        raw_abc = model.generate(
            seeds[i % len(seeds)],
            char2idx,
            idx2char,
            max_len=int(cfg.get("generate_max_len", 600)),
            temp=float(cfg.get("temperature", 0.9)),
            dev=device,
        )
        abc = clean_generated_abc(raw_abc, title=f"CharLSTM_{i:02d}")
        abc_list.append(abc)
        (out_dir / f"tune_{i:02d}.abc").write_text(abc, encoding="utf-8")
    return abc_list, test_nll, train_losses, val_losses


def evaluate_lstm_nll(model: CharLSTM, dl: DataLoader, criterion: nn.Module, vocab_size: int, device: torch.device) -> float:
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx)
            total += criterion(logits.reshape(-1, vocab_size), by.reshape(-1)).item()
            batches += 1
    return total / max(batches, 1)


def run_gpt2(train_tunes: list[dict[str, str]], cfg: dict[str, Any], run_paths: dict[str, Path], n_generate: int, device: torch.device) -> list[str]:
    model_dir = run_paths["models"] / "gpt2_finetuned"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    out_dir = run_paths["generated"] / "GPT2_FT"
    reset_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

    if cfg["finetune_n"] == "all":
        ft_tunes = train_tunes
    else:
        ft_tunes = train_tunes[: int(cfg["finetune_n"])]
    ft_text = "\n".join(make_melody_only(t["raw_abc"]) + tokenizer.eos_token for t in ft_tunes)
    log(f"GPT-2 fine-tune corpus: {len(ft_tunes)} tunes, {len(ft_text):,} chars.")

    tokens = tokenizer(ft_text, return_tensors="pt", truncation=False)["input_ids"][0]
    max_len = cfg["max_len"]
    chunks = []
    step = max(1, max_len // 2)
    for start in range(0, max(1, len(tokens) - 1), step):
        piece = tokens[start : start + max_len]
        if len(piece) < min(32, max_len):
            continue
        chunks.append({"input_ids": piece.tolist(), "attention_mask": [1] * len(piece)})
        if start + max_len >= len(tokens):
            break
    if not chunks:
        raise RuntimeError("GPT-2 tokenization produced no chunks.")
    chunk_tensors = [torch.tensor(c["input_ids"], dtype=torch.long) for c in chunks]

    def collate_gpt2(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        max_batch_len = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), max_batch_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_batch_len), dtype=torch.long)
        labels = torch.full((len(batch), max_batch_len), -100, dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
            labels[i, : len(ids)] = ids
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dl = DataLoader(chunk_tensors, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_gpt2)
    gpt2.to(device)
    gpt2.train()
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=cfg["lr"], weight_decay=0.01)
    grad_accum = max(1, int(cfg["grad_accum"]))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        total_loss, batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(dl, desc=f"GPT-2 epoch {epoch}/{cfg['epochs']}")
        for step_i, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = gpt2(**batch).loss
                scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            if step_i % grad_accum == 0 or step_i == len(dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix(nll=f"{total_loss / max(batches, 1):.3f}")
        log(f"GPT-2 epoch {epoch}: train_nll={total_loss / max(batches, 1):.4f}.")

    gpt2.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    gpt2.eval()
    abc_list = []
    for i in tqdm(range(n_generate), desc="GPT-2 generation"):
        prompt = "X:1\nT:Gen\nM:6/8\nL:1/8\nK:G\n"
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = gpt2.generate(
                **encoded,
                max_new_tokens=int(cfg.get("generate_max_new_tokens", 300)),
                temperature=float(cfg.get("temperature", 0.9)),
                top_p=float(cfg.get("top_p", 0.95)),
                top_k=int(cfg.get("top_k", 50)),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        abc = clean_generated_abc(tokenizer.decode(out_ids[0], skip_special_tokens=True), title=f"GPT2_FT_{i:02d}")
        abc_list.append(abc)
        (out_dir / f"tune_{i:02d}.abc").write_text(abc, encoding="utf-8")
    return abc_list


def split_abc_tunes(text: str) -> list[str]:
    return [
        block.strip()
        for block in re.split(r"(?=^X:\s*\d+)", text, flags=re.MULTILINE)
        if block.strip().startswith("X:")
    ]


def patch_tunesformer_repo(tf_dir: Path) -> None:
    gen = tf_dir / "generate.py"
    txt = gen.read_text(encoding="utf-8")
    txt = re.sub(
        r"torch\.load\(['\"]weights\.pth['\"](?:,\s*map_location=device)?(?:,\s*weights_only=False)?\)",
        "torch.load('weights.pth', map_location=device, weights_only=False)",
        txt,
    )
    txt = txt.replace("model.load_state_dict(checkpoint['model'])", "model.load_state_dict(checkpoint['model'], strict=False)")
    gen.write_text(txt, encoding="utf-8")


def run_tunesformer(cfg: dict[str, Any], run_paths: dict[str, Path]) -> list[str]:
    tf_dir = ROOT / "tunesformer_repo"
    if not tf_dir.exists():
        subprocess.run(["git", "clone", "https://github.com/sander-wood/tunesformer.git", str(tf_dir)], check=True)
    patch_tunesformer_repo(tf_dir)
    out_dir = run_paths["generated"] / "TunesFormer"
    reset_dir(out_dir)
    tf_out = tf_dir / "output_tunes"
    tf_out.mkdir(exist_ok=True)
    for old in tf_out.glob("*.abc"):
        old.unlink()
    prompt = cfg.get("prompt", "S:2 B:8 E:4\nM:6/8\nL:1/8\nK:G\n")
    (tf_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "generate.py",
            "-num_tunes",
            str(cfg["n_generate"]),
            "-max_patch",
            str(cfg["max_patch"]),
            "-top_p",
            str(cfg.get("top_p", 0.8)),
            "-top_k",
            str(cfg.get("top_k", 8)),
            "-temperature",
            str(cfg.get("temperature", 1.0)),
            "-seed",
            "42",
            "-show_control_code",
            "True",
        ],
        cwd=str(tf_dir),
        capture_output=True,
        text=True,
        timeout=7200,
    )
    log(f"TunesFormer returncode={result.returncode}.")
    (run_paths["models"] / "tunesformer_stdout.txt").write_text(result.stdout[-20000:], encoding="utf-8", errors="replace")
    (run_paths["models"] / "tunesformer_stderr.txt").write_text(result.stderr[-20000:], encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
    abc_list = []
    idx = 0
    for file in sorted(tf_out.glob("*.abc")):
        for block in split_abc_tunes(file.read_text(encoding="utf-8", errors="replace")):
            abc_list.append(block)
            (out_dir / f"tune_{idx:02d}.abc").write_text(block, encoding="utf-8")
            idx += 1
    return abc_list


def patch_melodyt5_repo(mt5_dir: Path) -> None:
    infer = mt5_dir / "inference.py"
    txt = infer.read_text(encoding="utf-8")
    txt = re.sub(
        r"torch\.load\(['\"]weights\.pth['\"](?:,\s*map_location=device)?(?:,\s*weights_only=False)?\)",
        "torch.load('weights.pth', map_location=device, weights_only=False)",
        txt,
    )
    txt = txt.replace("model.load_state_dict(checkpoint['model'])", "model.load_state_dict(checkpoint['model'], strict=False)")
    infer.write_text(txt, encoding="utf-8")

    utils = mt5_dir / "utils.py"
    utxt = utils.read_text(encoding="utf-8")
    if "import numpy as np" not in utxt:
        utxt = utxt.replace("import random\n", "import random\nimport numpy as np\n")
    old = """            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature, seed=n_seed)
"""
    new = """            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = np.asarray(prob, dtype=np.float64)
            prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
            prob = np.maximum(prob, 0.0)
            prob_sum = prob.sum()
            if (not np.isfinite(prob_sum)) or prob_sum <= 0:
                prob = np.ones_like(prob, dtype=np.float64) / len(prob)
            else:
                prob = prob / prob_sum
            token = temperature_sampling(prob, temperature=temperature, seed=n_seed)
"""
    if old in utxt and "prob = np.asarray(prob, dtype=np.float64)" not in utxt:
        utxt = utxt.replace(old, new)
    utils.write_text(utxt, encoding="utf-8")


def run_melodyt5(cfg: dict[str, Any], run_paths: dict[str, Path]) -> list[str]:
    mt5_dir = ROOT / "melodyt5_repo"
    if not mt5_dir.exists():
        subprocess.run(["git", "clone", "https://github.com/sanderwood/melodyt5.git", str(mt5_dir)], check=True)
    patch_melodyt5_repo(mt5_dir)
    weights_src = hf_hub_download("sander-wood/melodyt5", "weights.pth")
    shutil.copyfile(weights_src, mt5_dir / "weights.pth")
    if not (mt5_dir / "random_model").exists():
        subprocess.run([sys.executable, "random_model.py"], cwd=str(mt5_dir), check=True)

    out_dir = run_paths["generated"] / "MelodyT5"
    reset_dir(out_dir)
    mt5_out = mt5_dir / "output_tunes"
    mt5_out.mkdir(exist_ok=True)
    for old in mt5_out.glob("*.abc"):
        old.unlink()
    prompt = cfg.get("prompt", "%%input\n%%generation\n")
    (mt5_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "inference.py",
            "-num_tunes",
            str(cfg["n_generate"]),
            "-max_patch",
            str(cfg["max_patch"]),
            "-top_p",
            str(cfg.get("top_p", 0.8)),
            "-top_k",
            str(cfg.get("top_k", 8)),
            "-temperature",
            str(cfg.get("temperature", 2.0)),
            "-seed",
            "42",
            "-show_control_code",
            "True",
        ],
        cwd=str(mt5_dir),
        capture_output=True,
        text=True,
        timeout=7200,
    )
    log(f"MelodyT5 returncode={result.returncode}.")
    (run_paths["models"] / "melodyt5_stdout.txt").write_text(result.stdout[-20000:], encoding="utf-8", errors="replace")
    (run_paths["models"] / "melodyt5_stderr.txt").write_text(result.stderr[-20000:], encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
    abc_list = []
    idx = 0
    for file in sorted(mt5_out.glob("*.abc")):
        for block in split_abc_tunes(file.read_text(encoding="utf-8", errors="replace")):
            abc_list.append(block)
            (out_dir / f"tune_{idx:02d}.abc").write_text(block, encoding="utf-8")
            idx += 1
    return abc_list


def evaluate_model(
    abc_list: list[str],
    name: str,
    run_paths: dict[str, Path],
    real_pm_list: list[dict[str, Any]],
    real_rm_list: list[dict[str, Any]],
    real_sc_list: list[float],
    error: str = "",
) -> dict[str, Any]:
    midi_out = run_paths["generated"] / f"{name}_midi"
    reset_dir(midi_out)
    midi_paths: list[tuple[int, Path]] = []
    conversion_methods: Counter[str] = Counter()
    for i, abc in enumerate(abc_list):
        ap = midi_out / f"tune_{i:02d}.abc"
        mp = midi_out / f"tune_{i:02d}.mid"
        method = abc_to_midi(abc, ap, mp)
        if method:
            midi_paths.append((i, mp))
            conversion_methods[method] += 1

    grammar = [check_abc_grammar(a) for a in abc_list]
    key_re = re.compile(r"^K:\s*(\S+)", re.MULTILINE)
    pm_list, rm_list, sc_list = [], [], []
    for i, mp in midi_paths:
        m = key_re.search(abc_list[i])
        key = m.group(1).split("%")[0].strip() if m else "G"
        pm = compute_pitch_metrics(mp)
        rm = compute_rhythm_metrics(mp)
        sc = compute_scale_consistency(mp, key)
        if pm:
            pm_list.append(pm)
        if rm:
            rm_list.append(rm)
        if sc is not None:
            sc_list.append(sc)

    base = {
        "model": name,
        "n_generated": len(abc_list),
        "n_midi_converted": len(midi_paths),
        "n_midi_valid": len(pm_list),
        "n_conversion_failed": max(0, len(abc_list) - len(midi_paths)),
        "grammar_validity": float(np.mean(grammar)) if grammar else float("nan"),
        "test_nll": float("nan"),
        "test_ppl": float("nan"),
        "conversion_abc2midi": conversion_methods.get("abc2midi", 0),
        "conversion_music21": conversion_methods.get("music21", 0),
        "conversion_fallback": conversion_methods.get("fallback_note_stream", 0),
        "status": "failed" if error else ("no_abc" if not abc_list else ("no_valid_midi" if not pm_list else "ok")),
        "error": error,
    }
    if not pm_list:
        return base

    def mn(items: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([m[key] for m in items]))

    base.update(
        {
            "pitch_count_mean": mn(pm_list, "pitch_count"),
            "pch_entropy_mean": mn(pm_list, "pch_entropy"),
            "pitch_range_mean": mn(pm_list, "pitch_range"),
            "avg_pitch_interval_mean": mn(pm_list, "avg_pitch_interval"),
            "note_count_mean": mn(rm_list, "note_count") if rm_list else float("nan"),
            "avg_ioi_mean": mn(rm_list, "avg_ioi") if rm_list else float("nan"),
            "scale_consistency_mean": float(np.mean(sc_list)) if sc_list else float("nan"),
            "OA_pitch_count": overlapping_area([m["pitch_count"] for m in real_pm_list], [m["pitch_count"] for m in pm_list]),
            "OA_pch_entropy": overlapping_area([m["pch_entropy"] for m in real_pm_list], [m["pch_entropy"] for m in pm_list]),
            "OA_pitch_range": overlapping_area([m["pitch_range"] for m in real_pm_list], [m["pitch_range"] for m in pm_list]),
            "OA_scale_cons": overlapping_area(real_sc_list, sc_list) if sc_list else float("nan"),
            "OA_note_count": overlapping_area([m["note_count"] for m in real_rm_list], [m["note_count"] for m in rm_list]) if rm_list else float("nan"),
        }
    )
    base["OA_overall"] = float(
        np.nanmean(
            [
                base["OA_pitch_count"],
                base["OA_pch_entropy"],
                base["OA_pitch_range"],
                base["OA_scale_cons"],
            ]
        )
    )
    return base


def reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.iterdir():
        if old.is_file():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)


def make_plots(df: pd.DataFrame, run_paths: dict[str, Path], train_losses: list[float], val_losses: list[float], real_pch_avg: np.ndarray) -> None:
    plot_df = df[df["model"] != "Real_TestSet (Oracle)"].copy()
    oracle_vals = df[df["model"] == "Real_TestSet (Oracle)"].iloc[0]
    model_names = plot_df["model"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, max(1, len(model_names))))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def bplot(ax, col, title, ref_col=None):
        vals = [plot_df.iloc[i][col] if col in plot_df.columns else float("nan") for i in range(len(plot_df))]
        ax.bar(range(len(model_names)), vals, color=colors[: len(model_names)])
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ref = oracle_vals.get(ref_col or col, float("nan"))
        if not (ref is None or (isinstance(ref, float) and math.isnan(ref))):
            ax.axhline(ref, color="red", linestyle="--", linewidth=1.5, label=f"Real={ref:.3f}")
            ax.legend(fontsize=7)

    bplot(axes[0, 0], "grammar_validity", "Grammar Validity", "grammar_validity")
    bplot(axes[0, 1], "scale_consistency_mean", "Scale Consistency", "scale_consistency_mean")
    bplot(axes[0, 2], "pch_entropy_mean", "PCH Entropy", "pch_entropy_mean")
    bplot(axes[1, 0], "pitch_range_mean", "Pitch Range", "pitch_range_mean")
    bplot(axes[1, 1], "avg_pitch_interval_mean", "Avg Pitch Interval", "avg_pitch_interval_mean")
    bplot(axes[1, 2], "OA_overall", "Overlapping Area Overall", "OA_overall")
    plt.tight_layout()
    plt.savefig(run_paths["plots"] / "task1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if train_losses and val_losses:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        ep = list(range(1, len(train_losses) + 1))
        a1.plot(ep, train_losses, "b-o", label="Train NLL")
        a1.plot(ep, val_losses, "r-s", label="Val NLL")
        a1.set(xlabel="Epoch", ylabel="NLL", title="CharLSTM Training")
        a1.legend()
        a2.plot(ep, [math.exp(l) for l in train_losses], "b-o", label="Train PPL")
        a2.plot(ep, [math.exp(l) for l in val_losses], "r-s", label="Val PPL")
        a2.set(xlabel="Epoch", ylabel="Perplexity", title="CharLSTM PPL")
        a2.legend()
        plt.tight_layout()
        plt.savefig(run_paths["plots"] / "lstm_training.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], real_pch_avg, color="darkgreen")
    ax.set(title="Real Nottingham Pitch Class Histogram", xlabel="Pitch Class", ylabel="Frequency")
    plt.tight_layout()
    plt.savefig(run_paths["plots"] / "real_pch.png", dpi=150)
    plt.close(fig)


def select_final_midi(run_dir: Path, run_paths: dict[str, Path], metrics_df: pd.DataFrame) -> Path | None:
    def best_in_folder(midi_folder: Path) -> Path | None:
        best_path, best_sc, fallback_path, fallback_sc = None, -1.0, None, -1.0
        for midi_file in midi_folder.glob("*.mid"):
            abc_path = midi_file.with_suffix(".abc")
            key = "G"
            if abc_path.exists():
                m = re.search(r"^K:\s*(\S+)", abc_path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
                if m:
                    key = m.group(1).split("%")[0].strip()
            sc = compute_scale_consistency(midi_file, key)
            if sc is None:
                continue
            try:
                pm = pretty_midi.PrettyMIDI(str(midi_file))
                note_count = sum(len(i.notes) for i in pm.instruments)
                duration = pm.get_end_time()
            except Exception:
                continue
            if sc > fallback_sc:
                fallback_path, fallback_sc = midi_file, sc
            if 10 <= duration <= 180 and 50 <= note_count <= 600 and sc > best_sc:
                best_path, best_sc = midi_file, sc
        return best_path or fallback_path

    ranked = metrics_df[
        (metrics_df["model"] != "Real_TestSet (Oracle)")
        & (metrics_df.get("n_midi_valid", 0) > 0)
        & metrics_df.get("OA_overall", pd.Series(dtype=float)).notna()
    ].sort_values("OA_overall", ascending=False)
    for _, row in ranked.iterrows():
        candidate = best_in_folder(run_paths["generated"] / f"{row['model']}_midi")
        if candidate:
            shutil.copy(candidate, run_dir / "symbolic_unconditioned.mid")
            return candidate

    chosen = None
    for midi_folder in sorted(run_paths["generated"].glob("*_midi")):
        chosen = best_in_folder(midi_folder)
        if chosen:
            break
    if chosen:
        shutil.copy(chosen, run_dir / "symbolic_unconditioned.mid")
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(DEFAULT_MODE_CONFIGS), default="smoke")
    parser.add_argument("--models", default="markov,lstm,gpt2,tunesformer,melodyt5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}.")

    cfg = load_mode_config(args.mode)
    run_dir = args.run_dir or (ROOT / "runs" / args.mode)
    run_paths = ensure_dirs(run_dir)
    selected_models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    target_generate = int(cfg["n_generate"])
    qcfg = cfg.get("quality_filter", {})
    candidate_multiplier = int(qcfg.get("candidate_multiplier", 1)) if qcfg.get("enabled", False) else 1
    candidate_generate = max(target_generate, target_generate * max(1, candidate_multiplier))
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "models": sorted(selected_models),
                "device": str(device),
                "config": cfg,
                "target_generate": target_generate,
                "candidate_generate": candidate_generate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if candidate_generate > target_generate:
        log(f"Quality filter enabled: generating {candidate_generate} candidates per model, selecting {target_generate}.")

    all_tunes, train_tunes, val_tunes, test_tunes = ensure_nottingham_data(seed=args.seed)
    train_text = (DATA / "train_melody.txt").read_text(encoding="utf-8")
    val_text = (DATA / "val_melody.txt").read_text(encoding="utf-8")
    test_text = (DATA / "test_melody.txt").read_text(encoding="utf-8")

    midi_dir, test_key_map = prepare_real_test_midi(test_tunes, cfg.get("max_test_tunes"))
    oracle, real_pm_list, real_rm_list, real_sc_list, real_pch_avg = compute_oracle(midi_dir, test_key_map, run_paths)
    rows = [
        {
            "model": "Real_TestSet (Oracle)",
            **oracle,
            "test_nll": float("nan"),
            "test_ppl": float("nan"),
            "n_generated": len(list(midi_dir.glob("*.mid"))),
            "n_midi_converted": len(list(midi_dir.glob("*.mid"))),
            "n_midi_valid": len(real_pm_list),
            "n_conversion_failed": max(0, (cfg.get("max_test_tunes") or len(test_tunes)) - len(list(midi_dir.glob("*.mid")))),
            "conversion_abc2midi": float("nan"),
            "conversion_music21": float("nan"),
            "conversion_fallback": float("nan"),
            "status": "oracle",
            "error": "",
        }
    ]

    model_outputs: list[tuple[str, list[str]]] = []
    error_map: dict[str, str] = {}
    nll_map: dict[str, float] = {}
    selection_info: dict[str, dict[str, Any]] = {}
    train_losses: list[float] = []
    val_losses: list[float] = []

    if "markov" in selected_models:
        try:
            outputs, markov_nll = run_markov(train_text, val_text, test_text, run_paths, cfg.get("markov", {}), candidate_generate)
            model_outputs.extend(outputs)
            nll_map.update(markov_nll)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log(f"Markov failed: {err}")
            for model_name in ["Markov_Markov_k3_a0.1", "Markov_k2_a0.2"]:
                model_outputs.append((model_name, []))
                error_map[model_name] = err

    if "lstm" in selected_models:
        try:
            lstm_abc, lstm_test_nll, train_losses, val_losses = run_lstm(train_text, val_text, test_text, cfg["lstm"], run_paths, candidate_generate, device)
            model_outputs.append(("CharLSTM", lstm_abc))
            nll_map["CharLSTM"] = lstm_test_nll
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log(f"CharLSTM failed: {err}")
            model_outputs.append(("CharLSTM", []))
            error_map["CharLSTM"] = err

    if "gpt2" in selected_models and cfg["gpt2"].get("generate", True):
        try:
            gpt2_abc = run_gpt2(train_tunes, cfg["gpt2"], run_paths, candidate_generate, device)
            model_outputs.append(("GPT2_FT", gpt2_abc))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log(f"GPT2_FT failed: {err}")
            model_outputs.append(("GPT2_FT", []))
            error_map["GPT2_FT"] = err

    if cfg.get("run_pretrained", True) and "tunesformer" in selected_models:
        try:
            tf_cfg = dict(cfg["tunesformer"])
            tf_cfg["n_generate"] = candidate_generate
            tf_abc = run_tunesformer(tf_cfg, run_paths)
            model_outputs.append(("TunesFormer", tf_abc))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log(f"TunesFormer failed: {err}")
            model_outputs.append(("TunesFormer", []))
            error_map["TunesFormer"] = err

    if cfg.get("run_pretrained", True) and "melodyt5" in selected_models:
        try:
            mt5_cfg = dict(cfg["melodyt5"])
            mt5_cfg["n_generate"] = candidate_generate
            mt5_abc = run_melodyt5(mt5_cfg, run_paths)
            model_outputs.append(("MelodyT5", mt5_abc))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log(f"MelodyT5 failed: {err}")
            model_outputs.append(("MelodyT5", []))
            error_map["MelodyT5"] = err

    for model_name, abc_list in model_outputs:
        selected_abc, info = select_quality_candidates(
            abc_list,
            model_name,
            run_paths,
            real_pm_list,
            real_rm_list,
            real_sc_list,
            qcfg,
            target_generate,
        )
        selection_info[model_name] = info
        row = evaluate_model(selected_abc, model_name, run_paths, real_pm_list, real_rm_list, real_sc_list, error=error_map.get(model_name, ""))
        row.update(info)
        row["test_nll"] = nll_map.get(model_name, float("nan"))
        row["test_ppl"] = math.exp(row["test_nll"]) if not math.isnan(row["test_nll"]) else float("nan")
        rows.append(row)
        log(
            f"{model_name}: grammar={row.get('grammar_validity', float('nan')):.2f} "
            f"midi={row.get('n_midi_valid', 0)}/{row.get('n_generated', 0)} "
            f"candidates={row.get('n_candidates', row.get('n_generated', 0))} "
            f"passed={row.get('n_quality_passed', float('nan'))} "
            f"OA={row.get('OA_overall', float('nan')):.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics_all_models.csv", index=False)
    make_plots(df, run_paths, train_losses, val_losses, real_pch_avg)
    chosen = select_final_midi(run_dir, run_paths, df)
    log(f"Saved metrics: {run_dir / 'metrics_all_models.csv'}")
    if chosen:
        log(f"Selected final MIDI: {chosen}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
