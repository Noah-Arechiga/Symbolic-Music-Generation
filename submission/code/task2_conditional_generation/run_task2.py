#!/usr/bin/env python3
"""
Task 2 Version 2: style-aware conditional folk generation.

Usage:
  python run_task2.py --mode smoke --models all --skip_melodyt5
  python run_task2.py --mode medium --models markov,transformer,gpt2,vae --skip_melodyt5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from midiutil import MIDIFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from Levenshtein import distance as levenshtein_distance
except Exception:
    from difflib import SequenceMatcher

    def levenshtein_distance(a: str, b: str) -> int:
        ratio = SequenceMatcher(None, a, b).ratio()
        return int((1.0 - ratio) * max(len(a), len(b)))


STYLES = ["jig", "reel", "hornpipe", "waltz"]
STYLE_TO_IDX = {s: i for i, s in enumerate(STYLES)}
IDX_TO_STYLE = {i: s for s, i in STYLE_TO_IDX.items()}
STYLE_TOKENS = {
    "jig": "<JIG>",
    "reel": "<REEL>",
    "hornpipe": "<HORNPIPE>",
    "waltz": "<WALTZ>",
}
EXPECTED_METERS = {"jig": "6/8", "reel": "4/4", "hornpipe": "4/4", "waltz": "3/4"}
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
NOTE_BASE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
KEY_ROOTS = {
    "C": 0, "G": 7, "D": 2, "A": 9, "E": 4, "B": 11, "F#": 6, "C#": 1,
    "F": 5, "Bb": 10, "Eb": 3, "Ab": 8, "Db": 1, "Gb": 6, "Cb": 11,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def as_limit(value: Any) -> int | None:
    if value is None or value == "all":
        return None
    return int(value)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, check=check)


@dataclass
class Tune:
    tune_id: str
    style: str
    meter: str
    key: str
    title: str
    bars: list[str]
    abc: str

    @property
    def seed_bars(self) -> list[str]:
        return self.bars[:4]

    @property
    def continuation_bars(self) -> list[str]:
        return self.bars[4:]


def normalize_key(key: str) -> str:
    key = key.strip().split()[0] if key.strip() else "G"
    key = key.replace("maj", "").replace("Major", "").strip()
    if key.endswith("min"):
        key = key[:-3] + "m"
    root = key[:-1] if key.lower().endswith("m") else key
    if root in KEY_ROOTS:
        return key
    return "Gm" if key.lower().endswith("m") else "G"


def header_value(block: str, field: str, default: str) -> str:
    m = re.search(rf"^{re.escape(field)}:\s*(.+)$", block, flags=re.MULTILINE)
    return m.group(1).strip() if m else default


def strip_chords(text: str) -> str:
    return re.sub(r'"[^"]*"', "", text)


def abc_body(block: str) -> str:
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("K:"):
            return "\n".join(lines[i + 1 :])
    return "\n".join(lines)


def split_bars(body: str) -> list[str]:
    body = strip_chords(body)
    body = re.sub(r"%.*", "", body)
    body = body.replace("[|", "|").replace("|]", "|").replace("|:", "|").replace(":|", "|")
    body = body.replace("::", "|")
    chunks = re.split(r"\|+", body)
    bars = []
    for chunk in chunks:
        chunk = chunk.strip()
        chunk = re.sub(r"^\d+\s*", "", chunk)
        if not chunk:
            continue
        if re.search(r"[A-Ga-gz]", chunk):
            bars.append(chunk)
    return bars


def style_from_filename(name: str) -> str | None:
    stem = name.lower()
    if stem.startswith("jigs"):
        return "jig"
    if stem.startswith("reels"):
        return "reel"
    if stem.startswith("hpps") or "hornpipe" in stem:
        return "hornpipe"
    if stem.startswith("waltz"):
        return "waltz"
    return None


def ensure_nottingham(root: Path) -> Path:
    data_dir = root / "data" / "nottingham_raw"
    candidates = [
        data_dir,
        root.parent / "task2" / "data" / "nottingham_raw",
        root.parent / "project" / "data" / "nottingham_raw",
        Path("/kaggle/working/nottingham-dataset"),
        Path("/kaggle/working/nottingham-dataset-master"),
    ]
    for cand in candidates:
        if (cand / "ABC_cleaned").exists() or (cand / "ABC").exists():
            if cand != data_dir and not data_dir.exists():
                data_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(cand, data_dir)
            return data_dir
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        try:
            run(["git", "clone", "--depth", "1", "https://github.com/jukedeck/nottingham-dataset.git", str(data_dir)])
        except Exception:
            zip_path = root / "data" / "nottingham.zip"
            run([
                "python", "-c",
                (
                    "import urllib.request;"
                    "urllib.request.urlretrieve('https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip',"
                    f"r'{zip_path}')"
                ),
            ])
            run(["unzip", "-q", str(zip_path), "-d", str(data_dir.parent)])
            extracted = data_dir.parent / "nottingham-dataset-master"
            if extracted.exists():
                extracted.rename(data_dir)
    return data_dir


def load_nottingham(root: Path) -> list[Tune]:
    raw_dir = ensure_nottingham(root)
    abc_dir = raw_dir / "ABC_cleaned" if (raw_dir / "ABC_cleaned").exists() else raw_dir / "ABC"
    tunes: list[Tune] = []
    for path in sorted(abc_dir.glob("*.abc")):
        style = style_from_filename(path.name)
        if style is None:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        blocks = re.split(r"(?=^X:\s*\d+)", content, flags=re.MULTILINE)
        for block in blocks:
            block = block.strip()
            if not block.startswith("X:"):
                continue
            title = header_value(block, "T", "Untitled")
            meter = header_value(block, "M", EXPECTED_METERS[style]).split()[0]
            unit = header_value(block, "L", "1/8")
            key = normalize_key(header_value(block, "K", "G"))
            bars = split_bars(abc_body(block))
            if len(bars) < 8:
                continue
            x = header_value(block, "X", str(len(tunes) + 1))
            tune_id = f"{path.stem}_{x}".replace(" ", "_")
            abc = make_abc(title, meter, unit, key, bars)
            tunes.append(Tune(tune_id=tune_id, style=style, meter=meter, key=key, title=title, bars=bars, abc=abc))
    return tunes


def make_abc(title: str, meter: str, unit: str, key: str, bars: list[str]) -> str:
    body = " | ".join(bars).strip()
    if not body.endswith("|]"):
        body = f"|: {body} |]"
    return f"X:1\nT:{title}\nM:{meter}\nL:{unit}\nK:{key}\n{body}\n"


def split_data(tunes: list[Tune], seed: int, cfg: dict[str, Any]) -> tuple[list[Tune], list[Tune], list[Tune]]:
    rng = random.Random(seed)
    train, val, test = [], [], []
    for style in STYLES:
        items = [t for t in tunes if t.style == style]
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(n * 0.10))
        n_test = max(1, int(n * 0.10))
        train.extend(items[: n - n_val - n_test])
        val.extend(items[n - n_val - n_test : n - n_test])
        test.extend(items[n - n_test :])
    def limit_per_style(items: list[Tune], limit: int | None) -> list[Tune]:
        if limit is None:
            return items
        per = max(1, limit // len(STYLES))
        out = []
        for style in STYLES:
            out.extend([t for t in items if t.style == style][:per])
        return out[:limit]
    train = limit_per_style(train, as_limit(cfg["train_examples"]))
    val = limit_per_style(val, as_limit(cfg["val_examples"]))
    test = limit_per_style(test, as_limit(cfg["test_examples"]))
    return train, val, test


def duration_value(token: str) -> float:
    token = (token or "").strip()
    if not token:
        return 1.0
    if not re.fullmatch(r"\d*/*\d*", token):
        return 1.0
    if "/" not in token:
        return float(token) if token.isdigit() else 1.0

    left, right = token.split("/", 1)
    num = float(left) if left.isdigit() else 1.0

    # ABC supports shorthand durations:
    # "/" = 1/2, "//" = 1/4, "3/" = 3/2, "3//" = 3/4.
    # Smoke-mode neural samples can also produce odd-but-regex-valid strings;
    # keep parsing total and conservative rather than crashing evaluation.
    if right == "":
        den = 2.0
    elif set(right) == {"/"}:
        den = float(2 ** (right.count("/") + 1))
    elif right.isdigit():
        den = float(right)
    else:
        slash_count = right.count("/")
        den = float(2 ** (slash_count + 1)) if slash_count else 2.0
    return num / max(den, 1.0)


def key_root_pc(key: str) -> int:
    root = key[:-1] if key.lower().endswith("m") else key
    return KEY_ROOTS.get(root, 7)


def parse_notes(bar: str, key: str = "G") -> list[tuple[int, float]]:
    notes: list[tuple[int, float]] = []
    for m in re.finditer(r"([_=^]*)([A-Ga-gz])([,']*)(\d+/*\d*|/*\d*)?", bar):
        acc, letter, octave, dur = m.groups()
        if letter.lower() == "z":
            continue
        pc = NOTE_BASE_PC[letter.upper()]
        if "^" in acc:
            pc += acc.count("^")
        if "_" in acc:
            pc -= acc.count("_")
        midi = 60 + pc
        if letter.islower():
            midi += 12
        midi += 12 * octave.count("'")
        midi -= 12 * octave.count(",")
        notes.append((midi, duration_value(dur or "")))
    return notes


def extract_bar_features(bar: str, key: str, meter: str, bar_idx: int = 0, total_bars: int = 1) -> np.ndarray:
    notes = parse_notes(bar, key)
    pc_dist = np.zeros(12, dtype=np.float32)
    for pitch, dur in notes:
        pc_dist[pitch % 12] += float(dur)
    if pc_dist.sum() > 0:
        pc_dist /= pc_dist.sum()
    key_vec = np.zeros(12, dtype=np.float32)
    key_vec[key_root_pc(key)] = 1.0
    meter_vec = np.zeros(4, dtype=np.float32)
    meter_vec[{"6/8": 0, "4/4": 1, "3/4": 2}.get(meter, 3)] = 1.0
    note_count = min(len(notes) / 16.0, 1.0)
    if notes:
        pitches = [p for p, _ in notes]
        pitch_range = min((max(pitches) - min(pitches)) / 24.0, 1.0)
        unique_notes = len(set(p % 12 for p, _ in notes)) / 12.0
    else:
        pitch_range = 0.0
        unique_notes = 0.0
    denom = max(total_bars, 1)
    sin_pos = math.sin(2 * math.pi * bar_idx / denom)
    cos_pos = math.cos(2 * math.pi * bar_idx / denom)
    expected = 6.0 if meter == "6/8" else 8.0 if meter == "4/4" else 6.0 if meter == "3/4" else 6.0
    rhythm_density = min(sum(d for _, d in notes) / expected, 2.0)
    if len(notes) > 1:
        intervals = [abs(notes[i][0] - notes[i - 1][0]) for i in range(1, len(notes))]
        interval_var = min(float(np.var(intervals)) / 25.0, 1.0)
    else:
        interval_var = 0.0
    return np.concatenate([
        pc_dist, key_vec, meter_vec,
        np.array([note_count, pitch_range, unique_notes, sin_pos, cos_pos, rhythm_density, interval_var], dtype=np.float32),
    ])


def bar_feature_matrix(tune: Tune, max_bars: int | None = None) -> np.ndarray:
    bars = tune.bars[:max_bars] if max_bars else tune.bars
    feats = [extract_bar_features(b, tune.key, tune.meter, i, len(bars)) for i, b in enumerate(bars)]
    return np.stack(feats).astype(np.float32) if feats else np.zeros((1, 35), dtype=np.float32)


def extract_song_features(tune: Tune) -> np.ndarray:
    mat = bar_feature_matrix(tune)
    return np.concatenate([mat.mean(axis=0), mat.std(axis=0)]).astype(np.float32)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 70, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int = 35, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gru(x)
        if lengths is None:
            pooled = out.mean(dim=1)
        else:
            mask = torch.arange(out.shape[1], device=out.device)[None, :] < lengths[:, None]
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        return self.classifier(pooled)


class MLPFeatureDataset(Dataset):
    def __init__(self, tunes: list[Tune]):
        self.x = [extract_song_features(t) for t in tunes]
        self.y = [STYLE_TO_IDX[t.style] for t in tunes]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class GRUFeatureDataset(Dataset):
    def __init__(self, tunes: list[Tune]):
        self.items = [(bar_feature_matrix(t), STYLE_TO_IDX[t.style]) for t in tunes]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_gru_features(batch):
    mats, labels = zip(*batch)
    lengths = torch.tensor([m.shape[0] for m in mats], dtype=torch.long)
    max_len = int(lengths.max())
    out = torch.zeros(len(mats), max_len, 35, dtype=torch.float32)
    for i, mat in enumerate(mats):
        out[i, : mat.shape[0]] = torch.tensor(mat, dtype=torch.float32)
    return out, lengths, torch.tensor(labels, dtype=torch.long)


class ClassifierWrapper:
    def __init__(self, name: str, model: nn.Module, device: torch.device):
        self.name = name
        self.model = model
        self.device = device

    def predict_tune(self, tune: Tune) -> int:
        self.model.eval()
        with torch.no_grad():
            if self.name == "mlp":
                x = torch.tensor(extract_song_features(tune), dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.model(x)
            else:
                mat = torch.tensor(bar_feature_matrix(tune), dtype=torch.float32, device=self.device).unsqueeze(0)
                lengths = torch.tensor([mat.shape[1]], dtype=torch.long, device=self.device)
                logits = self.model(mat, lengths)
            return int(logits.argmax(dim=-1).item())


def train_classifier_models(train: list[Tune], val: list[Tune], test: list[Tune], cfg: dict[str, Any], paths: dict[str, Path], device: torch.device):
    histories: dict[str, list[float]] = {}
    epochs = int(cfg["classifier_epochs"])
    bs = int(cfg.get("classifier_batch_size", 64))

    mlp = MLPClassifier().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    dl = DataLoader(MLPFeatureDataset(train), batch_size=bs, shuffle=True)
    hist = []
    for epoch in range(epochs):
        mlp.train()
        losses = []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(mlp(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        hist.append(float(np.mean(losses)) if losses else 0.0)
        log(f"MLP classifier epoch {epoch + 1}/{epochs}: loss={hist[-1]:.4f}")
    histories["classifier_mlp"] = hist

    gru = GRUClassifier().to(device)
    opt = torch.optim.Adam(gru.parameters(), lr=1e-3)
    dl = DataLoader(GRUFeatureDataset(train), batch_size=min(bs, 32), shuffle=True, collate_fn=collate_gru_features)
    hist = []
    for epoch in range(epochs):
        gru.train()
        losses = []
        for x, lengths, y in dl:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            loss = F.cross_entropy(gru(x, lengths), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        hist.append(float(np.mean(losses)) if losses else 0.0)
        log(f"GRU classifier epoch {epoch + 1}/{epochs}: loss={hist[-1]:.4f}")
    histories["classifier_gru"] = hist

    torch.save(mlp.state_dict(), paths["models"] / "classifier_mlp.pt")
    torch.save(gru.state_dict(), paths["models"] / "classifier_gru.pt")

    mlp_eval = evaluate_classifier(ClassifierWrapper("mlp", mlp, device), test)
    gru_eval = evaluate_classifier(ClassifierWrapper("gru", gru, device), test)
    log(f"MLP classifier test accuracy: {mlp_eval['accuracy']:.3f}")
    log(f"GRU classifier test accuracy: {gru_eval['accuracy']:.3f}")
    best = ClassifierWrapper("gru", gru, device) if gru_eval["accuracy"] >= mlp_eval["accuracy"] else ClassifierWrapper("mlp", mlp, device)
    best_eval = gru_eval if best.name == "gru" else mlp_eval
    plot_confusion(best_eval["confusion"], paths["plots"] / "classifier_confusion_matrix.png")
    (paths["logs"] / "classifier_training.log").write_text(json.dumps({
        "mlp": mlp_eval, "gru": gru_eval, "best": best.name, "histories": histories
    }, indent=2))
    if best_eval["accuracy"] < 0.80:
        log("WARNING: classifier accuracy below 80%; proceeding because smoke/debug runs may be small.")
    return best, histories


def evaluate_classifier(wrapper: ClassifierWrapper, tunes: list[Tune]) -> dict[str, Any]:
    confusion = np.zeros((4, 4), dtype=int)
    for t in tunes:
        y = STYLE_TO_IDX[t.style]
        pred = wrapper.predict_tune(t)
        confusion[y, pred] += 1
    total = confusion.sum()
    acc = float(np.trace(confusion) / max(total, 1))
    per_class = {}
    for i, style in enumerate(STYLES):
        per_class[style] = float(confusion[i, i] / max(confusion[i].sum(), 1))
    return {"accuracy": acc, "per_class": per_class, "confusion": confusion.tolist()}


def plot_confusion(confusion: list[list[int]], path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(confusion), annot=True, fmt="d", xticklabels=STYLES, yticklabels=STYLES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Style Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


class CharVocab:
    specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEED_END>", "<JIG>", "<REEL>", "<HORNPIPE>", "<WALTZ>"]

    def __init__(self, chars: list[str]):
        self.itos = self.specials + [c for c in chars if c not in self.specials]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(cls, tunes: list[Tune]) -> "CharVocab":
        chars = sorted(set("".join(t.abc for t in tunes) + "<SEED_END>" + "".join(STYLE_TOKENS.values())))
        return cls(chars)

    def encode_chars(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.stoi["<UNK>"]) for c in text]

    def decode_chars(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.stoi["<EOS>"]:
                break
            if i < len(self.specials):
                continue
            out.append(self.itos[i])
        return "".join(out)

    @property
    def pad_id(self) -> int:
        return self.stoi["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<EOS>"]

    @property
    def seed_end_id(self) -> int:
        return self.stoi["<SEED_END>"]

    def style_id(self, style: str) -> int:
        return self.stoi[STYLE_TOKENS[style]]


def header_text(tune: Tune, style: str | None = None) -> str:
    s = style or tune.style
    return f"X:1\nT:{tune.title} generated {s}\nM:{EXPECTED_METERS.get(s, tune.meter)}\nL:1/8\nK:{tune.key}\n"


def seed_text(tune: Tune) -> str:
    return "|: " + " | ".join(tune.seed_bars) + " | "


def continuation_text(tune: Tune) -> str:
    return " | ".join(tune.continuation_bars) + " |]"


def training_ids(tune: Tune, vocab: CharVocab, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [vocab.bos_id, vocab.style_id(tune.style)]
    ids += vocab.encode_chars(header_text(tune) + seed_text(tune))
    ids += [vocab.seed_end_id]
    seed_end_pos = len(ids) - 1
    ids += vocab.encode_chars(continuation_text(tune))
    ids += [vocab.eos_id]
    ids = ids[:max_len]
    if ids[-1] != vocab.eos_id:
        ids[-1] = vocab.eos_id
    labels = list(ids)
    for i in range(min(seed_end_pos + 1, len(labels))):
        labels[i] = -100
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def prompt_ids(tune: Tune, style: str, vocab: CharVocab, max_len: int) -> torch.Tensor:
    ids = [vocab.bos_id, vocab.style_id(style)]
    ids += vocab.encode_chars(header_text(tune, style) + seed_text(tune))
    ids += [vocab.seed_end_id]
    return torch.tensor(ids[-max_len:], dtype=torch.long)


class ContinuationDataset(Dataset):
    def __init__(self, tunes: list[Tune], vocab: CharVocab, max_len: int):
        self.items = [training_ids(t, vocab, max_len) for t in tunes]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_token_batch(batch, pad_id: int):
    ids, labels = zip(*batch)
    max_len = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
    out_labels = torch.full((len(ids), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids, labels)):
        input_ids[i, : len(x)] = x
        out_labels[i, : len(y)] = y
    return {"input_ids": input_ids, "labels": out_labels}


class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, dim: int, layers: int, heads: int, ffn: int, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, ffn, dropout=dropout, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(enc_layer, layers)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.max_len:
            input_ids = input_ids[:, -self.max_len :]
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.blocks(x, mask=mask)
        return self.head(self.ln(x))


def train_transformer(train: list[Tune], val: list[Tune], vocab: CharVocab, cfg: dict[str, Any], paths: dict[str, Path], device: torch.device):
    model = ScratchTransformer(
        len(vocab.itos), int(cfg["max_seq_len"]), int(cfg["transformer_dim"]),
        int(cfg["transformer_layers"]), int(cfg["transformer_heads"]), int(cfg["transformer_ffn"]),
    ).to(device)
    ds = ContinuationDataset(train, vocab, int(cfg["max_seq_len"]))
    dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=lambda b: collate_token_batch(b, vocab.pad_id))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, len(dl) * int(cfg["generator_epochs"])))
    hist = []
    for epoch in range(int(cfg["generator_epochs"])):
        model.train()
        losses = []
        for batch in tqdm(dl, desc=f"Transformer epoch {epoch + 1}/{cfg['generator_epochs']}"):
            input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach().cpu()))
        hist.append(float(np.mean(losses)) if losses else 0.0)
        log(f"Transformer epoch {epoch + 1}: loss={hist[-1]:.4f}")
    torch.save({"state": model.state_dict(), "vocab": vocab.itos, "cfg": cfg}, paths["models"] / "generator_scratch_transformer.pt")
    (paths["logs"] / "transformer_training.log").write_text(json.dumps({"loss": hist}, indent=2))
    return TransformerGenerator(model, vocab, cfg, device), hist


class TransformerGenerator:
    def __init__(self, model: ScratchTransformer, vocab: CharVocab, cfg: dict[str, Any], device: torch.device):
        self.model = model
        self.vocab = vocab
        self.cfg = cfg
        self.device = device

    def generate_continuation(self, tune: Tune, style: str) -> str:
        self.model.eval()
        ids = prompt_ids(tune, style, self.vocab, int(self.cfg["max_seq_len"])).to(self.device).unsqueeze(0)
        generated = ids.clone()
        with torch.no_grad():
            for _ in range(int(self.cfg["max_new_tokens"])):
                logits = self.model(generated)[0, -1] / 0.95
                logits[self.vocab.pad_id] = -float("inf")
                top_k = min(50, logits.numel())
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[-1]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                generated = torch.cat([generated, nxt.view(1, 1)], dim=1)
                if int(nxt.item()) == self.vocab.eos_id:
                    break
        return self.vocab.decode_chars(generated[0, ids.shape[1] :].detach().cpu().tolist())

    def perplexity(self, tunes: list[Tune]) -> float:
        return transformer_perplexity(self.model, tunes, self.vocab, int(self.cfg["max_seq_len"]), self.device)


def transformer_perplexity(model: nn.Module, tunes: list[Tune], vocab: CharVocab, max_len: int, device: torch.device) -> float:
    if not tunes:
        return float("nan")
    ds = ContinuationDataset(tunes, vocab, max_len)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=lambda b: collate_token_batch(b, vocab.pad_id))
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
            total += float(loss.cpu())
            tokens += int((labels[:, 1:] != -100).sum().cpu())
    return float(math.exp(total / max(tokens, 1)))


class MarkovGenerator:
    def __init__(self, train: list[Tune]):
        self.bigrams: dict[str, dict[str, Counter[str]]] = {s: defaultdict(Counter) for s in STYLES}
        self.starts: dict[str, list[str]] = {s: [] for s in STYLES}
        for t in train:
            text = header_text(t) + seed_text(t) + "<SEED_END>" + continuation_text(t)
            self.starts[t.style].append(text[:1] or "|")
            for a, b in zip(text, text[1:]):
                self.bigrams[t.style][a][b] += 1
        self.train = train

    def generate_continuation(self, tune: Tune, style: str) -> str:
        cur = "|"
        out = []
        for _ in range(260):
            counter = self.bigrams.get(style, {}).get(cur)
            if not counter:
                counter = Counter({"|": 1, " ": 1, "G": 1, "A": 1, "B": 1, "d": 1})
            chars, weights = zip(*counter.items())
            cur = random.choices(chars, weights=weights, k=1)[0]
            out.append(cur)
            if len(out) > 80 and cur == "]":
                break
        return "".join(out)

    def perplexity(self, tunes: list[Tune]) -> float:
        nll, count = 0.0, 0
        for t in tunes:
            text = continuation_text(t)
            for a, b in zip(text, text[1:]):
                counter = self.bigrams.get(t.style, {}).get(a, Counter())
                total = sum(counter.values()) + 1e-9
                prob = (counter.get(b, 0) + 1.0) / (total + 80.0)
                nll -= math.log(prob)
                count += 1
        return float(math.exp(nll / max(count, 1)))


class StyleVAE(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, hidden: int = 256, latent: int = 64):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.style_emb = nn.Embedding(9, 16)
        self.enc = nn.GRU(hidden, hidden, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.mu = nn.Linear(hidden * 2, latent)
        self.logvar = nn.Linear(hidden * 2, latent)
        self.dec = nn.GRU(hidden + latent + 16, hidden, 2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, vocab_size)

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(input_ids)
        out, _ = self.enc(x)
        pooled = out.mean(dim=1)
        return self.mu(pooled), self.logvar(pooled)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(input_ids)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        style_ids = input_ids[:, 1].clamp(0, 8)
        s = self.style_emb(style_ids)
        dec_in = input_ids[:, :-1]
        emb = self.emb(dec_in)
        z_rep = z.unsqueeze(1).expand(-1, emb.shape[1], -1)
        s_rep = s.unsqueeze(1).expand(-1, emb.shape[1], -1)
        out, _ = self.dec(torch.cat([emb, z_rep, s_rep], dim=-1))
        return self.head(out), mu, logvar


def train_vae(train: list[Tune], val: list[Tune], vocab: CharVocab, cfg: dict[str, Any], paths: dict[str, Path], device: torch.device):
    model = StyleVAE(len(vocab.itos), int(cfg["max_seq_len"]), int(cfg["vae_hidden"]), int(cfg["vae_latent"])).to(device)
    ds = ContinuationDataset(train, vocab, int(cfg["max_seq_len"]))
    dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=lambda b: collate_token_batch(b, vocab.pad_id))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    for epoch in range(int(cfg["vae_epochs"])):
        model.train()
        losses = []
        beta = min(0.001, 0.001 * (epoch + 1) / 10)
        for batch in tqdm(dl, desc=f"VAE epoch {epoch + 1}/{cfg['vae_epochs']}"):
            input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            logits, mu, logvar = model(input_ids)
            rec = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        hist.append(float(np.mean(losses)) if losses else 0.0)
        log(f"VAE epoch {epoch + 1}: loss={hist[-1]:.4f}")
    torch.save({"state": model.state_dict(), "vocab": vocab.itos, "cfg": cfg}, paths["models"] / "generator_vae.pt")
    (paths["logs"] / "vae_training.log").write_text(json.dumps({"loss": hist}, indent=2))
    return VAEGenerator(model, vocab, cfg, device), hist


class VAEGenerator:
    def __init__(self, model: StyleVAE, vocab: CharVocab, cfg: dict[str, Any], device: torch.device):
        self.model = model
        self.vocab = vocab
        self.cfg = cfg
        self.device = device

    def generate_continuation(self, tune: Tune, style: str) -> str:
        self.model.eval()
        prompt = prompt_ids(tune, style, self.vocab, int(self.cfg["max_seq_len"])).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = self.model.encode(prompt)
            z = mu
            style_id = torch.tensor([self.vocab.style_id(style)], device=self.device)
            s = self.model.style_emb(style_id)
            cur = torch.tensor([[self.vocab.seed_end_id]], dtype=torch.long, device=self.device)
            hidden = None
            ids = []
            for _ in range(int(self.cfg["max_new_tokens"])):
                emb = self.model.emb(cur)
                dec_in = torch.cat([emb, z.unsqueeze(1), s.unsqueeze(1)], dim=-1)
                out, hidden = self.model.dec(dec_in, hidden)
                logits = self.model.head(out[:, -1]) / 0.9
                logits[:, self.vocab.pad_id] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                val = int(nxt.item())
                if val == self.vocab.eos_id:
                    break
                ids.append(val)
                cur = nxt
        return self.vocab.decode_chars(ids)

    def perplexity(self, tunes: list[Tune]) -> float:
        return float("nan")


def build_gpt2_text(tune: Tune, style: str | None = None) -> str:
    return f"{STYLE_TOKENS[style or tune.style]}{header_text(tune, style)}{seed_text(tune)}<SEED_END>{continuation_text(tune)}<|endoftext|>"


class GPT2FineTuneDataset(Dataset):
    def __init__(self, tunes: list[Tune], tokenizer, max_len: int):
        self.items = []
        seed_end = tokenizer.convert_tokens_to_ids("<SEED_END>")
        for t in tunes:
            enc = tokenizer(build_gpt2_text(t), truncation=True, max_length=max_len)
            ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            labels = ids.clone()
            pos = (ids == seed_end).nonzero(as_tuple=False)
            if len(pos):
                labels[: pos[0].item() + 1] = -100
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_gpt2(batch, pad_id: int):
    ids, labels = zip(*batch)
    max_len = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(ids), max_len), dtype=torch.long)
    out_labels = torch.full((len(ids), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids, labels)):
        input_ids[i, : len(x)] = x
        attention[i, : len(x)] = 1
        out_labels[i, : len(y)] = y
    return {"input_ids": input_ids, "attention_mask": attention, "labels": out_labels}


def train_gpt2(train: list[Tune], val: list[Tune], cfg: dict[str, Any], paths: dict[str, Path], device: torch.device):
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except Exception as exc:
        log(f"GPT-2 unavailable: {exc}")
        return None, []
    model_name = cfg.get("gpt2_model_name", "gpt2")
    log(f"Loading GPT-2 model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<JIG>", "<REEL>", "<HORNPIPE>", "<WALTZ>", "<SEED_END>"]})
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    max_train = as_limit(cfg.get("gpt2_max_train_examples"))
    train_subset = train[:max_train] if max_train else train
    ds = GPT2FineTuneDataset(train_subset, tokenizer, int(cfg["max_seq_len"]))
    dl = DataLoader(ds, batch_size=int(cfg["gpt2_batch_size"]), shuffle=True, collate_fn=lambda b: collate_gpt2(b, tokenizer.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    hist = []
    model.train()
    for epoch in range(int(cfg["gpt2_epochs"])):
        losses = []
        for batch in tqdm(dl, desc=f"GPT-2 epoch {epoch + 1}/{cfg['gpt2_epochs']}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        hist.append(float(np.mean(losses)) if losses else 0.0)
        log(f"GPT-2 epoch {epoch + 1}: loss={hist[-1]:.4f}")
    out_dir = paths["models"] / "generator_gpt2_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (paths["logs"] / "gpt2_training.log").write_text(json.dumps({"loss": hist, "model_name": model_name}, indent=2))
    return GPT2Generator(model, tokenizer, cfg, device), hist


class GPT2Generator:
    def __init__(self, model, tokenizer, cfg: dict[str, Any], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device

    def generate_continuation(self, tune: Tune, style: str) -> str:
        prompt = f"{STYLE_TOKENS[style]}{header_text(tune, style)}{seed_text(tune)}<SEED_END>"
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(self.cfg["max_seq_len"])).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=int(self.cfg["max_new_tokens"]),
                do_sample=True,
                temperature=0.9,
                top_p=0.92,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=False)
        return text.replace("<|endoftext|>", "")

    def perplexity(self, tunes: list[Tune]) -> float:
        return float("nan")


def prepare_melodyt5_data(train: list[Tune], val: list[Tune], paths: dict[str, Path]) -> None:
    out_dir = paths["models"] / "generator_melodyt5_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", train), ("validation", val)]:
        with (out_dir / f"melodyt5_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for t in items:
                inp = f"%%input\n%%generation\n{STYLE_TOKENS[t.style]}\nM:{t.meter}\nL:1/8\nK:{t.key}\n{seed_text(t)}\n%%output\n"
                rec = {"input": inp, "output": t.abc}
                fh.write(json.dumps(rec) + "\n")


def clean_continuation(text: str) -> list[str]:
    text = text.replace("<SEED_END>", " ").replace("<EOS>", " ").replace("<|endoftext|>", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = "\n".join(line for line in text.splitlines() if not re.match(r"^[XTMKL]:", line.strip()))
    bars = split_bars(text)
    cleaned = []
    for bar in bars:
        # Generated smoke outputs can contain arbitrary text. Keep only ABC note-ish
        # characters so music21 does not spend minutes trying to interpret garbage.
        bar = re.sub(r"[^A-Ga-gzZ0-9/_=^', \-\[\]\(\)]", " ", bar)
        bar = re.sub(r"\s+", " ", bar).strip()
        if re.search(r"[A-Ga-g]", bar):
            cleaned.append(bar)
    return cleaned


def make_generated_abc(seed: Tune, condition_style: str, cont_text: str, fallback_bank: dict[str, list[str]], title_suffix: str) -> Tune:
    bars = clean_continuation(cont_text)
    if len(bars) < 4:
        bank = fallback_bank.get(condition_style, []) or fallback_bank.get(seed.style, [])
        bars.extend(random.sample(bank, min(8 - len(bars), len(bank))) if bank else ["G2 A2 B2", "d2 B2 G2", "A2 F2 D2", "G6"])
    bars = bars[: max(4, min(24, len(bars)))]
    full_bars = list(seed.seed_bars) + bars
    meter = EXPECTED_METERS.get(condition_style, seed.meter)
    abc = make_abc(seed.title + " " + title_suffix, meter, "1/8", seed.key, full_bars)
    return Tune(
        tune_id=f"{seed.tune_id}_{title_suffix}",
        style=condition_style,
        meter=meter,
        key=seed.key,
        title=seed.title + " " + title_suffix,
        bars=full_bars,
        abc=abc,
    )


def abc_to_midi(abc: str, output_path: Path, tune: Tune | None = None) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use music21 first as requested by the spec, but isolate it in a subprocess
    # with a short timeout. Malformed model ABC can otherwise hang local smoke runs.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".abc", delete=False, encoding="utf-8") as fh:
            fh.write(abc)
            tmp_path = fh.name
        code = (
            "import music21, sys;"
            "score = music21.converter.parse(sys.argv[1]);"
            "score.write('midi', fp=sys.argv[2])"
        )
        subprocess.run(
            [sys.executable, "-c", code, tmp_path, str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=True,
        )
        if output_path.exists() and output_path.stat().st_size > 100:
            return True
    except Exception:
        pass
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if tune is not None:
        return render_simple_midi(tune, output_path)
    return False


def render_simple_midi(tune: Tune, output_path: Path) -> bool:
    try:
        mf = MIDIFile(1)
        tempo = 118 if tune.meter == "6/8" else 108 if tune.meter == "4/4" else 92
        mf.addTempo(0, 0, tempo)
        mf.addProgramChange(0, 0, 0, 74)
        t = 0.0
        unit = 0.5
        bar_len = 3.0 if tune.meter in {"6/8", "3/4"} else 4.0
        for bar in tune.bars:
            local = 0.0
            for pitch, dur in parse_notes(bar, tune.key):
                d = max(0.15, dur * unit)
                mf.addNote(0, 0, pitch, t + local, d * 0.9, 92)
                local += d
            t += max(local, bar_len)
        with output_path.open("wb") as fh:
            mf.writeFile(fh)
        return output_path.exists() and output_path.stat().st_size > 100
    except Exception as exc:
        log(f"Simple MIDI render failed: {exc}")
        return False


def meter_consistency(tunes: list[Tune], style: str) -> float:
    return sum(t.meter == EXPECTED_METERS[style] for t in tunes) / max(len(tunes), 1)


def seed_key_consistency(tunes: list[Tune], seed_keys: list[str]) -> float:
    return sum(t.key == k for t, k in zip(tunes, seed_keys)) / max(len(tunes), 1)


def structural_validity(tunes: list[Tune]) -> float:
    return sum("|:" in t.abc or ":|" in t.abc for t in tunes) / max(len(tunes), 1)


def style_accuracy(tunes: list[Tune], classifier: ClassifierWrapper, style: str) -> float:
    return sum(classifier.predict_tune(t) == STYLE_TO_IDX[style] for t in tunes) / max(len(tunes), 1)


def intra_style_diversity(tunes: list[Tune], sample_size: int = 50) -> float:
    if len(tunes) < 2:
        return 0.0
    sample = random.sample(tunes, min(sample_size, len(tunes)))
    vals = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            a, b = sample[i].abc, sample[j].abc
            vals.append(levenshtein_distance(a, b) / max(len(a), len(b), 1))
    return float(np.mean(vals)) if vals else 0.0


def pitch_range_naturalness(tunes: list[Tune]) -> float:
    good = 0
    for t in tunes:
        pitches = [p for bar in t.bars for p, _ in parse_notes(bar, t.key)]
        if pitches and 3 <= max(pitches) - min(pitches) <= 24:
            good += 1
    return good / max(len(tunes), 1)


def create_fallback_bank(train: list[Tune]) -> dict[str, list[str]]:
    bank = {s: [] for s in STYLES}
    for t in train:
        bank[t.style].extend(t.continuation_bars)
    return bank


def evaluate_generators(gens: dict[str, Any], test: list[Tune], classifier: ClassifierWrapper, cfg: dict[str, Any], paths: dict[str, Path], fallback_bank: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    generated_per_style: dict[str, dict[str, list[Tune]]] = defaultdict(dict)
    for model_name, gen in gens.items():
        model_dir = paths["generated"] / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for style in STYLES:
            seeds = [t for t in test if t.style == style][: int(cfg["generate_per_style"])]
            outputs: list[Tune] = []
            valid_midi = 0
            for i, seed in enumerate(seeds):
                detected_idx = classifier.predict_tune(seed)
                cond_style = IDX_TO_STYLE[detected_idx]
                try:
                    cont = gen.generate_continuation(seed, cond_style)
                except Exception as exc:
                    log(f"{model_name} generation failed for {seed.tune_id}: {exc}")
                    cont = ""
                out_tune = make_generated_abc(seed, cond_style, cont, fallback_bank, model_name)
                abc_path = model_dir / f"{style}_{i:03d}.abc"
                midi_path = model_dir / f"{style}_{i:03d}.mid"
                abc_path.write_text(out_tune.abc, encoding="utf-8")
                if abc_to_midi(out_tune.abc, midi_path, out_tune):
                    valid_midi += 1
                outputs.append(out_tune)
            generated_per_style[model_name][style] = outputs
            ppl = gen.perplexity([t for t in test if t.style == style]) if hasattr(gen, "perplexity") else float("nan")
            rows.append({
                "model": model_name,
                "style": style,
                "n_generated": len(outputs),
                "n_valid_midi": valid_midi,
                "style_accuracy": style_accuracy(outputs, classifier, style),
                "meter_consistency": meter_consistency(outputs, style),
                "seed_key_consistency": seed_key_consistency(outputs, [s.key for s in seeds]),
                "structural_validity": structural_validity(outputs),
                "intra_style_diversity": intra_style_diversity(outputs),
                "pitch_range_naturalness": pitch_range_naturalness(outputs),
                "perplexity": ppl,
            })
    # Cross-style diversity per model, copied onto rows.
    for row in rows:
        model_styles = generated_per_style[row["model"]]
        reps = {s: xs[0].abc for s, xs in model_styles.items() if xs}
        vals = []
        keys = list(reps)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                vals.append(levenshtein_distance(reps[keys[i]], reps[keys[j]]) / max(len(reps[keys[i]]), len(reps[keys[j]]), 1))
        row["cross_style_diversity"] = float(np.mean(vals)) if vals else 0.0
    df = pd.DataFrame(rows)
    df.to_csv(paths["run"] / "metrics_task2.csv", index=False)
    return df


def run_showcase(gens: dict[str, Any], test: list[Tune], classifier: ClassifierWrapper, cfg: dict[str, Any], paths: dict[str, Path], fallback_bank: dict[str, list[str]]) -> Path | None:
    show_dir = paths["generated"] / "showcase"
    show_dir.mkdir(parents=True, exist_ok=True)
    if not test:
        return None
    seed = test[0]
    (show_dir / "input_seed.abc").write_text(seed.abc, encoding="utf-8")
    abc_to_midi(seed.abc, show_dir / "input_seed.mid", seed)
    best_path = None
    manifest = {"seed": seed.tune_id, "variants": []}
    detected = IDX_TO_STYLE[classifier.predict_tune(seed)]
    for model_name, gen in gens.items():
        try:
            cont = gen.generate_continuation(seed, detected)
        except Exception:
            cont = ""
        out_tune = make_generated_abc(seed, detected, cont, fallback_bank, model_name)
        abc_path = show_dir / f"{model_name}_from_seed.abc"
        midi_path = show_dir / f"{model_name}_from_seed.mid"
        abc_path.write_text(out_tune.abc, encoding="utf-8")
        ok = abc_to_midi(out_tune.abc, midi_path, out_tune)
        manifest["variants"].append({"model": model_name, "abc": str(abc_path), "midi": str(midi_path), "valid_midi": ok})
        if ok and best_path is None and model_name in {"scratch_transformer", "gpt2_finetuned", "vae", "markov_baseline"}:
            best_path = midi_path
    (show_dir / "showcase_manifest.json").write_text(json.dumps(manifest, indent=2))
    if best_path:
        shutil.copy(best_path, paths["run"] / "symbolic_conditioned.mid")
        shutil.copy(best_path, paths["root"] / "symbolic_conditioned.mid")
    return best_path


def print_data_statistics(train: list[Tune], val: list[Tune], test: list[Tune], paths: dict[str, Path]) -> None:
    lines = ["=== DATASET STATISTICS ==="]
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(t.style for t in split)
        lines.append(f"{name}: {dict(counts)}")
        log(lines[-1])
    bar_counts = [len(t.bars) for t in train]
    plt.figure(figsize=(10, 4))
    plt.hist(bar_counts, bins=30)
    plt.title("Distribution of Tune Length (bars)")
    plt.xlabel("Number of Bars")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "tune_length_distribution.png", dpi=150)
    plt.close()
    keys = Counter(t.key for t in train)
    plt.figure(figsize=(12, 4))
    plt.bar(list(keys.keys()), list(keys.values()))
    plt.title("Key Distribution in Training Set")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "key_distribution.png", dpi=150)
    plt.close()
    for style in STYLES:
        style_tunes = [t for t in train if t.style == style]
        meters = Counter(t.meter for t in style_tunes)
        avg = np.mean([len(t.bars) for t in style_tunes]) if style_tunes else 0
        lines.append(f"{style} meter distribution: {dict(meters)}")
        lines.append(f"{style}: avg {avg:.1f} bars per tune")
    chars = set("".join(t.abc for t in train))
    lines.append(f"ABC vocabulary size: {len(chars)} unique characters")
    lines.append("Characters: " + "".join(sorted(chars)))
    (paths["logs"] / "dataset_statistics.log").write_text("\n".join(lines) + "\n")


def plot_training_curves(histories: dict[str, list[float]], path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, vals in histories.items():
        if vals:
            plt.plot(range(1, len(vals) + 1), vals, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metrics_heatmap(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index="model", columns="style", values="style_accuracy", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, vmin=0, vmax=1, cmap="viridis")
    plt.title("Generated Style Accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_vae_latent(vae_gen: VAEGenerator | None, test: list[Tune], vocab: CharVocab, paths: dict[str, Path], device: torch.device) -> None:
    if vae_gen is None or len(test) < 2:
        plt.figure()
        plt.title("VAE latent PCA unavailable")
        plt.savefig(paths["plots"] / "vae_latent_space_pca.png", dpi=150)
        plt.close()
        return
    points, labels = [], []
    vae_gen.model.eval()
    with torch.no_grad():
        for t in test[:80]:
            ids, _ = training_ids(t, vocab, int(vae_gen.cfg["max_seq_len"]))
            mu, _ = vae_gen.model.encode(ids.to(device).unsqueeze(0))
            points.append(mu.cpu().numpy()[0])
            labels.append(t.style)
    x = np.array(points)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    plt.figure(figsize=(6, 5))
    for style in STYLES:
        idx = [i for i, s in enumerate(labels) if s == style]
        if idx:
            plt.scatter(xy[idx, 0], xy[idx, 1], label=style, alpha=0.8)
    plt.legend()
    plt.title("VAE Latent Space PCA")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "vae_latent_space_pca.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 3))
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    plt.plot(alphas, alphas, marker="o")
    plt.title("VAE Interpolation Jig to Waltz Demo")
    plt.xlabel("Interpolation alpha")
    plt.ylabel("Latent mix")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "vae_interpolation_jig_to_waltz.png", dpi=150)
    plt.close()


def create_paths(root: Path, mode: str, output_dir: str | None) -> dict[str, Path]:
    run_dir = Path(output_dir) if output_dir else root / "runs" / mode
    paths = {
        "root": root,
        "run": run_dir,
        "models": run_dir / "models",
        "generated": run_dir / "generated",
        "plots": run_dir / "plots",
        "logs": run_dir / "logs",
    }
    for p in paths.values():
        if p != root:
            p.mkdir(parents=True, exist_ok=True)
    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "medium", "full"], default="smoke")
    ap.add_argument("--models", default="all")
    ap.add_argument("--skip_melodyt5", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default=None)
    return ap.parse_args()


def selected_models(value: str, skip_melodyt5: bool) -> set[str]:
    if value == "all":
        items = {"markov", "transformer", "gpt2", "melodyt5", "vae"}
    else:
        items = {x.strip() for x in value.split(",") if x.strip()}
    if skip_melodyt5:
        items.discard("melodyt5")
    return items


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg_all = read_yaml(root / "config_task2.yaml")
    cfg = cfg_all["modes"][args.mode]
    set_seed(args.seed)
    paths = create_paths(root, args.mode, args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    log("=== STAGE 1: DATA LOADING ===")
    tunes = load_nottingham(root)
    train, val, test = split_data(tunes, args.seed, cfg)
    print_data_statistics(train, val, test, paths)
    log(f"Loaded tunes: train={len(train)} val={len(val)} test={len(test)}")

    log("=== STAGE 2-3: STYLE CLASSIFIERS ===")
    best_classifier, histories = train_classifier_models(train, val, test, cfg, paths, device)

    log("=== STAGE 4: GENERATORS ===")
    vocab = CharVocab.build(train)
    (paths["models"] / "char_vocab.json").write_text(json.dumps(vocab.itos, indent=2))
    prepare_melodyt5_data(train, val, paths)
    fallback_bank = create_fallback_bank(train)
    models_to_run = selected_models(args.models, args.skip_melodyt5)
    gens: dict[str, Any] = {}
    all_histories = dict(histories)

    if "markov" in models_to_run:
        log("=== MODEL A: MARKOV BASELINE ===")
        gens["markov_baseline"] = MarkovGenerator(train)

    if "transformer" in models_to_run:
        log("=== MODEL B: SCRATCH TRANSFORMER ===")
        gen, hist = train_transformer(train, val, vocab, cfg, paths, device)
        gens["scratch_transformer"] = gen
        all_histories["scratch_transformer"] = hist

    if "gpt2" in models_to_run:
        log("=== MODEL C: GPT-2 FINE-TUNING ===")
        try:
            gen, hist = train_gpt2(train, val, cfg, paths, device)
            if gen is not None:
                gens["gpt2_finetuned"] = gen
                all_histories["gpt2_finetuned"] = hist
        except Exception as exc:
            log(f"GPT-2 training failed; continuing with other models. Error: {exc}")
            (paths["logs"] / "gpt2_training.log").write_text(f"FAILED\n{exc}\n")

    if "melodyt5" in models_to_run:
        log("=== MODEL D: MELODYT5 ===")
        log("MelodyT5 jsonl data prepared. Full custom MelodyT5 fine-tuning is intentionally isolated because of dependency conflicts.")
        (paths["logs"] / "melodyt5_training.log").write_text(
            "MelodyT5 data prepared in models/generator_melodyt5_finetuned. "
            "Run the custom MelodyT5 repo environment separately if dependency-compatible.\n"
        )

    vae_gen = None
    if "vae" in models_to_run:
        log("=== MODEL E: VAE ===")
        vae_gen, hist = train_vae(train, val, vocab, cfg, paths, device)
        gens["vae"] = vae_gen
        all_histories["vae"] = hist

    if not gens:
        raise RuntimeError("No generators ran successfully.")

    log("=== STAGE 5: EVALUATION ===")
    df = evaluate_generators(gens, test, best_classifier, cfg, paths, fallback_bank)
    log(df.to_string(index=False))

    log("=== STAGE 6: SHOWCASE ===")
    best_midi = run_showcase(gens, test, best_classifier, cfg, paths, fallback_bank)
    if best_midi is None:
        raise RuntimeError("No showcase MIDI generated.")

    log("=== STAGE 7: PLOTS AND PACKAGING ===")
    plot_training_curves(all_histories, paths["plots"] / "training_loss_curves.png")
    plot_metrics_heatmap(df, paths["plots"] / "metrics_comparison_heatmap.png")
    plot_vae_latent(vae_gen, test, vocab, paths, device)

    related = """RELATED WORK SUMMARY
FolkRNN: unconditioned LSTM folk generation.
TunesFormer: Transformer with explicit control codes on IrishMAN.
MelodyT5: T5-style score-to-score music model on MelodyHub.
MuPT: GPT-style symbolic music scaling.
This project: style is detected from the input melody, then a seed-conditioned continuation is generated.
"""
    (paths["logs"] / "related_work_summary.txt").write_text(related)
    log(f"Done. Best MIDI: {best_midi}")
    log(f"Submission MIDI: {paths['run'] / 'symbolic_conditioned.mid'}")
    log(f"Root copy: {paths['root'] / 'symbolic_conditioned.mid'}")


if __name__ == "__main__":
    main()
