"""Shared data/model utilities for the phase-by-phase FNO/PINO pipeline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


TIME_FREQUENCIES = 3
INPUT_CHANNELS = 7 + 2 * TIME_FREQUENCIES


def rows(dataset_root: Path, split: str | None = None) -> list[dict[str, str]]:
    with (dataset_root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        values = list(csv.DictReader(stream))
    return values if split is None else [row for row in values if row["split"] == split]


def prepared_path(dataset_root: Path, row: dict[str, str]) -> Path:
    return dataset_root / "prepared" / row["split"] / f"{row['case_id']}.h5"


def available_rows(dataset_root: Path, split: str) -> list[dict[str, str]]:
    return [row for row in rows(dataset_root, split) if prepared_path(dataset_root, row).is_file()]


def load_metadata(dataset_root: Path) -> dict[str, object]:
    return json.loads((dataset_root / "prepared" / "normalization.json").read_text())


def read_frame(path: Path, index: int) -> np.ndarray:
    with h5py.File(path, "r") as archive:
        return archive["velocity"][index].astype(np.float32)


def read_times(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as archive:
        return archive["time"][:].astype(np.float32)


def coordinate_channels(metadata: dict[str, object]) -> tuple[np.ndarray, ...]:
    x = np.asarray(metadata["x"], dtype=np.float32)
    y = np.asarray(metadata["y"], dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xn = 2 * (xx - x.min()) / (x.max() - x.min()) - 1
    yn = 2 * (yy - y.min()) / (y.max() - y.min()) - 1
    boundary = np.minimum.reduce((xn + 1, 1 - xn, yn + 1, 1 - yn))
    return xx, yy, xn, yn, boundary


def model_input(
    row: dict[str, str], time_value: float, metadata: dict[str, object]
) -> np.ndarray:
    xx, yy, xn, yn, boundary = coordinate_channels(metadata)
    sigma = 2.0 * float(metadata["dx_km"])
    x0, y0 = float(row["x0_km"]), float(row["y0_km"])
    source = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
    t0, duration = float(row["t0_s"]), float(row["T_s"])
    temporal = math.exp(-((time_value - t0) ** 2) / (2 * duration**2))
    tmax = float(metadata["time"][-1])
    tn = time_value / tmax
    center_y = float(metadata["center_y_km"])
    channels = [
        source,
        source * temporal,
        xn,
        yn,
        boundary,
        np.full_like(xx, tn),
        np.full_like(xx, (y0 - center_y) / 5.0),
    ]
    for frequency in range(1, TIME_FREQUENCIES + 1):
        phase = 2 * math.pi * frequency * tn
        channels.extend((np.full_like(xx, math.sin(phase)),
                         np.full_like(xx, math.cos(phase))))
    return np.stack(channels).astype(np.float32)


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        scale = 1.0 / width
        shape = (width, width, modes, modes)
        self.modes = modes
        self.positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def multiply(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", values, weights)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft2(values, norm="ortho")
        output = torch.zeros_like(transformed)
        mx = min(self.modes, transformed.shape[-2])
        my = min(self.modes, transformed.shape[-1])
        output[:, :, :mx, :my] = self.multiply(
            transformed[:, :, :mx, :my], self.positive[:, :, :mx, :my])
        output[:, :, -mx:, :my] = self.multiply(
            transformed[:, :, -mx:, :my], self.negative[:, :, :mx, :my])
        return torch.fft.irfft2(output, s=values.shape[-2:], norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, width: int = 32, modes: int = 16, layers: int = 4, padding: int = 8):
        super().__init__()
        self.config = dict(width=width, modes=modes, layers=layers, padding=padding)
        self.padding = padding
        self.lift = nn.Conv2d(INPUT_CHANNELS, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.norm = nn.ModuleList(nn.InstanceNorm2d(width) for _ in range(layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, 2, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.pad(self.lift(values), (0, self.padding, 0, self.padding))
        for index, (spectral, local, norm) in enumerate(
                zip(self.spectral, self.local, self.norm)):
            values = spectral(values) + local(values)
            if index + 1 < len(self.spectral):
                values = F.gelu(norm(values))
        return self.project(values[..., :-self.padding, :-self.padding])


def supervised_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    field = F.mse_loss(prediction, target) + 0.05 * F.l1_loss(prediction, target)
    dx = F.l1_loss(prediction[..., 1:, :]-prediction[..., :-1, :],
                   target[..., 1:, :]-target[..., :-1, :])
    dy = F.l1_loss(prediction[..., :, 1:]-prediction[..., :, :-1],
                   target[..., :, 1:]-target[..., :, :-1])
    spectrum = F.l1_loss(torch.log1p(torch.abs(torch.fft.rfft2(prediction))),
                         torch.log1p(torch.abs(torch.fft.rfft2(target))))
    return field + 0.05 * (dx + dy) + 0.01 * spectrum


def random_batch(
    dataset_root: Path,
    selected: list[dict[str, str]],
    metadata: dict[str, object],
    batch_size: int,
    rng: random.Random,
    triplets: bool = False,
) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor, list[int]]:
    scales = np.asarray(metadata["velocity_scale"], dtype=np.float32)
    inputs, targets, chosen, indices = [], [], [], []
    for _ in range(batch_size):
        row = rng.choice(selected)
        path = prepared_path(dataset_root, row)
        with h5py.File(path, "r") as archive:
            count = archive["velocity"].shape[0]
            index = rng.randrange(1 if triplets else 0, count - (1 if triplets else 0))
            time_values = archive["time"][index-1:index+2] if triplets else [archive["time"][index]]
            frame = archive["velocity"][index].astype(np.float32)
        inputs.extend(model_input(row, float(value), metadata) for value in time_values)
        targets.append(frame.transpose(2, 0, 1) / scales[:, None, None])
        chosen.append(row)
        indices.append(index)
    return chosen, torch.from_numpy(np.asarray(inputs)), torch.from_numpy(np.asarray(targets)), indices


@torch.no_grad()
def validation_relative_l2(
    model: nn.Module,
    dataset_root: Path,
    selected: list[dict[str, str]],
    metadata: dict[str, object],
    device: torch.device,
    frames_per_case: int = 8,
) -> float:
    model.eval()
    scales = torch.tensor(metadata["velocity_scale"], device=device).view(1, 2, 1, 1)
    error, truth = 0.0, 0.0
    for row in selected:
        path = prepared_path(dataset_root, row)
        with h5py.File(path, "r") as archive:
            indices = np.linspace(0, archive["velocity"].shape[0]-1, frames_per_case, dtype=int)
            times = archive["time"][indices]
            targets = archive["velocity"][indices].astype(np.float32).transpose(0, 3, 1, 2)
        inputs = torch.from_numpy(np.asarray([
            model_input(row, float(value), metadata) for value in times])).to(device)
        target = torch.from_numpy(targets).to(device)
        prediction = model(inputs) * scales
        error += float(torch.sum((prediction-target)**2).cpu())
        truth += float(torch.sum(target**2).cpu())
    return math.sqrt(error / max(truth, 1e-30))


def save_checkpoint(path: Path, model: FNO2d, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": model.config, **extra}, path)


def load_model(path: Path, device: torch.device) -> tuple[FNO2d, dict[str, object]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = FNO2d(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    return model, state


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, default=str) + "\n")


def now() -> float:
    return time.monotonic()
