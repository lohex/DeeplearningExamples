"""Reproducible BBBC021 morphology patch extraction with group-aware splits."""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from PIL import Image
from tqdm.auto import tqdm

COMPOUND = "Image_Metadata_Compound"
CONCENTRATION = "Image_Metadata_Concentration"
PLATE = "Image_Metadata_Plate_DAPI"
WELL = "Image_Metadata_Well_DAPI"
CHANNELS = {
    "dapi": ("Image_PathName_DAPI", "Image_FileName_DAPI"),
    "actin": ("Image_PathName_Actin", "Image_FileName_Actin"),
    "tubulin": ("Image_PathName_Tubulin", "Image_FileName_Tubulin"),
}
TARGETS = {
    "DMSO": "control",
    "cytochalasin B": "actin_disruption",
    "nocodazole": "microtubule_destabilization",
    "taxol": "microtubule_stabilization",
    "AZ-A": "aurora_kinase_inhibition",
}


@dataclass(frozen=True, slots=True)
class PatchExtractionConfig:
    base_url: str = "https://data.broadinstitute.org/bbbc/BBBC021/"
    work_dir: Path = Path("bbbc021_morphology")
    patch_size: int = 224
    total_patches: int = 200_000
    shard_size: int = 2_000
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    lower_percentile: float = 0.5
    upper_percentile: float = 99.8
    minimum_foreground_fraction: float = 0.02


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    work: Path
    archives: Path
    images: Path
    output: Path


def create_paths(config: PatchExtractionConfig) -> PipelinePaths:
    paths = PipelinePaths(
        work=config.work_dir,
        archives=config.work_dir / "archives",
        images=config.work_dir / "images",
        output=config.work_dir / "patches_wds",
    )
    for path in (paths.work, paths.archives, paths.images, paths.output):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_selected_metadata(config: PatchExtractionConfig, paths: PipelinePaths) -> pd.DataFrame:
    metadata_path = _download(
        urljoin(config.base_url, "BBBC021_v1_image.csv"),
        paths.work / "BBBC021_v1_image.csv",
    )
    metadata = pd.read_csv(metadata_path)
    available = {
        str(value).casefold(): str(value)
        for value in metadata[COMPOUND].dropna().unique()
    }
    missing = [compound for compound in TARGETS if compound.casefold() not in available]
    if missing:
        raise ValueError(f"Missing compounds: {missing}")

    selected = [available[compound.casefold()] for compound in TARGETS]
    subset = metadata[metadata[COMPOUND].isin(selected)].copy()
    class_map = {available[name.casefold()]: label for name, label in TARGETS.items()}
    subset["class_name"] = subset[COMPOUND].map(class_map)
    subset["group_id"] = subset[PLATE].astype(str) + "::" + subset[WELL].astype(str)
    return subset


def assign_splits(metadata: pd.DataFrame, config: PatchExtractionConfig) -> pd.DataFrame:
    split_frames: list[pd.DataFrame] = []
    for class_name, group in metadata.groupby("class_name", sort=True):
        group = group.copy()
        group_ids = group["group_id"].drop_duplicates().to_numpy()
        if len(group_ids) < 3:
            raise ValueError(f"{class_name} has only {len(group_ids)} wells.")

        rng = np.random.default_rng(config.seed)
        rng.shuffle(group_ids)
        training_count = max(1, round(config.train_fraction * len(group_ids)))
        validation_count = max(1, round(config.validation_fraction * len(group_ids)))
        if training_count + validation_count >= len(group_ids):
            training_count = len(group_ids) - 2
            validation_count = 1

        mapping = {value: "train" for value in group_ids[:training_count]}
        mapping.update(
            {
                value: "validation"
                for value in group_ids[training_count : training_count + validation_count]
            }
        )
        mapping.update({value: "test" for value in group_ids[training_count + validation_count :]})
        group["split"] = group["group_id"].map(mapping)
        split_frames.append(group)

    return pd.concat(split_frames, ignore_index=True)


def source_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby(["class_name", "split"])
        .agg(
            fovs=("ImageNumber", "size"),
            wells=("group_id", "nunique"),
            plates=(PLATE, "nunique"),
        )
        .reset_index()
    )


def download_source_images(
    metadata: pd.DataFrame,
    config: PatchExtractionConfig,
    paths: PipelinePaths,
) -> pd.DataFrame:
    plates = sorted(metadata[PLATE].astype(str).unique())
    manifest = pd.DataFrame({"plate": plates})
    manifest["archive"] = manifest["plate"].map(
        lambda plate: f"BBBC021_v1_images_{plate}.zip"
    )
    manifest["url"] = manifest["archive"].map(
        lambda name: urljoin(config.base_url, name)
    )
    manifest["bytes"] = [
        _remote_size(url)
        for url in tqdm(manifest["url"], desc="Inspecting archives")
    ]
    manifest["GiB"] = manifest["bytes"] / 2**30

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Downloading",
    ):
        _download(row.url, paths.archives / row.archive)

    needed = {
        str(value)
        for _, filename_column in CHANNELS.values()
        for value in metadata[filename_column].dropna()
    }
    found: set[str] = set()
    for archive_path in tqdm(sorted(paths.archives.glob("*.zip")), desc="Extracting TIFFs"):
        with zipfile.ZipFile(archive_path) as archive:
            matches = {
                Path(member.filename).name: member
                for member in archive.infolist()
                if Path(member.filename).name in needed
            }
            for name, member in matches.items():
                destination = paths.images / name
                if not destination.exists():
                    with archive.open(member) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                found.add(name)

    missing = needed - found
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} TIFFs, examples: {sorted(missing)[:10]}"
        )
    return manifest


def pilot_storage_estimate(
    metadata: pd.DataFrame,
    config: PatchExtractionConfig,
    paths: PipelinePaths,
    sample_size: int = 100,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    sampled = metadata.sample(min(sample_size, len(metadata)), random_state=config.seed)
    encoded_sizes: list[int] = []
    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Pilot patches"):
        patch, _, _ = _random_patch(
            _load_fov(row, config, paths),
            rng,
            config.patch_size,
        )
        encoded_sizes.append(len(_encode_png(patch)))

    raw_uint8 = config.total_patches * config.patch_size**2 * 3 / 2**30
    return pd.DataFrame(
        {
            "representation": [
                "PNG pilot estimate",
                "raw uint8",
                "raw uint16",
                "raw float32",
            ],
            "GiB": [
                np.mean(encoded_sizes) * config.total_patches / 2**30,
                raw_uint8,
                2 * raw_uint8,
                4 * raw_uint8,
            ],
        }
    )


def allocate_patches(config: PatchExtractionConfig) -> pd.DataFrame:
    per_class, remainder = divmod(config.total_patches, len(TARGETS))
    if remainder:
        raise ValueError("total_patches must be divisible by the number of classes.")

    per_split = _largest_remainder_counts(
        per_class,
        {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": 1.0 - config.train_fraction - config.validation_fraction,
        },
    )
    return pd.DataFrame(
        [
            {"class_name": class_name, "split": split, "patches": count}
            for class_name in sorted(TARGETS.values())
            for split, count in per_split.items()
        ]
    )


def generate_dataset(
    metadata: pd.DataFrame,
    allocation: pd.DataFrame,
    config: PatchExtractionConfig,
    paths: PipelinePaths,
) -> pd.DataFrame:
    for shard in paths.output.glob("bbbc021-*.tar"):
        shard.unlink()

    records: list[dict[str, object]] = []
    rng = np.random.default_rng(config.seed)
    with _ShardWriter(paths.output, config.shard_size) as writer:
        for class_name in sorted(TARGETS.values()):
            for split in ("train", "validation", "test"):
                group = metadata[
                    (metadata["class_name"] == class_name)
                    & (metadata["split"] == split)
                ]
                patch_count = int(
                    allocation[
                        (allocation["class_name"] == class_name)
                        & (allocation["split"] == split)
                    ]["patches"].iloc[0]
                )
                _generate_group(
                    group,
                    patch_count,
                    class_name,
                    split,
                    writer,
                    records,
                    rng,
                    config,
                    paths,
                )

    manifest = pd.DataFrame(records)
    manifest.to_csv(paths.output / "manifest.csv", index=False)
    (paths.output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str)
    )
    return manifest


def validate_manifest(
    manifest: pd.DataFrame,
    config: PatchExtractionConfig,
) -> pd.DataFrame:
    if len(manifest) != config.total_patches:
        raise ValueError(
            f"Expected {config.total_patches} patches, received {len(manifest)}."
        )
    if manifest.groupby("class_name").size().nunique() != 1:
        raise ValueError("Class allocation is not balanced.")
    if manifest.groupby(["plate", "well"])["split"].nunique().max() != 1:
        raise ValueError("A plate-well group occurs in more than one split.")

    return (
        manifest.groupby(["class_name", "split"])
        .agg(
            patches=("key", "size"),
            source_fovs=("source_fov_id", "nunique"),
            wells=("well", "nunique"),
            plates=("plate", "nunique"),
            patches_per_fov=(
                "source_fov_id",
                lambda values: len(values) / values.nunique(),
            ),
        )
        .reset_index()
    )


def _download(url: str, destination: Path, chunk_size: int = 2**20) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    headers: dict[str, str] = {}
    mode = "wb"
    if temporary.exists():
        headers["Range"] = f"bytes={temporary.stat().st_size}-"
        mode = "ab"

    with requests.get(url, stream=True, timeout=120, headers=headers) as response:
        if response.status_code == 200 and mode == "ab":
            temporary.unlink()
            mode = "wb"
        response.raise_for_status()
        with temporary.open(mode) as output:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    output.write(chunk)

    temporary.replace(destination)
    return destination


def _remote_size(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        return int(value) if value else None
    except requests.RequestException:
        return None


def _normalize_channel(
    channel: NDArray[np.generic],
    config: PatchExtractionConfig,
) -> NDArray[np.uint8]:
    values = channel.astype(np.float32)
    low = np.percentile(values, config.lower_percentile)
    high = np.percentile(values, config.upper_percentile)
    if high <= low:
        return np.zeros_like(values, dtype=np.uint8)
    return np.round(
        255 * np.clip((values - low) / (high - low), 0, 1)
    ).astype(np.uint8)


def _load_fov(
    row: pd.Series,
    config: PatchExtractionConfig,
    paths: PipelinePaths,
) -> NDArray[np.uint8]:
    channels = []
    for channel_name in ("dapi", "actin", "tubulin"):
        _, filename_column = CHANNELS[channel_name]
        with Image.open(paths.images / str(row[filename_column])) as image:
            channels.append(_normalize_channel(np.asarray(image), config))

    if len({channel.shape for channel in channels}) != 1:
        raise ValueError("Channel shape mismatch.")
    return np.stack(channels, axis=-1)


def _random_patch(
    image: NDArray[np.uint8],
    rng: np.random.Generator,
    patch_size: int,
) -> tuple[NDArray[np.uint8], int, int]:
    height, width, _ = image.shape
    if height < patch_size or width < patch_size:
        raise ValueError(f"FOV {image.shape} is smaller than the requested patch.")

    y = int(rng.integers(0, height - patch_size + 1))
    x = int(rng.integers(0, width - patch_size + 1))
    return image[y : y + patch_size, x : x + patch_size], y, x


def _encode_png(image: NDArray[np.uint8]) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def _largest_remainder_counts(
    total: int,
    fractions: dict[str, float],
) -> dict[str, int]:
    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        raw,
        key=lambda key: raw[key] - counts[key],
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _row_id(row: pd.Series) -> str:
    value = f"{row[PLATE]}|{row[WELL]}|{row['ImageNumber']}"
    return hashlib.sha1(value.encode()).hexdigest()[:12]


def _generate_group(
    group: pd.DataFrame,
    count: int,
    class_name: str,
    split: str,
    writer: _ShardWriter,
    records: list[dict[str, object]],
    rng: np.random.Generator,
    config: PatchExtractionConfig,
    paths: PipelinePaths,
) -> None:
    rows = [row for _, row in group.iterrows()]
    if not rows:
        raise ValueError(f"No FOVs for {class_name}/{split}.")

    cache: dict[str, NDArray[np.uint8]] = {}
    used: dict[str, set[tuple[int, int]]] = {}
    accepted = 0
    attempts = 0
    progress = tqdm(total=count, desc=f"{class_name}/{split}")

    while accepted < count and attempts < 20 * count:
        row = rows[accepted % len(rows)]
        source_id = _row_id(row)
        if source_id not in cache:
            cache[source_id] = _load_fov(row, config, paths)
            used[source_id] = set()

        patch, y, x = _random_patch(
            cache[source_id],
            rng,
            config.patch_size,
        )
        attempts += 1
        if (y, x) in used[source_id]:
            continue
        used[source_id].add((y, x))

        foreground = float((np.max(patch, axis=-1) >= 8).mean())
        if foreground < config.minimum_foreground_fraction:
            continue

        key = f"{writer.count:09d}"
        record = {
            "key": key,
            "class_name": class_name,
            "split": split,
            "compound": str(row[COMPOUND]),
            "concentration": float(row[CONCENTRATION]),
            "plate": str(row[PLATE]),
            "well": str(row[WELL]),
            "fov_image_number": int(row["ImageNumber"]),
            "source_fov_id": source_id,
            "patch_y": y,
            "patch_x": x,
            "patch_size": config.patch_size,
            "channel_order": ["DAPI", "Actin", "Tubulin"],
        }
        writer.write(key, _encode_png(patch), record)
        records.append(record)
        accepted += 1
        progress.update(1)

        if len(cache) > 16:
            cache.pop(next(iter(cache)))

    progress.close()
    if accepted < count:
        raise RuntimeError(
            f"Only generated {accepted}/{count} patches for {class_name}/{split}."
        )


class _ShardWriter:
    def __init__(self, output: Path, shard_size: int) -> None:
        self.output = output
        self.shard_size = shard_size
        self.count = 0
        self.shard_index = -1
        self.archive: tarfile.TarFile | None = None

    def __enter__(self) -> _ShardWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _next_archive(self) -> None:
        if self.archive is not None:
            self.archive.close()
        self.shard_index += 1
        self.archive = tarfile.open(
            self.output / f"bbbc021-{self.shard_index:05d}.tar",
            "w",
        )

    def write(self, key: str, png: bytes, metadata: dict[str, object]) -> None:
        if self.count % self.shard_size == 0:
            self._next_archive()
        if self.archive is None:
            raise RuntimeError("Shard archive is not open.")

        payloads = {
            f"{key}.png": png,
            f"{key}.json": json.dumps(metadata, sort_keys=True).encode(),
        }
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            self.archive.addfile(info, io.BytesIO(payload))
        self.count += 1

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()
            self.archive = None
