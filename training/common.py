from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch


EPS = 1.0e-8


TEXT_FIELDS = (
    "text",
    "prompt",
    "instruction",
    "player_input",
    "query",
    "request",
    "goal",
    "input",
    "content",
)


def sha256_file(
    path: str | Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    rows = []

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSONL {path}, "
                    f"line {line_number}: {error}"
                ) from error

            if not isinstance(
                row,
                dict,
            ):
                raise TypeError(
                    f"Expected JSON object in "
                    f"{path}, line {line_number}."
                )

            rows.append(row)

    return rows




def normalize_text(
    text: str,
) -> str:
    """
    Canonical normalization used by the formal detector
    data pipeline for hashing and deduplication.
    """
    return re.sub(
        r"\s+",
        " ",
        str(text).strip(),
    ).casefold()



def text_hash(
    text: str,
) -> str:
    return hashlib.sha256(
        normalize_text(text).encode(
            "utf-8"
        )
    ).hexdigest()



def load_source(
    source: str | Path,
) -> list[dict[str, Any]]:
    source_str = str(source)

    if source_str.startswith(
        "hf://"
    ):
        try:
            from datasets import (
                load_dataset,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install `datasets` "
                "for hf:// sources."
            ) from exc

        parsed = urlparse(
            source_str
        )

        dataset_id = (
            f"{parsed.netloc}"
            f"{parsed.path}"
        ).strip("/")

        query = parse_qs(
            parsed.query
        )

        split = query.get(
            "split",
            ["train"],
        )[0]

        config = query.get(
            "config",
            [None],
        )[0]

        return [
            dict(row)
            for row in load_dataset(
                dataset_id,
                config,
                split=split,
            )
        ]

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(
            f"Data source not found: "
            f"{path}"
        )

    suffix = (
        path.suffix.lower()
    )

    if suffix in {
        ".jsonl",
        ".ndjson",
    }:
        return read_jsonl(
            path
        )

    if suffix == ".json":
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            value,
            list,
        ):
            return [
                (
                    row
                    if isinstance(
                        row,
                        dict,
                    )
                    else {
                        "text": str(row)
                    }
                )
                for row in value
            ]

        if isinstance(
            value,
            dict,
        ):
            for key in (
                "data",
                "rows",
                "samples",
                "records",
            ):
                if isinstance(
                    value.get(key),
                    list,
                ):
                    return [
                        (
                            row
                            if isinstance(
                                row,
                                dict,
                            )
                            else {
                                "text": str(
                                    row
                                )
                            }
                        )
                        for row
                        in value[key]
                    ]

            return [value]

        return [
            {
                "text": str(value)
            }
        ]

    if suffix == ".csv":
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return [
                dict(row)
                for row
                in csv.DictReader(file)
            ]

    if suffix in {
        ".txt",
        ".text",
    }:
        return [
            {
                "text": line.strip()
            }
            for line
            in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    raise ValueError(
        f"Unsupported source: "
        f"{path}"
    )



def extract_text(
    row: dict[str, Any],
    text_field: str | None = None,
) -> str:
    if text_field:
        if text_field not in row:
            raise KeyError(
                f"Field {text_field!r} "
                f"not found; available="
                f"{sorted(row)}"
            )

        text = str(
            row[text_field]
        ).strip()

        if (
            text_field
            == "instruction"
            and row.get("input")
        ):
            text = (
                f"{text}\n\n"
                f"Input:\n"
                f"{row['input']}"
            ).strip()

        return text

    for field in TEXT_FIELDS:
        value = row.get(
            field
        )

        if (
            value is None
            or not str(value).strip()
        ):
            continue

        text = str(
            value
        ).strip()

        if (
            field == "instruction"
            and row.get("input")
        ):
            text = (
                f"{text}\n\n"
                f"Input:\n"
                f"{row['input']}"
            ).strip()

        return text

    raise KeyError(
        "Cannot infer text field; "
        f"available={sorted(row)}"
    )



def load_text_pool(
    source: str | Path,
    text_field: str | None = None,
) -> list[dict[str, Any]]:
    output = []
    seen = set()

    source_str = str(source)

    # Keep reusable provenance without leaking
    # machine-specific absolute paths.
    if source_str.startswith("hf://"):
        source_label = source_str
    else:
        source_label = Path(
            source_str
        ).name

    for index, row in enumerate(
        load_source(source)
    ):
        text = extract_text(
            row,
            text_field,
        )

        key = text_hash(
            text
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            {
                "text": text,
                "text_hash": key,
                "source": source_label,
                "source_index": index,
            }
        )

    if not output:
        raise RuntimeError(
            f"No usable text in "
            f"{source}"
        )

    return output



def deterministic_partition(
    rows: list[dict[str, Any]],
    counts: list[int],
    seed: int,
) -> list[list[dict[str, Any]]]:
    required = sum(
        counts
    )

    if len(rows) < required:
        raise ValueError(
            f"Need {required} unique "
            f"samples, found "
            f"{len(rows)}"
        )

    indices = list(
        range(len(rows))
    )

    random.Random(
        seed
    ).shuffle(
        indices
    )

    selected = [
        rows[index]
        for index
        in indices[:required]
    ]

    output = []
    start = 0

    for count in counts:
        output.append(
            selected[
                start:
                start + count
            ]
        )

        start += count

    return output



def write_jsonl(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

def _safe_torch_load(
    path: str | Path,
) -> Any:
    """
    Load MATE-generated detector/cache files.

    Current public artifacts contain tensors and simple Python
    containers only.
    """
    path = Path(path)

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch.
        return torch.load(
            path,
            map_location="cpu",
        )


def load_cache(
    path: str | Path,
) -> dict[str, Any]:
    cache = _safe_torch_load(
        path
    )

    if not isinstance(
        cache,
        dict,
    ):
        raise TypeError(
            "Hidden-state cache must "
            "contain a dictionary."
        )

    required = {
        "rows",
        "features",
        "manifest_sha256",
    }

    missing = (
        required - set(cache)
    )

    if missing:
        raise KeyError(
            "Hidden-state cache missing "
            f"fields: {sorted(missing)}"
        )

    features = cache["features"]

    if not isinstance(
        features,
        torch.Tensor,
    ):
        raise TypeError(
            "cache['features'] must "
            "be a torch.Tensor."
        )

    if features.ndim != 3:
        raise ValueError(
            "Expected hidden cache shape "
            "[samples, layers, hidden_size], "
            f"got {tuple(features.shape)}."
        )

    if len(
        cache["rows"]
    ) != features.shape[0]:
        raise ValueError(
            "Number of manifest rows does "
            "not match hidden features."
        )

    return cache


def load_bundle(
    path: str | Path,
) -> dict[str, Any]:
    bundle = _safe_torch_load(
        path
    )

    if not isinstance(
        bundle,
        dict,
    ):
        raise TypeError(
            "Detector artifact must "
            "contain a dictionary."
        )

    return bundle


def validate_layers(
    cache: dict[str, Any],
    layers: list[int],
) -> None:
    if not layers:
        raise ValueError(
            "At least one layer is required."
        )

    num_layers = int(
        cache["features"].shape[1]
    )

    for layer in layers:
        if layer < 0:
            raise ValueError(
                f"Negative layer: {layer}"
            )

        if layer >= num_layers:
            raise IndexError(
                f"Requested layer {layer}, "
                f"but cache has {num_layers} layers."
            )


def risk_family_subset(
    cache: dict[str, Any],
    risk_family: str,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    list[dict[str, Any]],
]:
    """
    Original Multi-RAAC uses category-specific positive AND
    category-specific benign samples for each head.
    """
    rows = cache["rows"]

    indices = [
        index
        for index, row
        in enumerate(rows)
        if (
            str(
                row.get(
                    "risk_family",
                    "",
                )
            ).lower()
            == risk_family.lower()
        )
    ]

    if not indices:
        raise RuntimeError(
            f"No {risk_family!r} samples "
            "found in cache."
        )

    selected_rows = [
        rows[index]
        for index in indices
    ]

    labels = np.asarray(
        [
            int(row["label"])
            for row in selected_rows
        ],
        dtype=np.int64,
    )

    if set(
        labels.tolist()
    ) != {0, 1}:
        raise RuntimeError(
            f"{risk_family}: both positive "
            "and negative labels are required."
        )

    features = (
        cache["features"][indices]
        .float()
    )

    return (
        features,
        labels,
        selected_rows,
    )


def score_mean_difference(
    features: torch.Tensor,
    bundle: dict[str, Any],
) -> np.ndarray:
    """
    Runtime-equivalent Original Multi-RAAC score.

    For each selected layer:

        score_l = h_l^T v_l - b_l

    The head score is the arithmetic mean over layers.
    """
    layers = [
        int(layer)
        for layer in bundle["layers"]
    ]

    selected = (
        features[:, layers, :]
        .float()
    )

    directions = torch.stack(
        [
            bundle["vec"][layer]
            .detach()
            .float()
            for layer in layers
        ],
        dim=0,
    )

    bias = torch.tensor(
        [
            float(
                bundle["bias"][layer]
            )
            for layer in layers
        ],
        dtype=torch.float32,
    )

    scores = (
        torch.einsum(
            "nlh,lh->nl",
            selected,
            directions,
        )
        - bias
    ).mean(dim=1)

    return (
        scores
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )


def score_logistic_hidden(
    features: torch.Tensor,
    bundle: dict[str, Any],
) -> np.ndarray:
    layers = [
        int(layer)
        for layer in bundle["layers"]
    ]

    vector = (
        features[:, layers, :]
        .float()
        .reshape(
            features.shape[0],
            -1,
        )
    )

    indices = bundle.get(
        "feature_indices"
    )

    if indices is not None:
        vector = vector.index_select(
            1,
            indices.long(),
        )

    mean = (
        bundle["scaler_mean"]
        .float()
    )

    scale = (
        bundle["scaler_scale"]
        .float()
        .clamp_min(EPS)
    )

    coef = bundle["coef"].float()

    logits = (
        (
            vector - mean
        )
        / scale
    ) @ coef + float(
        bundle["intercept"]
    )

    return (
        torch.sigmoid(logits)
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )


def score_hidden_bundle(
    features: torch.Tensor,
    bundle: dict[str, Any],
) -> np.ndarray:
    method = str(
        bundle.get(
            "method",
            "",
        )
    )

    if method == "mean_difference":
        return score_mean_difference(
            features,
            bundle,
        )

    if method == "logistic_hidden":
        return score_logistic_hidden(
            features,
            bundle,
        )

    raise ValueError(
        "Unsupported hidden detector "
        f"method: {method}"
    )


def normalized_margin(
    scores: np.ndarray,
    threshold: float,
    scale: float,
) -> np.ndarray:
    return (
        np.asarray(
            scores,
            dtype=np.float64,
        )
        - float(threshold)
    ) / max(
        float(scale),
        EPS,
    )
