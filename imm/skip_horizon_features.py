from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


FEATURE_NAMES = (
    "skip_available_frac",
    "skip_best_h_norm",
    "skip_best_sharpness",
    "skip_best_entropy_norm",
    "skip_best_max_p",
    "skip_best_log_mass_norm",
    "skip_mean_sharpness",
    "skip_late_best_sharpness",
    "skip_local_sharpness",
)

COUNT_PRIOR_EPS = 1.0e-8


@dataclass(frozen=True)
class SkipHorizonArtifact:
    db_path: Path
    ids_path: Path
    meta_path: Path
    meta: dict[str, object]
    cache_loaded: bool


def ids_digest(ids: Sequence[int]) -> str:
    return hashlib.sha256(bytes(ids)).hexdigest()


def vocab_digest(chars: Sequence[str]) -> str:
    payload = json.dumps(list(chars), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def skip_horizon_meta(
    train_ids: Sequence[int],
    vocab_chars: Sequence[str],
    depth: int,
    horizon: int,
) -> dict[str, object]:
    return {
        "schema": "skip_horizon_artifact_v1",
        "train_digest": ids_digest(train_ids),
        "vocab_digest": vocab_digest(vocab_chars),
        "vocab_chars": list(vocab_chars),
        "depth": depth,
        "horizon": horizon,
        "builder": "crystal_sqlite_id_horizon_arrays",
        "tokenization": "train_vocab_ids_u8",
    }


def skip_horizon_count_meta(
    train_ids: Sequence[int],
    vocab_chars: Sequence[str],
    depth: int,
    horizon: int,
) -> dict[str, object]:
    return {
        "schema": "skip_horizon_counts_artifact_v1",
        "train_digest": ids_digest(train_ids),
        "vocab_digest": vocab_digest(vocab_chars),
        "vocab_chars": list(vocab_chars),
        "depth": depth,
        "horizon": horizon,
        "builder": "crystal_sqlite_id_horizon_counts",
        "tokenization": "train_vocab_ids_u8",
    }


def ensure_ids_fit_bytes(ids: Sequence[int]) -> None:
    bad = [token for token in ids if token < 0 or token > 255]
    if bad:
        raise ValueError("skip horizon artifact requires token ids in 0..255")


def default_skip_horizon_cache_dir(count_prior_cache: Path | None) -> Path:
    if count_prior_cache is not None:
        return count_prior_cache.with_suffix(f"{count_prior_cache.suffix}.skip_horizon")
    return Path(os.environ.get("IMM_SKIP_HORIZON_CACHE", "/tmp/imm-skip-horizon-cache"))


def resolve_skip_horizon_paths(
    train_ids: Sequence[int],
    depth: int,
    horizon: int,
    cache_dir: Path,
    db_path: Path | None,
) -> tuple[Path, Path, Path]:
    digest = ids_digest(train_ids)[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    if db_path is None:
        db_path = cache_dir / f"train_{digest}_d{depth}_w{horizon}.sqlite"
    ids_path = cache_dir / f"train_{digest}.ids"
    meta_path = db_path.with_suffix(f"{db_path.suffix}.meta.json")
    return db_path, ids_path, meta_path


def resolve_skip_horizon_count_paths(
    train_ids: Sequence[int],
    depth: int,
    horizon: int,
    cache_dir: Path,
    db_path: Path | None,
) -> tuple[Path, Path, Path]:
    digest = ids_digest(train_ids)[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    if db_path is None:
        db_path = cache_dir / f"train_{digest}_d{depth}_w{horizon}_counts.sqlite"
    ids_path = cache_dir / f"train_{digest}.ids"
    meta_path = db_path.with_suffix(f"{db_path.suffix}.meta.json")
    return db_path, ids_path, meta_path


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_token_id_bytes(path: Path, ids: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytes(ids)
    if path.exists() and path.read_bytes() == data:
        return
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def run_crystal_builder(command: list[str], repo_root: Path) -> None:
    env = os.environ.copy()
    env.setdefault("CRYSTAL_CACHE_DIR", "/tmp/imm-crystal-cache")
    Path(env["CRYSTAL_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def ensure_skip_horizon_artifact(
    train_ids: Sequence[int],
    vocab_chars: Sequence[str],
    depth: int,
    horizon: int,
    repo_root: Path,
    cache_dir: Path,
    db_path: Path | None = None,
) -> SkipHorizonArtifact:
    if depth <= 0:
        raise ValueError("--skip-horizon-depth must be positive")
    if horizon <= 0:
        raise ValueError("--skip-horizon-max must be positive")
    if len(train_ids) < depth + horizon:
        raise ValueError("training ids must be at least skip horizon depth + horizon")
    ensure_ids_fit_bytes(train_ids)

    meta = skip_horizon_meta(train_ids, vocab_chars, depth, horizon)
    resolved_db_path, ids_path, meta_path = resolve_skip_horizon_paths(
        train_ids,
        depth,
        horizon,
        cache_dir,
        db_path,
    )
    existing_meta = load_json(meta_path)
    if resolved_db_path.exists() and existing_meta == meta:
        return SkipHorizonArtifact(
            db_path=resolved_db_path,
            ids_path=ids_path,
            meta_path=meta_path,
            meta=meta,
            cache_loaded=True,
        )
    if resolved_db_path.exists() and db_path is not None:
        raise ValueError(f"skip horizon DB metadata mismatch: {resolved_db_path}")

    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    write_token_id_bytes(ids_path, train_ids)
    run_crystal_builder(
        [
            "crystal",
            "run",
            "bin/imm_tree_build.cr",
            "--",
            "--input",
            str(ids_path),
            "--output",
            str(resolved_db_path),
            "--depth",
            str(depth),
            "--schema",
            "id",
            "--direction",
            "forward",
            "--force",
        ],
        repo_root,
    )
    run_crystal_builder(
        [
            "crystal",
            "run",
            "bin/imm_horizon_array_build.cr",
            "--",
            "--db",
            str(resolved_db_path),
            "--input",
            str(ids_path),
            "--horizon",
            str(horizon),
            "--direction",
            "forward",
            "--replace",
        ],
        repo_root,
    )
    tmp_meta = meta_path.with_suffix(f"{meta_path.suffix}.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    tmp_meta.replace(meta_path)
    return SkipHorizonArtifact(
        db_path=resolved_db_path,
        ids_path=ids_path,
        meta_path=meta_path,
        meta=meta,
        cache_loaded=False,
    )


def ensure_skip_horizon_count_artifact(
    train_ids: Sequence[int],
    vocab_chars: Sequence[str],
    depth: int,
    horizon: int,
    repo_root: Path,
    cache_dir: Path,
    db_path: Path | None = None,
) -> SkipHorizonArtifact:
    if depth <= 0:
        raise ValueError("--skip-horizon-depth must be positive")
    if horizon <= 0:
        raise ValueError("--skip-horizon-max must be positive")
    if len(train_ids) < depth + horizon:
        raise ValueError("training ids must be at least skip horizon depth + horizon")
    ensure_ids_fit_bytes(train_ids)

    meta = skip_horizon_count_meta(train_ids, vocab_chars, depth, horizon)
    resolved_db_path, ids_path, meta_path = resolve_skip_horizon_count_paths(
        train_ids,
        depth,
        horizon,
        cache_dir,
        db_path,
    )
    existing_meta = load_json(meta_path)
    if resolved_db_path.exists() and existing_meta == meta:
        return SkipHorizonArtifact(
            db_path=resolved_db_path,
            ids_path=ids_path,
            meta_path=meta_path,
            meta=meta,
            cache_loaded=True,
        )
    if resolved_db_path.exists() and db_path is not None:
        raise ValueError(f"skip horizon count DB metadata mismatch: {resolved_db_path}")

    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    write_token_id_bytes(ids_path, train_ids)
    run_crystal_builder(
        [
            "crystal",
            "run",
            "bin/imm_tree_build.cr",
            "--",
            "--input",
            str(ids_path),
            "--output",
            str(resolved_db_path),
            "--depth",
            str(depth),
            "--schema",
            "id",
            "--direction",
            "forward",
            "--force",
        ],
        repo_root,
    )
    run_crystal_builder(
        [
            "crystal",
            "run",
            "bin/imm_horizon_build.cr",
            "--",
            "--db",
            str(resolved_db_path),
            "--input",
            str(ids_path),
            "--horizon",
            str(horizon),
            "--direction",
            "forward",
            "--replace",
        ],
        repo_root,
    )
    tmp_meta = meta_path.with_suffix(f"{meta_path.suffix}.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    tmp_meta.replace(meta_path)
    return SkipHorizonArtifact(
        db_path=resolved_db_path,
        ids_path=ids_path,
        meta_path=meta_path,
        meta=meta,
        cache_loaded=False,
    )


def load_forward_node_map(conn: sqlite3.Connection) -> dict[bytes, int]:
    rows = conn.execute("SELECT key, id FROM nodes WHERE direction = 0")
    return {bytes(key): int(node_id) for key, node_id in rows}


def read_horizon_arrays(
    conn: sqlite3.Connection,
    node_id: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    row = conn.execute(
        "SELECT total_mass, entropy, max_mass, target_count FROM horizon_arrays WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    if row is None:
        return None
    total = np.frombuffer(row[0], dtype="<u4", count=horizon)
    entropy = np.frombuffer(row[1], dtype="<f4", count=horizon)
    max_mass = np.frombuffer(row[2], dtype="<u4", count=horizon)
    target_count = np.frombuffer(row[3], dtype="<u2", count=horizon)
    return total, entropy, max_mass, target_count


def deepest_node_id_by_end(
    ids: Sequence[int],
    depth: int,
    node_by_key: dict[bytes, int],
) -> list[int | None]:
    tokens = bytes(ids)
    node_ids: list[int | None] = []
    for end in range(len(ids)):
        max_depth = min(depth, end + 1)
        node_id = None
        for current_depth in range(max_depth, 0, -1):
            key = tokens[end - current_depth + 1 : end + 1]
            node_id = node_by_key.get(key)
            if node_id is not None:
                break
        node_ids.append(node_id)
    return node_ids


def compile_skip_horizon_features(
    ids: Sequence[int],
    db_path: Path,
    depth: int,
    horizon: int,
    vocab_size: int,
    train_length: int,
    output_path: Path | None = None,
) -> np.ndarray | np.memmap:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    ensure_ids_fit_bytes(ids)

    row_count = max(0, len(ids) - 1)
    shape = (row_count, len(FEATURE_NAMES))
    if output_path is None:
        rows: np.ndarray | np.memmap = np.zeros(shape, dtype=np.float32)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = np.memmap(output_path, dtype=np.float32, mode="w+", shape=shape)
        rows[:] = 0.0
    if row_count == 0:
        if isinstance(rows, np.memmap):
            rows.flush()
            return np.memmap(output_path, dtype=np.float32, mode="r", shape=shape)
        return rows

    available = np.zeros(row_count, dtype=np.float32)
    sum_sharpness = np.zeros(row_count, dtype=np.float32)
    best_score = np.full(row_count, -1.0, dtype=np.float32)
    best_h = np.zeros(row_count, dtype=np.float32)
    best_sharpness = np.zeros(row_count, dtype=np.float32)
    best_entropy_norm = np.zeros(row_count, dtype=np.float32)
    best_max_p = np.zeros(row_count, dtype=np.float32)
    best_log_mass_norm = np.zeros(row_count, dtype=np.float32)
    late_best_sharpness = np.zeros(row_count, dtype=np.float32)
    local_sharpness = np.zeros(row_count, dtype=np.float32)

    entropy_den = math.log(max(2, vocab_size))
    mass_den = math.log1p(max(1, train_length))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        node_by_key = load_forward_node_map(conn)
        node_ids = deepest_node_id_by_end(ids, depth, node_by_key)
        arrays_by_node: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = {}
        for end, node_id in enumerate(node_ids):
            if node_id is None:
                continue
            arrays = arrays_by_node.get(node_id)
            if arrays is None and node_id not in arrays_by_node:
                arrays = read_horizon_arrays(conn, node_id, horizon)
                arrays_by_node[node_id] = arrays
            if arrays is None:
                continue
            total_mass, entropy, max_mass, _target_count = arrays
            max_possible_h = min(horizon, row_count - end)
            for index in range(max_possible_h):
                total = int(total_mass[index])
                if total <= 0:
                    continue
                row = end + index
                h = index + 1
                entropy_norm = min(1.0, max(0.0, float(entropy[index]) / entropy_den))
                sharpness = 1.0 - entropy_norm
                max_p = min(1.0, max(0.0, float(max_mass[index]) / float(total)))
                log_mass_norm = min(1.0, math.log1p(total) / mass_den)
                score = sharpness * log_mass_norm

                available[row] += 1.0
                sum_sharpness[row] += sharpness
                if h == 1:
                    local_sharpness[row] = sharpness
                elif sharpness > late_best_sharpness[row]:
                    late_best_sharpness[row] = sharpness
                if score > best_score[row]:
                    best_score[row] = score
                    best_h[row] = float(h)
                    best_sharpness[row] = sharpness
                    best_entropy_norm[row] = entropy_norm
                    best_max_p[row] = max_p
                    best_log_mass_norm[row] = log_mass_norm
    finally:
        conn.close()

    rows[:, 0] = available / float(horizon)
    rows[:, 1] = best_h / float(horizon)
    rows[:, 2] = best_sharpness
    rows[:, 3] = best_entropy_norm
    rows[:, 4] = best_max_p
    rows[:, 5] = best_log_mass_norm
    rows[:, 6] = np.divide(
        sum_sharpness,
        available,
        out=np.zeros_like(sum_sharpness),
        where=available > 0,
    )
    rows[:, 7] = late_best_sharpness
    rows[:, 8] = local_sharpness

    if isinstance(rows, np.memmap):
        rows.flush()
        return np.memmap(output_path, dtype=np.float32, mode="r", shape=shape)
    return rows


def load_horizon_count_distributions(
    conn: sqlite3.Connection,
    vocab_size: int,
) -> dict[tuple[int, int], tuple[np.ndarray, int, float]]:
    distributions: dict[tuple[int, int], tuple[np.ndarray, int, float]] = {}
    cursor = conn.execute(
        "SELECT node_id, horizon, target, mass FROM horizon_counts ORDER BY node_id, horizon, target"
    )
    current_key: tuple[int, int] | None = None
    counts = np.zeros((vocab_size,), dtype=np.float32)
    total = 0

    def flush() -> None:
        nonlocal counts, total, current_key
        if current_key is None or total <= 0:
            return
        probs = counts / float(total)
        entropy = float(-(probs[probs > 0.0] * np.log(probs[probs > 0.0])).sum())
        distributions[current_key] = (probs.astype(np.float32, copy=True), total, entropy)

    for node_id_raw, horizon_raw, target_raw, mass_raw in cursor:
        key = (int(node_id_raw), int(horizon_raw))
        if current_key is not None and key != current_key:
            flush()
            counts.fill(0.0)
            total = 0
        current_key = key
        target = int(target_raw)
        if 0 <= target < vocab_size:
            mass = int(mass_raw)
            counts[target] += float(mass)
            total += mass
    flush()
    return distributions


def compile_skip_horizon_distribution_rows(
    ids: Sequence[int],
    db_path: Path,
    depth: int,
    horizon: int,
    vocab_size: int,
    train_length: int,
    log_output_path: Path | None = None,
    strength_output_path: Path | None = None,
    dtype: np.dtype = np.dtype("float32"),
) -> tuple[np.ndarray | np.memmap, np.ndarray | np.memmap]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    ensure_ids_fit_bytes(ids)

    row_count = max(0, len(ids) - 1)
    log_shape = (row_count, vocab_size)
    strength_shape = (row_count, 1)
    if log_output_path is None:
        log_rows: np.ndarray | np.memmap = np.zeros(log_shape, dtype=dtype)
    else:
        log_output_path.parent.mkdir(parents=True, exist_ok=True)
        log_rows = np.memmap(log_output_path, dtype=dtype, mode="w+", shape=log_shape)
    if strength_output_path is None:
        strength_rows: np.ndarray | np.memmap = np.zeros(strength_shape, dtype=np.float32)
    else:
        strength_output_path.parent.mkdir(parents=True, exist_ok=True)
        strength_rows = np.memmap(strength_output_path, dtype=np.float32, mode="w+", shape=strength_shape)
    if row_count == 0:
        if isinstance(log_rows, np.memmap):
            log_rows.flush()
            strength_rows.flush()
            return (
                np.memmap(log_output_path, dtype=dtype, mode="r", shape=log_shape),
                np.memmap(strength_output_path, dtype=np.float32, mode="r", shape=strength_shape),
            )
        return log_rows, strength_rows

    accum = np.zeros((row_count, vocab_size), dtype=np.float32)
    weight_sum = np.zeros((row_count,), dtype=np.float32)
    entropy_den = math.log(max(2, vocab_size))
    mass_den = math.log1p(max(1, train_length))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        node_by_key = load_forward_node_map(conn)
        node_ids = deepest_node_id_by_end(ids, depth, node_by_key)
        distributions = load_horizon_count_distributions(conn, vocab_size)
        for end, node_id in enumerate(node_ids):
            if node_id is None:
                continue
            max_possible_h = min(horizon, row_count - end)
            for h in range(1, max_possible_h + 1):
                item = distributions.get((node_id, h))
                if item is None:
                    continue
                probs, total, entropy = item
                entropy_norm = min(1.0, max(0.0, entropy / entropy_den))
                sharpness = 1.0 - entropy_norm
                log_mass_norm = min(1.0, math.log1p(total) / mass_den)
                weight = sharpness * log_mass_norm
                if weight <= 0.0:
                    continue
                row = end + h - 1
                accum[row] += probs * weight
                weight_sum[row] += weight
    finally:
        conn.close()

    uniform = np.full((vocab_size,), 1.0 / float(vocab_size), dtype=np.float32)
    for start in range(0, row_count, 100_000):
        end = min(row_count, start + 100_000)
        chunk_weights = weight_sum[start:end]
        probs = np.divide(
            accum[start:end],
            chunk_weights[:, None],
            out=np.broadcast_to(uniform, (end - start, vocab_size)).copy(),
            where=chunk_weights[:, None] > 0.0,
        )
        log_rows[start:end] = np.log(np.maximum(probs, COUNT_PRIOR_EPS)).astype(dtype, copy=False)
        strength_rows[start:end, 0] = np.minimum(1.0, chunk_weights / float(horizon))

    if isinstance(log_rows, np.memmap):
        log_rows.flush()
        strength_rows.flush()
        return (
            np.memmap(log_output_path, dtype=dtype, mode="r", shape=log_shape),
            np.memmap(strength_output_path, dtype=np.float32, mode="r", shape=strength_shape),
        )
    return log_rows, strength_rows


def compile_skip_horizon_channel_rows(
    ids: Sequence[int],
    db_path: Path,
    depth: int,
    horizon: int,
    vocab_size: int,
    train_length: int,
    log_output_path: Path | None = None,
    strength_output_path: Path | None = None,
    dtype: np.dtype = np.dtype("float32"),
) -> tuple[np.ndarray | np.memmap, np.ndarray | np.memmap]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    ensure_ids_fit_bytes(ids)

    row_count = max(0, len(ids) - 1)
    log_shape = (row_count, horizon, vocab_size)
    strength_shape = (row_count, horizon)
    if log_output_path is None:
        log_rows: np.ndarray | np.memmap = np.zeros(log_shape, dtype=dtype)
    else:
        log_output_path.parent.mkdir(parents=True, exist_ok=True)
        log_rows = np.memmap(log_output_path, dtype=dtype, mode="w+", shape=log_shape)
    if strength_output_path is None:
        strength_rows: np.ndarray | np.memmap = np.zeros(strength_shape, dtype=np.float32)
    else:
        strength_output_path.parent.mkdir(parents=True, exist_ok=True)
        strength_rows = np.memmap(strength_output_path, dtype=np.float32, mode="w+", shape=strength_shape)
    if row_count == 0:
        if isinstance(log_rows, np.memmap):
            log_rows.flush()
            strength_rows.flush()
            return (
                np.memmap(log_output_path, dtype=dtype, mode="r", shape=log_shape),
                np.memmap(strength_output_path, dtype=np.float32, mode="r", shape=strength_shape),
            )
        return log_rows, strength_rows

    uniform_log = math.log(1.0 / float(vocab_size))
    log_rows[:] = uniform_log
    strength_rows[:] = 0.0
    entropy_den = math.log(max(2, vocab_size))
    mass_den = math.log1p(max(1, train_length))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        node_by_key = load_forward_node_map(conn)
        node_ids = deepest_node_id_by_end(ids, depth, node_by_key)
        distributions = load_horizon_count_distributions(conn, vocab_size)
        for end, node_id in enumerate(node_ids):
            if node_id is None:
                continue
            max_possible_h = min(horizon, row_count - end)
            for h in range(1, max_possible_h + 1):
                item = distributions.get((node_id, h))
                if item is None:
                    continue
                probs, total, entropy = item
                entropy_norm = min(1.0, max(0.0, entropy / entropy_den))
                sharpness = 1.0 - entropy_norm
                log_mass_norm = min(1.0, math.log1p(total) / mass_den)
                strength = sharpness * log_mass_norm
                if strength <= 0.0:
                    continue
                row = end + h - 1
                h_index = h - 1
                log_rows[row, h_index] = np.log(np.maximum(probs, COUNT_PRIOR_EPS)).astype(dtype, copy=False)
                strength_rows[row, h_index] = strength
    finally:
        conn.close()

    if isinstance(log_rows, np.memmap):
        log_rows.flush()
        strength_rows.flush()
        return (
            np.memmap(log_output_path, dtype=dtype, mode="r", shape=log_shape),
            np.memmap(strength_output_path, dtype=np.float32, mode="r", shape=strength_shape),
        )
    return log_rows, strength_rows


def blend_log_prob_rows(
    base_log_probs: np.ndarray | np.memmap,
    skip_log_probs: np.ndarray | np.memmap,
    skip_strength: np.ndarray | np.memmap,
    skip_weight: float,
    output_path: Path,
    dtype: np.dtype,
    chunk_size: int = 100_000,
) -> np.memmap:
    if base_log_probs.shape != skip_log_probs.shape:
        raise ValueError(f"log-prob shape mismatch: base={base_log_probs.shape} skip={skip_log_probs.shape}")
    if skip_strength.shape[0] != base_log_probs.shape[0]:
        raise ValueError("skip strength rows do not match log-prob rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(output_path, dtype=dtype, mode="w+", shape=base_log_probs.shape)
    row_count = base_log_probs.shape[0]
    for start in range(0, row_count, chunk_size):
        end = min(row_count, start + chunk_size)
        alpha = np.asarray(skip_strength[start:end], dtype=np.float32) * float(skip_weight)
        alpha = np.clip(alpha, 0.0, 1.0)
        base_probs = np.exp(np.asarray(base_log_probs[start:end], dtype=np.float32))
        skip_probs = np.exp(np.asarray(skip_log_probs[start:end], dtype=np.float32))
        probs = (1.0 - alpha) * base_probs + alpha * skip_probs
        out[start:end] = np.log(np.maximum(probs, COUNT_PRIOR_EPS)).astype(dtype, copy=False)
    out.flush()
    del out
    return np.memmap(output_path, dtype=dtype, mode="r", shape=base_log_probs.shape)
