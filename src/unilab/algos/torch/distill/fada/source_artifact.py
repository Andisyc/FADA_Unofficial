"""FADA causal-window artifact persistence owner."""

from __future__ import annotations

import shutil
import uuid
import weakref
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADASourceBatch
from unilab.algos.torch.fada_official.checkpoint import validate_student_binding

FADA_SOURCE_BATCH_SCHEMA_VERSION = 4
FADA_SHARDED_SOURCE_SCHEMA_VERSION = 5
_FADA_SOURCE_SHARD_SCHEMA_VERSION = 1


def _cleanup_uncommitted_artifact(shard_dir: Path, manifest_temporary: Path) -> None:
    shutil.rmtree(shard_dir, ignore_errors=True)
    manifest_temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedFADASourceBatch:
    """One materialized collector artifact and its iteration metadata."""

    batch: FADASourceBatch
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _FADASourceShard:
    path: Path
    rows: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class LoadedFADASourceArtifact:
    """Validated artifact metadata whose tensor shards are loaded on demand."""

    config: FADAArchitectureConfig
    metadata: Mapping[str, Any]
    num_samples: int
    shards: tuple[_FADASourceShard, ...]
    legacy_batch: FADASourceBatch | None = None
    _validated_planner_eligible: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def iter_batches(self) -> Iterator[FADASourceBatch]:
        for index in range(self.batch_count):
            yield self.load_batch(index)

    @property
    def batch_count(self) -> int:
        return 1 if self.legacy_batch is not None else len(self.shards)

    def load_batch(self, index: int) -> FADASourceBatch:
        if self.legacy_batch is not None:
            if index != 0:
                raise IndexError(index)
            return self.legacy_batch
        return _load_source_shard(self.shards[index], config=self.config)

    def planner_role_vectors(self) -> tuple[torch.Tensor, ...]:
        roles = self._validated_planner_eligible
        if roles is None:
            raise RuntimeError("FADA artifact row identity must be validated before replay admission")
        sizes = (
            (int(self.legacy_batch.command.shape[0]),)
            if self.legacy_batch is not None
            else tuple(shard.rows for shard in self.shards)
        )
        return tuple(torch.split(roles, sizes))

    def identity_fields(self) -> dict[str, torch.Tensor]:
        names = (
            "command",
            "oracle_shadow_valid",
            "idm_source_role",
            "command_scenario",
            "planner_eligible",
            "cold_start",
        )
        fields: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        for batch in self.iter_batches():
            for name in names:
                fields[name].append(getattr(batch, name).clone())
        identity = {name: torch.cat(values) for name, values in fields.items()}
        object.__setattr__(self, "_validated_planner_eligible", identity["planner_eligible"])
        return identity

    def select_indices(self, indices: torch.Tensor) -> FADASourceBatch:
        if indices.ndim != 1 or indices.dtype != torch.int64 or indices.numel() == 0:
            raise ValueError("FADA artifact selection indices must be non-empty rank-1 int64")
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_samples:
            raise IndexError("FADA artifact selection index is out of range")
        outputs: dict[str, torch.Tensor] | None = None
        offset = 0
        for index in range(self.batch_count):
            rows = (
                int(self.legacy_batch.command.shape[0])
                if self.legacy_batch is not None
                else self.shards[index].rows
            )
            positions = torch.nonzero(
                (indices >= offset) & (indices < offset + rows), as_tuple=False
            ).flatten()
            if positions.numel() > 0:
                batch = self.load_batch(index)
                local = indices.index_select(0, positions) - offset
                if outputs is None:
                    outputs = {
                        field: torch.empty(
                            (indices.numel(), *getattr(batch, field).shape[1:]),
                            dtype=getattr(batch, field).dtype,
                        )
                        for field in FADASourceBatch.__dataclass_fields__
                    }
                for field in FADASourceBatch.__dataclass_fields__:
                    outputs[field].index_copy_(
                        0,
                        positions,
                        getattr(batch, field).index_select(0, local),
                    )
            offset += rows
        if outputs is None:
            raise RuntimeError("FADA artifact selection produced no rows")
        return FADASourceBatch(**outputs).validate(self.config)


def _load_architecture_config(
    architecture: Any,
    *,
    schema_version: Any,
    contract_schema_version: Any,
    context: str,
) -> FADAArchitectureConfig:
    if not isinstance(architecture, dict):
        raise ValueError(f"{context} architecture must be a mapping")
    if schema_version == contract_schema_version and "observation_contract" not in architecture:
        raise ValueError(f"{context} architecture must contain observation_contract")
    try:
        return FADAArchitectureConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {context} architecture: {architecture}") from exc


def _batch_to_device(batch: FADASourceBatch, device: torch.device) -> FADASourceBatch:
    return FADASourceBatch(
        **{
            field: getattr(batch, field).to(device)
            for field in FADASourceBatch.__dataclass_fields__
        }
    )


def _batch_payload(batch: FADASourceBatch) -> dict[str, torch.Tensor]:
    return {field: getattr(batch, field) for field in FADASourceBatch.__dataclass_fields__}


def _batch_from_payload(tensors: Any, *, context: str) -> FADASourceBatch:
    if not isinstance(tensors, dict) or set(tensors) != set(FADASourceBatch.__dataclass_fields__):
        raise ValueError(f"{context} tensor fields are incomplete")
    return FADASourceBatch(**tensors)


def _validated_cpu_batch(
    batch: FADASourceBatch,
    *,
    config: FADAArchitectureConfig,
) -> FADASourceBatch:
    return _batch_to_device(batch.validate(config), torch.device("cpu"))


load_architecture_config = _load_architecture_config
batch_to_device = _batch_to_device


def save_fada_source_batch(
    path: str | Path,
    batch: FADASourceBatch,
    *,
    config: FADAArchitectureConfig,
    metadata: Mapping[str, Any],
) -> Path:
    """Persist the legacy v4 monolith for offline and compatibility callers."""

    validate_student_binding(config, metadata)
    validated = _validated_cpu_batch(batch, config=config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": FADA_SOURCE_BATCH_SCHEMA_VERSION,
            "architecture": asdict(config),
            "batch": _batch_payload(validated),
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(target)
    return target


class FADAShardedSourceWriter:
    """Write one async artifact without retaining or concatenating its batches."""

    def __init__(
        self,
        path: str | Path,
        *,
        config: FADAArchitectureConfig,
        replace_existing: bool = False,
    ) -> None:
        self.target = Path(path)
        self.config = config
        self.replace_existing = bool(replace_existing)
        self._token = uuid.uuid4().hex
        self._shard_dir = self.target.parent / f"{self.target.name}.shards-{self._token}"
        self._manifest_temporary = self.target.with_suffix(
            self.target.suffix + f".tmp-{self._token}"
        )
        self._shards: list[_FADASourceShard] = []
        self._replaced_shard_dirs: tuple[Path, ...] = ()
        self._committed = False
        self._cleanup: weakref.finalize | None = None

    @property
    def num_samples(self) -> int:
        return sum(shard.rows for shard in self._shards)

    def __enter__(self) -> FADAShardedSourceWriter:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.target.exists() and not self.replace_existing:
            raise FileExistsError(
                f"refusing to overwrite existing FADA source artifact: {self.target}"
            )
        if self.replace_existing:
            self._replaced_shard_dirs = tuple(
                candidate
                for candidate in self.target.parent.glob(f"{self.target.name}.shards-*")
                if candidate.is_dir() and candidate != self._shard_dir
            )
        self._shard_dir.mkdir()
        self._cleanup = weakref.finalize(
            self,
            _cleanup_uncommitted_artifact,
            self._shard_dir,
            self._manifest_temporary,
        )
        return self

    def append(self, batch: FADASourceBatch) -> None:
        if self._committed:
            raise RuntimeError("cannot append to a committed FADA source artifact")
        validated = _validated_cpu_batch(batch, config=self.config)
        rows = int(validated.command.shape[0])
        if rows <= 0:
            raise ValueError("FADA source artifact shards must contain at least one row")
        shard_path = self._shard_dir / f"shard_{len(self._shards):04d}.pt"
        temporary = shard_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema_version": _FADA_SOURCE_SHARD_SCHEMA_VERSION,
                "rows": rows,
                "batch": _batch_payload(validated),
            },
            temporary,
        )
        temporary.replace(shard_path)
        self._shards.append(
            _FADASourceShard(
                path=shard_path,
                rows=rows,
                size_bytes=shard_path.stat().st_size,
                sha256=_file_sha256(shard_path),
            )
        )

    def commit(self, *, metadata: Mapping[str, Any]) -> Path:
        validate_student_binding(self.config, metadata)
        if self._committed:
            raise RuntimeError("FADA source artifact was already committed")
        if not self._shards:
            raise ValueError("FADA source artifact must contain at least one shard")
        if self.target.exists() and not self.replace_existing:
            raise FileExistsError(
                f"refusing to overwrite existing FADA source artifact: {self.target}"
            )
        torch.save(
            {
                "schema_version": FADA_SHARDED_SOURCE_SCHEMA_VERSION,
                "architecture": asdict(self.config),
                "num_samples": self.num_samples,
                "shards": [
                    {
                        "path": str(shard.path.relative_to(self.target.parent)),
                        "rows": shard.rows,
                        "size_bytes": shard.size_bytes,
                        "sha256": shard.sha256,
                    }
                    for shard in self._shards
                ],
                "metadata": dict(metadata),
            },
            self._manifest_temporary,
        )
        self._manifest_temporary.replace(self.target)
        self._committed = True
        if self._cleanup is not None:
            self._cleanup.detach()
        for replaced in self._replaced_shard_dirs:
            shutil.rmtree(replaced, ignore_errors=True)
        return self.target

    def close(self) -> None:
        if not self._committed and self._cleanup is not None and self._cleanup.alive:
            self._cleanup()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _resolve_shard(path: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise ValueError("FADA source artifact shard path must be a string")
    candidate = Path(relative)
    expected_prefix = f"{path.name}.shards-"
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or len(candidate.parts) != 2
        or not candidate.parts[0].startswith(expected_prefix)
        or not candidate.parts[1].startswith("shard_")
        or candidate.parts[1] != Path(candidate.parts[1]).name
    ):
        raise ValueError(f"unsafe FADA source artifact shard path: {relative!r}")
    resolved = (path.parent / candidate).resolve()
    if path.parent.resolve() not in resolved.parents:
        raise ValueError(f"unsafe FADA source artifact shard path: {relative!r}")
    return resolved


def _load_source_shard(
    shard: _FADASourceShard,
    *,
    config: FADAArchitectureConfig,
) -> FADASourceBatch:
    if shard.path.stat().st_size != shard.size_bytes or _file_sha256(shard.path) != shard.sha256:
        raise ValueError(f"FADA source shard content identity mismatch: {shard.path}")
    payload = torch.load(shard.path, map_location="cpu", weights_only=True, mmap=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _FADA_SOURCE_SHARD_SCHEMA_VERSION
        or payload.get("rows") != shard.rows
    ):
        raise ValueError(f"unsupported or malformed FADA source shard: {shard.path}")
    batch = _batch_from_payload(payload.get("batch"), context="FADA source shard").validate(config)
    if int(batch.command.shape[0]) != shard.rows:
        raise ValueError(
            f"FADA source shard row mismatch: manifest={shard.rows} "
            f"observed={batch.command.shape[0]} path={shard.path}"
        )
    return batch


def open_fada_source_artifact(
    path: str | Path,
    *,
    config: FADAArchitectureConfig,
) -> LoadedFADASourceArtifact:
    """Open v5 lazily, while admitting existing v4 monolithic artifacts."""

    target = Path(path)
    payload = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("unsupported or malformed FADA source batch schema")
    schema_version = payload.get("schema_version")
    if schema_version not in {FADA_SOURCE_BATCH_SCHEMA_VERSION, FADA_SHARDED_SOURCE_SCHEMA_VERSION}:
        raise ValueError("unsupported or malformed FADA source batch schema")
    observed = _load_architecture_config(
        payload.get("architecture"),
        schema_version=schema_version,
        contract_schema_version=schema_version,
        context="FADA source artifact",
    )
    if observed != config:
        raise ValueError(
            f"FADA source artifact architecture mismatch: expected={config} observed={observed}"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("FADA source artifact metadata must be a mapping")
    validate_student_binding(config, metadata)
    if schema_version == FADA_SOURCE_BATCH_SCHEMA_VERSION:
        batch = _batch_from_payload(payload.get("batch"), context="FADA source batch").validate(config)
        return LoadedFADASourceArtifact(
            config=config,
            metadata=metadata,
            num_samples=int(batch.command.shape[0]),
            shards=(),
            legacy_batch=batch,
        )
    entries = payload.get("shards")
    num_samples = payload.get("num_samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError("FADA source artifact shards must be a non-empty list")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("FADA source artifact num_samples must be positive")
    shards: list[_FADASourceShard] = []
    seen: set[Path] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("rows"), int)
            or not isinstance(entry.get("size_bytes"), int)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise ValueError("FADA source artifact shard entries are malformed")
        shard = _FADASourceShard(
            path=_resolve_shard(target, entry.get("path")),
            rows=int(entry["rows"]),
            size_bytes=int(entry["size_bytes"]),
            sha256=str(entry["sha256"]),
        )
        if (
            shard.rows <= 0
            or shard.size_bytes <= 0
            or len(shard.sha256) != 64
            or shard.path in seen
        ):
            raise ValueError("FADA source artifact shard entries must be unique and non-empty")
        if not shard.path.is_file():
            raise FileNotFoundError(f"FADA source artifact shard does not exist: {shard.path}")
        seen.add(shard.path)
        shards.append(shard)
    if sum(shard.rows for shard in shards) != num_samples:
        raise ValueError("FADA source artifact manifest row count does not match its shards")
    return LoadedFADASourceArtifact(
        config=config,
        metadata=metadata,
        num_samples=num_samples,
        shards=tuple(shards),
    )


def load_fada_source_batch(
    path: str | Path,
    *,
    config: FADAArchitectureConfig,
) -> LoadedFADASourceBatch:
    """Materialize v4 or v5 for compatibility-only offline consumers."""

    artifact = open_fada_source_artifact(path, config=config)
    batches = tuple(artifact.iter_batches())
    if len(batches) == 1:
        batch = batches[0]
    else:
        batch = FADASourceBatch(
            **{
                field: torch.cat([getattr(item, field) for item in batches], dim=0)
                for field in FADASourceBatch.__dataclass_fields__
            }
        ).validate(config)
    return LoadedFADASourceBatch(batch=batch, metadata=artifact.metadata)


def validate_fada_async_artifact_identity(
    metadata: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    identity_fields = (
        "request_id",
        "scenario",
        "iteration",
        "checkpoint_path",
        "expected_weight_version",
        "producer_pid",
    )
    missing = [name for name in identity_fields if name not in metadata or name not in expected]
    if missing:
        raise ValueError(f"FADA async artifact identity is incomplete: {missing}")
    mismatches = [name for name in identity_fields if metadata.get(name) != expected.get(name)]
    if "official_oracle_identity" in expected:
        if metadata.get("official_oracle_identity") != expected.get("official_oracle_identity"):
            mismatches.append("official_oracle_identity")
    if mismatches:
        raise ValueError(f"FADA async artifact identity mismatch: {mismatches}")
