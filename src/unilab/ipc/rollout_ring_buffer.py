"""Shared rollout IPC ring buffer for APPO / async PPO."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")

_FIELD_SHAPES = {
    "obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, od),
    "critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, cd),
    "actions": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, ad),
    "log_probs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "rewards": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "dones": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "truncated": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "last_obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, od),
    "last_critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, cd),
}

FADA_PPO_PROTOCOL = "fada_ppo_v1"
_PPO_FIELDS = {
    "values": lambda s, n, t, o, a, c: (s, n, t),
    "action_mean": lambda s, n, t, o, a, c: (s, n, t, a),
    "action_std": lambda s, n, t, o, a, c: (s, n, t, a),
    "final_critic": lambda s, n, t, o, a, c: (s, n, t, c),
    "policy_version": lambda s, n, t, o, a, c: (s, 1),
    "valid_steps": lambda s, n, t, o, a, c: (s, 1),
}


def rollout_field_specs(num_slots, num_envs, num_steps, obs_dim, action_dim, critic_dim=0, *, protocol="appo"):
    """One source of truth for allocation, attachment and memory estimation."""
    if protocol not in ("appo", FADA_PPO_PROTOCOL):
        raise ValueError(f"Unknown rollout protocol: {protocol}")
    fields = dict(_FIELD_SHAPES)
    if protocol == FADA_PPO_PROTOCOL:
        if min(num_slots, num_envs, num_steps, obs_dim, action_dim, critic_dim) < 1:
            raise ValueError("FADA PPO rollout dimensions must all be positive")
        fields.update(_PPO_FIELDS)
    if critic_dim == 0:
        fields.pop("critic", None)
        fields.pop("last_critic", None)
    return {name: (shape(num_slots, num_envs, num_steps, obs_dim, action_dim, critic_dim),
                   np.dtype(np.int64 if name in ("policy_version", "valid_steps") else np.float32))
            for name, shape in fields.items()}


class RolloutRingBuffer:
    """N-slot shared-memory ring buffer for raw rollout payloads."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        critic_dim: int = 0,
        num_slots: int = 4,
        create: bool = True,
        shm_name_prefix: Dict[str, str] | None = None,
        protocol: str = "appo",
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_dim = critic_dim
        self.num_slots = num_slots
        self.protocol = protocol

        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}

        specs = rollout_field_specs(num_slots, num_envs, num_steps, obs_dim, action_dim,
                                    critic_dim, protocol=protocol)
        if not create and (shm_name_prefix is None or specs.keys() != shm_name_prefix.keys()):
            raise ValueError("Rollout attachment fields/protocol mismatch")
        try:
            for field, (shape, dtype) in specs.items():
                nbytes = int(np.prod(shape)) * dtype.itemsize
                shm = (shared_memory.SharedMemory(create=True, size=max(nbytes, 1)) if create else
                       shared_memory.SharedMemory(name=shm_name_prefix[field], create=False))
                self._shm_blocks[field] = shm
                # macOS rounds POSIX shared-memory allocations up to pages.
                # Field identities and logical tensor shapes come from the
                # same spawn descriptor; the mapping must have enough capacity.
                if shm.size < max(nbytes, 1):
                    raise ValueError(f"Rollout attachment byte size mismatch: {field}")
                self._arrays[field] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                if create and field in ("policy_version", "valid_steps"):
                    self._arrays[field].fill(0)
            if create:
                self._write_ptr = _SPAWN_CTX.Value("l", 0)
                self._read_ptr = _SPAWN_CTX.Value("l", 0)
        except BaseException:
            self.cleanup() if create else self.close()
            raise

    @property
    def name(self) -> Dict[str, str]:
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    @property
    def slot_shapes(self) -> Dict[str, tuple[int, ...]]:
        return {field: tuple(arr.shape[1:]) for field, arr in self._arrays.items()}

    @property
    def sync_primitives(self):
        return self._write_ptr, self._read_ptr

    def _strict_counts(self):
        with self._write_ptr.get_lock(), self._read_ptr.get_lock():
            wp, rp = int(self._write_ptr.value), int(self._read_ptr.value)
            if not 0 <= wp - rp <= self.num_slots:
                raise RuntimeError("Strict rollout pointer window violated; no samples may be skipped")
            return wp, rp

    def attach_sync_primitives(self, write_ptr, read_ptr) -> None:
        self._write_ptr = write_ptr
        self._read_ptr = read_ptr

    def _clamp_read_ptr_to_valid_window(self) -> None:
        if self.protocol == FADA_PPO_PROTOCOL:
            self._strict_counts()
            return
        wp = int(self._write_ptr.value)
        oldest_available = max(0, wp - self.num_slots)
        if int(self._read_ptr.value) >= oldest_available:
            return
        with self._read_ptr.get_lock():
            if int(self._read_ptr.value) < oldest_available:
                self._read_ptr.value = oldest_available

    @property
    def write_slot(self) -> int:
        return int(self._write_ptr.value) % self.num_slots

    @property
    def write_buffer(self) -> Dict[str, np.ndarray]:
        if self.protocol == FADA_PPO_PROTOCOL and self.available() == self.num_slots:
            raise BufferError("Strict rollout buffer is full")
        s = self.write_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def signal_write_done(self) -> None:
        if self.protocol == FADA_PPO_PROTOCOL:
            with self._write_ptr.get_lock(), self._read_ptr.get_lock():
                wp, rp = self._strict_counts()
                if wp - rp == self.num_slots:
                    raise BufferError("Strict rollout buffer is full")
                if self._arrays["valid_steps"][wp % self.num_slots, 0] != self.num_steps:
                    raise ValueError("Cannot publish an incomplete PPO rollout")
                self._write_ptr.value += 1
            return
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    def available(self) -> int:
        if self.protocol == FADA_PPO_PROTOCOL:
            wp, rp = self._strict_counts()
            return wp - rp
        self._clamp_read_ptr_to_valid_window()
        return min(max(0, int(self._write_ptr.value) - int(self._read_ptr.value)), self.num_slots)

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    @property
    def read_slot(self) -> int:
        if self.protocol == FADA_PPO_PROTOCOL and self.available() == 0:
            raise BufferError("Strict rollout buffer is empty")
        self._clamp_read_ptr_to_valid_window()
        return int(self._read_ptr.value) % self.num_slots

    def read_numpy_views(self) -> dict[str, np.ndarray]:
        """Return shared-memory views for the current read slot.

        The returned arrays are borrowed views. Consumers must copy them into
        owned storage before calling advance_read().
        """
        s = self.read_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def copy_read_slot_to_torch(self, destination: dict) -> None:
        import torch

        s = self.read_slot
        for field, arr in self._arrays.items():
            if field not in destination:
                raise KeyError(f"missing destination tensor for rollout field {field!r}")
            dst = destination[field]
            src_view = arr[s]
            if tuple(dst.shape) != tuple(src_view.shape):
                raise ValueError(
                    f"destination shape mismatch for {field!r}: "
                    f"expected {tuple(src_view.shape)}, got {tuple(dst.shape)}"
                )
            expected_dtype = torch.int64 if arr.dtype == np.int64 else torch.float32
            if dst.dtype != expected_dtype:
                raise TypeError(f"destination tensor for {field!r} must be {expected_dtype}")
            dst.copy_(torch.from_numpy(src_view), non_blocking=False)

    def read_torch(self, device: str) -> dict:
        import torch

        result = {
            field: torch.empty(tuple(arr.shape[1:]), dtype=torch.int64 if arr.dtype == np.int64 else torch.float32, device=device)
            for field, arr in self._arrays.items()
        }
        self.copy_read_slot_to_torch(result)
        return result

    def advance_read(self) -> None:
        if self.protocol == FADA_PPO_PROTOCOL:
            with self._write_ptr.get_lock(), self._read_ptr.get_lock():
                wp, rp = self._strict_counts()
                if wp == rp:
                    raise BufferError("Strict rollout buffer is empty")
                self._read_ptr.value += 1
            return
        with self._read_ptr.get_lock():
            wp = int(self._write_ptr.value)
            rp = min(int(self._read_ptr.value) + 1, wp)
            oldest_available = max(0, wp - self.num_slots)
            self._read_ptr.value = max(rp, oldest_available)

    def cleanup(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def close(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass
