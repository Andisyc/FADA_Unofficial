from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_ACTION_DIM,
    FADA_G1_COMMAND_DIM,
    FADA_G1_STATE_DIM,
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    FADA_LEGACY_OBSERVATION_CONTRACT,
    FADA_OFFICIAL_STUDENT_CONTRACT,
)

FADA_COMMAND_SCENARIOS = ("walk", "static_stand", "walk_to_stand")
FADA_SCENARIO_IDS = {name: index for index, name in enumerate(FADA_COMMAND_SCENARIOS)}
FADA_IDM_SOURCE_ROLES = ("trajectory", "oracle_shadow")
FADA_IDM_SOURCE_ROLE_IDS = {name: index for index, name in enumerate(FADA_IDM_SOURCE_ROLES)}


@dataclass(frozen=True)
class FADAArchitectureConfig:
    """Paper-defined Planner-IDM dimensions with task-owned feature sizes."""

    obs_dim: int
    action_dim: int
    command_dim: int
    observation_contract: str = FADA_LEGACY_OBSERVATION_CONTRACT
    history_length: int = 30
    prediction_horizon: int = 6
    hidden_dim: int = 128
    num_heads: int = 4
    planner_layers: int = 3
    idm_encoder_layers: int = 3
    idm_decoder_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "command_dim": self.command_dim,
            "history_length": self.history_length,
            "prediction_horizon": self.prediction_horizon,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "planner_layers": self.planner_layers,
            "idm_encoder_layers": self.idm_encoder_layers,
            "idm_decoder_layers": self.idm_decoder_layers,
            "feedforward_dim": self.feedforward_dim,
        }
        for name, value in integer_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.observation_contract not in {
            FADA_LEGACY_OBSERVATION_CONTRACT,
            FADA_G1_STATE_OBSERVATION_CONTRACT,
            FADA_OFFICIAL_STUDENT_CONTRACT,
        }:
            raise ValueError(
                f"unsupported FADA observation_contract: {self.observation_contract!r}"
            )
        dimensions = (int(self.obs_dim), int(self.action_dim), int(self.command_dim))
        g1_v2_dimensions = (FADA_G1_STATE_DIM, FADA_G1_ACTION_DIM, FADA_G1_COMMAND_DIM)
        if self.observation_contract in {FADA_G1_STATE_OBSERVATION_CONTRACT, FADA_OFFICIAL_STUDENT_CONTRACT}:
            if dimensions != g1_v2_dimensions:
                raise ValueError(
                    "g1_fada_state_v2 requires obs/action/command dimensions "
                    f"{g1_v2_dimensions}, got {dimensions}"
                )
        elif dimensions == g1_v2_dimensions:
            raise ValueError("the 66/29/3 FADA architecture tuple is reserved for g1_fada_state_v2")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={self.hidden_dim} num_heads={self.num_heads}"
            )
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class PlannerIDMOutput:
    """One receding-horizon Planner-IDM query."""

    predicted_future: torch.Tensor
    action_chunk: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True)
class FADASourceBatch:
    """Source window with separate causal-IDM and oracle-Planner targets."""

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    realized_future: torch.Tensor
    executed_action_chunk: torch.Tensor
    oracle_future: torch.Tensor
    oracle_action_chunk: torch.Tensor
    oracle_shadow_valid: torch.Tensor
    idm_source_role: torch.Tensor
    oracle_first_action: torch.Tensor
    command_scenario: torch.Tensor
    planner_eligible: torch.Tensor
    cold_start: torch.Tensor

    def validate(self, config: FADAArchitectureConfig) -> FADASourceBatch:
        # B1: 分别校验 history, causal future/action pair 与 oracle label 的 shape 和有限性.
        _validate_sequence(
            "observation_history",
            self.observation_history,
            length=config.history_length,
            feature_dim=config.obs_dim,
        )
        _validate_sequence(
            "action_history",
            self.action_history,
            length=config.history_length,
            feature_dim=config.action_dim,
        )
        _validate_sequence(
            "realized_future",
            self.realized_future,
            length=config.prediction_horizon,
            feature_dim=config.obs_dim,
        )
        _validate_sequence(
            "executed_action_chunk",
            self.executed_action_chunk,
            length=config.prediction_horizon,
            feature_dim=config.action_dim,
        )
        _validate_sequence(
            "oracle_future",
            self.oracle_future,
            length=config.prediction_horizon,
            feature_dim=config.obs_dim,
        )
        _validate_sequence(
            "oracle_action_chunk",
            self.oracle_action_chunk,
            length=config.prediction_horizon,
            feature_dim=config.action_dim,
        )
        if self.oracle_shadow_valid.ndim != 1 or self.oracle_shadow_valid.dtype != torch.bool:
            raise ValueError(
                "oracle_shadow_valid must be rank-1 bool, got "
                f"shape={tuple(self.oracle_shadow_valid.shape)} "
                f"dtype={self.oracle_shadow_valid.dtype}"
            )
        if self.idm_source_role.ndim != 1 or self.idm_source_role.dtype != torch.int64:
            raise ValueError(
                "idm_source_role must be rank-1 int64, got "
                f"shape={tuple(self.idm_source_role.shape)} dtype={self.idm_source_role.dtype}"
            )
        valid_roles = torch.zeros_like(self.idm_source_role, dtype=torch.bool)
        for role_id in FADA_IDM_SOURCE_ROLE_IDS.values():
            valid_roles |= self.idm_source_role == role_id
        if not bool(valid_roles.all()):
            raise ValueError("idm_source_role contains an unknown role id")
        _validate_matrix("command", self.command, feature_dim=config.command_dim)
        _validate_matrix(
            "oracle_first_action",
            self.oracle_first_action,
            feature_dim=config.action_dim,
        )
        _validate_row_identity(
            self.command_scenario,
            planner_eligible=self.planner_eligible,
            cold_start=self.cold_start,
        )
        tensors = (
            self.observation_history,
            self.action_history,
            self.command,
            self.realized_future,
            self.executed_action_chunk,
            self.oracle_future,
            self.oracle_action_chunk,
            self.oracle_shadow_valid,
            self.idm_source_role,
            self.oracle_first_action,
            self.command_scenario,
            self.planner_eligible,
            self.cold_start,
        )
        # B2: 在 batch 轴上绑定两类监督对象, 防止跨 rollout 行错配.
        batch_sizes = {int(tensor.shape[0]) for tensor in tensors}
        if len(batch_sizes) != 1:
            raise ValueError(f"FADA source batch sizes must match, got {sorted(batch_sizes)}")
        return self


def _validate_row_identity(
    command_scenario: torch.Tensor,
    *,
    planner_eligible: torch.Tensor,
    cold_start: torch.Tensor,
) -> None:
    identities = {
        "command_scenario": (command_scenario, torch.int64),
        "planner_eligible": (planner_eligible, torch.bool),
        "cold_start": (cold_start, torch.bool),
    }
    for name, (tensor, dtype) in identities.items():
        if tensor.ndim != 1 or tensor.dtype != dtype:
            raise ValueError(f"{name} must be rank-1 {dtype}, got {tensor.shape} {tensor.dtype}")
    valid_ids = torch.zeros_like(command_scenario, dtype=torch.bool)
    for scenario_id in FADA_SCENARIO_IDS.values():
        valid_ids |= command_scenario == scenario_id
    if not bool(valid_ids.all()):
        raise ValueError("command_scenario contains an unknown scenario id")
    cold_start_owner = (command_scenario == FADA_SCENARIO_IDS["walk"]) | (
        command_scenario == FADA_SCENARIO_IDS["static_stand"]
    )
    if bool((cold_start & ~cold_start_owner).any()):
        raise ValueError("cold_start rows must belong to walk or static_stand")
    if bool((cold_start & ~planner_eligible).any()):
        raise ValueError("cold_start rows must be Planner eligible")


def _validate_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be floating point, got dtype={tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_sequence(name: str, tensor: torch.Tensor, *, length: int, feature_dim: int) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be rank-3 [batch, time, feature], got {tensor.shape}")
    if tuple(tensor.shape[1:]) != (length, feature_dim):
        raise ValueError(
            f"{name} shape mismatch: expected [batch, {length}, {feature_dim}], "
            f"got {tuple(tensor.shape)}"
        )
    _validate_finite(name, tensor)


def _validate_matrix(name: str, tensor: torch.Tensor, *, feature_dim: int) -> None:
    if tensor.ndim != 2 or tensor.shape[-1] != feature_dim:
        raise ValueError(
            f"{name} shape mismatch: expected [batch, {feature_dim}], got {tuple(tensor.shape)}"
        )
    _validate_finite(name, tensor)


class _LearnedPositionalEncoding(nn.Module):
    """Bounded positional encoding for one fixed paper window."""

    def __init__(self, *, length: int, hidden_dim: int) -> None:
        super().__init__()
        self.length = int(length)
        self.embedding = nn.Parameter(torch.empty(1, self.length, int(hidden_dim)))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.length:
            raise ValueError(
                f"positional length mismatch: expected {self.length}, got {tokens.shape[1]}"
            )
        return tokens + self.embedding.to(dtype=tokens.dtype)


def _encoder_layer(config: FADAArchitectureConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.hidden_dim,
        nhead=config.num_heads,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
    )


class FADAPlanner(nn.Module):
    """Map observation history and a complete task command to future proprioception."""

    def __init__(self, config: FADAArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_embedding = nn.Linear(config.obs_dim, config.hidden_dim)
        self.command_embedding = nn.Linear(config.command_dim, config.hidden_dim)
        self.position = _LearnedPositionalEncoding(
            length=config.history_length,
            hidden_dim=config.hidden_dim,
        )
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(config),
            num_layers=config.planner_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.future_head = nn.Linear(
            config.hidden_dim,
            config.prediction_horizon * config.obs_dim,
        )

    def forward(self, observation_history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        # B1: 校验 deployable observation history 与完整 task command 的公共输入契约.
        _validate_sequence(
            "observation_history",
            observation_history,
            length=self.config.history_length,
            feature_dim=self.config.obs_dim,
        )
        _validate_matrix("command", command, feature_dim=self.config.command_dim)
        if observation_history.shape[0] != command.shape[0]:
            raise ValueError("Planner observation_history and command batch sizes must match")

        # B2: 构造 command-conditioned history tokens 并编码 command-to-intent 表示.
        command_token = self.command_embedding(command).unsqueeze(1)
        tokens = self.observation_embedding(observation_history) + command_token
        encoded = self.encoder(self.position(tokens))
        # B3: 预测相对最新 observation 的 K-step residual, 产出绝对 future proprioception.
        residual = self.future_head(encoded[:, -1]).reshape(
            observation_history.shape[0],
            self.config.prediction_horizon,
            self.config.obs_dim,
        )
        return observation_history[:, -1:].expand(-1, self.config.prediction_horizon, -1) + residual


class FADAInverseDynamicsModel(nn.Module):
    """Map matched future proprioception and execution history to an action chunk."""

    def __init__(self, config: FADAArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_embedding = nn.Linear(config.obs_dim, config.hidden_dim)
        self.action_embedding = nn.Linear(config.action_dim, config.hidden_dim)
        self.history_position = _LearnedPositionalEncoding(
            length=config.history_length,
            hidden_dim=config.hidden_dim,
        )
        self.history_encoder = nn.TransformerEncoder(
            _encoder_layer(config),
            num_layers=config.idm_encoder_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.future_embedding = nn.Linear(config.obs_dim, config.hidden_dim)
        self.future_position = _LearnedPositionalEncoding(
            length=config.prediction_horizon,
            hidden_dim=config.hidden_dim,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.future_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.idm_decoder_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

    def encode_latent(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        # B1: 校验 execution history 与 future chunk 的 H/K/feature 边界.
        _validate_sequence(
            "observation_history",
            observation_history,
            length=self.config.history_length,
            feature_dim=self.config.obs_dim,
        )
        _validate_sequence(
            "action_history",
            action_history,
            length=self.config.history_length,
            feature_dim=self.config.action_dim,
        )
        _validate_sequence(
            "future",
            future,
            length=self.config.prediction_horizon,
            feature_dim=self.config.obs_dim,
        )
        if len({observation_history.shape[0], action_history.shape[0], future.shape[0]}) != 1:
            raise ValueError("IDM observation, action, and future batch sizes must match")

        # B2: 编码 observation-action history, 产出全部 future tokens 可 cross-attend 的 memory.
        history_tokens = self.observation_embedding(observation_history)
        history_tokens = history_tokens + self.action_embedding(action_history)
        memory = self.history_encoder(self.history_position(history_tokens))
        # B3: 无 causal mask 解码全部 future tokens, 并行产出 K-step action chunk.
        future_tokens = self.future_position(self.future_embedding(future))
        # No tgt mask: every future token has full self-attention over the K-step chunk.
        return self.future_decoder(tgt=future_tokens, memory=memory)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the existing final IDM hidden tokens without changing checkpoint structure."""

        _validate_sequence(
            "latent",
            latent,
            length=self.config.prediction_horizon,
            feature_dim=self.config.hidden_dim,
        )
        return self.action_head(latent)

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode_latent(self.encode_latent(observation_history, action_history, future))


class FADAPlannerIDMPolicy(nn.Module):
    """Compose Planner and IDM under the paper's first-action execution interface."""

    def __init__(
        self,
        config: FADAArchitectureConfig,
        *,
        planner: FADAPlanner | None = None,
        idm: FADAInverseDynamicsModel | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.planner = planner if planner is not None else FADAPlanner(config)
        self.idm = idm if idm is not None else FADAInverseDynamicsModel(config)

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> PlannerIDMOutput:
        # B1: Planner 产生 command-conditioned future proprioceptive intent.
        predicted_future = self.planner(observation_history, command)
        # B2: IDM 使用 intent 与 execution history 产生完整 action chunk.
        action_chunk = self.idm(observation_history, action_history, predicted_future)
        # B3: 仅将 chunk 第一项暴露为 receding-horizon 当前动作.
        return PlannerIDMOutput(
            predicted_future=predicted_future,
            action_chunk=action_chunk,
            action=action_chunk[:, 0],
        )

    @torch.no_grad()
    def explore(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        return self(observation_history, action_history, command).action


def first_action_mse(action_chunk: torch.Tensor, target_action_chunk: torch.Tensor) -> torch.Tensor:
    """Eq. 4.2 boundary: supervise only the physically executed first action."""

    # B1: 验证 predicted/target chunk 具有完全一致的 batch, horizon 与 action 轴.
    if action_chunk.ndim != 3 or target_action_chunk.ndim != 3:
        raise ValueError("action chunks must be rank-3 [batch, horizon, action_dim]")
    if action_chunk.shape != target_action_chunk.shape:
        raise ValueError(
            "action chunk shape mismatch: "
            f"predicted={tuple(action_chunk.shape)} target={tuple(target_action_chunk.shape)}"
        )
    _validate_finite("action_chunk", action_chunk)
    _validate_finite("target_action_chunk", target_action_chunk)
    # B2: 只比较实际会执行的 first action, 产出 Eq. 4.2 标量 loss.
    return F.mse_loss(action_chunk[:, 0], target_action_chunk[:, 0])


def idm_source_loss(
    idm: FADAInverseDynamicsModel,
    batch: FADASourceBatch,
) -> torch.Tensor:
    """Train IDM from one role-selected causal pair per source row."""

    # B1: 先关闭 shape/finite/row mismatch, 确认 causal window 可作为 IDM 证据.
    batch.validate(idm.config)
    trajectory = batch.idm_source_role == FADA_IDM_SOURCE_ROLE_IDS["trajectory"]
    oracle_shadow = (
        (batch.idm_source_role == FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"])
        & batch.oracle_shadow_valid
    )
    selected = trajectory | oracle_shadow
    if not bool(selected.any()):
        raise ValueError("IDM source batch contains no valid role-selected causal rows")
    future = torch.where(
        oracle_shadow[:, None, None], batch.oracle_future, batch.realized_future
    )
    target = torch.where(
        oracle_shadow[:, None, None], batch.oracle_action_chunk, batch.executed_action_chunk
    )
    predicted = idm(
        batch.observation_history[selected],
        batch.action_history[selected],
        future[selected],
    )
    return F.mse_loss(predicted[:, 0], target[selected, 0])


@contextmanager
def _temporarily_frozen(module: nn.Module) -> Iterator[None]:
    original = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original, strict=True):
            parameter.requires_grad_(requires_grad)


def planner_source_loss(
    planner: FADAPlanner,
    idm: FADAInverseDynamicsModel,
    batch: FADASourceBatch,
) -> torch.Tensor:
    """Eq. 4.3: update Planner through a fixed IDM toward oracle first actions."""

    # B1: 校验 Planner/IDM 共享同一 H/K/feature 契约与 source batch 行身份.
    if planner.config != idm.config:
        raise ValueError("Planner and IDM must share one FADA architecture config")
    batch.validate(planner.config)
    # B2: 固定 IDM 参数但保留对 Planner future 的可微路径.
    with _temporarily_frozen(idm):
        predicted_future = planner(batch.observation_history, batch.command)
        predicted_chunk = idm(
            batch.observation_history,
            batch.action_history,
            predicted_future,
        )
        # B3: 仅用 oracle first-action relabel 产出 Eq. 4.3 Planner loss.
        return F.mse_loss(predicted_chunk[:, 0], batch.oracle_first_action)
