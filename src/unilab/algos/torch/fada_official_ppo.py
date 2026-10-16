"""Independent FADA Oracle PPO learner, using UniLab's RSL-RL contracts.

Numerical reference: LeCAR-Lab/FADA@290b3a9c5c0ebe2e9851b61350617544be05e9bc,
agents/ppo/ppo.py and agents/modules/ppo_modules.py. No collector or checkpoint
format conversion lives here; those are separate integration owners.
"""
from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal, kl_divergence

from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO

LEARNER_SCHEMA = "fada_official_ppo_v1"


class OfficialFadaGaussianDistribution(GaussianDistribution):
    """Direct learnable std, with the same effective scale as official FADA."""

    def __init__(self, output_dim, init_std=.8, std_type="scalar"):
        if std_type != "scalar":
            raise ValueError("Official FADA uses direct std parameters, not log_std")
        super().__init__(output_dim, init_std=init_std, std_type=std_type)

    def update(self, mlp_output):
        scale = torch.nan_to_num(
            self.std_param.clamp(min=1e-6), nan=1e-6, posinf=1., neginf=1e-6
        )
        self._distribution = Normal(mlp_output, scale.expand_as(mlp_output))


class OfficialFadaPPO(FinalObservationAwarePPO):
    """Official losses and separate AdamW optimizers; old PPO is untouched."""

    def __init__(
        self, actor, critic, storage, *, actor_learning_rate=1e-5,
        critic_learning_rate=1e-5, weight_decay=.001, enable_compile=False,
        amp=False, **kwargs,
    ):
        if enable_compile or amp:
            raise ValueError("Official FADA v1 does not support compile or AMP")
        if any(kwargs.get(name) is not None for name in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg")):
            raise ValueError("Official FADA v1 requires no RND, symmetry or multi-GPU")
        if kwargs.get("normalize_advantage_per_mini_batch", False):
            raise ValueError("Official FADA normalizes advantages over the whole rollout")
        if not kwargs.get("use_clipped_value_loss", True):
            raise ValueError("Official FADA requires clipped value loss")
        if actor.is_recurrent or critic.is_recurrent or actor.obs_normalization or critic.obs_normalization:
            raise ValueError("Official FADA requires nonrecurrent, unnormalized models")
        if not isinstance(actor.distribution, OfficialFadaGaussianDistribution):
            raise ValueError("Official FADA requires its protected direct-std Gaussian")
        actor_params, critic_params = list(actor.parameters()), list(critic.parameters())
        if {id(p) for p in actor_params} & {id(p) for p in critic_params}:
            raise ValueError("Official FADA actor and critic must not share parameters")
        if any(not math.isfinite(lr) or lr <= 0 for lr in (actor_learning_rate, critic_learning_rate)):
            raise ValueError("Official FADA learning rates must be positive and finite")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Official FADA weight_decay must be finite and nonnegative")
        optimizer = kwargs.pop("optimizer", "adamw")
        if optimizer != "adamw":
            raise ValueError("Official FADA requires AdamW")
        # The parent owns transition/storage setup, but its temporary shared
        # optimizer must never participate in updates or serialization.
        kwargs.setdefault("learning_rate", actor_learning_rate)
        super().__init__(actor, critic, storage, optimizer="adamw", enable_compile=False, **kwargs)
        del self.optimizer
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.actor_optimizer = torch.optim.AdamW(
            actor_params, lr=actor_learning_rate, weight_decay=weight_decay, betas=(.9, .999), eps=1e-8
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic_params, lr=critic_learning_rate, weight_decay=weight_decay, betas=(.9, .999), eps=1e-8
        )
        self.learning_rate = self.actor_learning_rate  # RSL-RL logger compatibility
        self._returns_ready = False
        self._validate_allocation()

    def _validate_allocation(self):
        size = self.storage.num_envs * self.storage.num_transitions_per_env
        if (self.num_mini_batches < 1 or self.num_learning_epochs < 1 or size < 2
                or size < self.num_mini_batches or size % self.num_mini_batches):
            raise ValueError("Official FADA requires complete, evenly divisible rollout minibatches")

    def process_env_step(self, obs, rewards, dones, extras):
        if self._returns_ready:
            raise RuntimeError("Update the completed rollout before collecting another transition")
        timeouts = extras.get("time_outs")
        if timeouts is not None and torch.any(timeouts):
            if extras.get("time_out_bootstrap_obs") is None:
                raise ValueError("Official FADA timeout requires pre-reset time_out_bootstrap_obs")
            if torch.any(timeouts.to(dones.device).bool() & ~dones.bool()):
                raise ValueError("Timeout rows must also be done rows")
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("Official FADA returns require a full rollout")
        super().compute_returns(obs)
        self._returns_ready = True

    def _compute_losses(self, batch):
        # Official PPO calls actor.act during updates too, consuming a Gaussian
        # draw even though only its distribution is needed for the loss.
        self.actor(batch.observations, stochastic_output=True)
        logp = self.actor.get_output_log_prob(batch.actions)
        values = self.critic(batch.observations)
        entropy = self.actor.output_entropy.mean()
        mean, std = self.actor.output_distribution_params
        old_mean, old_std = batch.old_distribution_params
        with torch.no_grad():
            kl = kl_divergence(Normal(old_mean, old_std), Normal(mean, std)).sum(-1).mean()
        ratio = (logp - batch.old_actions_log_prob.squeeze(-1)).exp()
        advantage = batch.advantages.squeeze(-1)
        surrogate = torch.maximum(
            -advantage * ratio,
            -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
        ).mean()
        clipped_value = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.maximum(
            (values - batch.returns).square(), (clipped_value - batch.returns).square()
        ).mean()
        return {
            "actor_loss": surrogate - self.entropy_coef * entropy,
            "critic_loss": self.value_loss_coef * value_loss,
            "surrogate": surrogate, "value": value_loss, "entropy": entropy, "kl": kl,
        }

    def _update_learning_rates(self, kl):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        kl = float(kl)
        for name, optimizer in (("actor_learning_rate", self.actor_optimizer),
                                ("critic_learning_rate", self.critic_optimizer)):
            lr = getattr(self, name)
            if kl > 2 * self.desired_kl:
                lr = max(1e-5, lr / 1.5)
            elif 0 < kl < self.desired_kl / 2:
                lr = min(1e-2, lr * 1.5)
            setattr(self, name, lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
        self.learning_rate = self.actor_learning_rate

    def update(self):
        self._validate_allocation()
        if self.storage.step != self.storage.num_transitions_per_env or not self._returns_ready:
            raise RuntimeError("Official FADA update requires full rollout and computed returns")
        totals = dict.fromkeys(("value", "surrogate", "entropy", "kl"), 0.)
        count = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            losses = self._compute_losses(batch)
            self._update_learning_rates(losses["kl"])
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            (losses["actor_loss"] + losses["critic_loss"]).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach())
            count += 1
        if count != self.num_learning_epochs * self.num_mini_batches:
            raise RuntimeError("Official FADA minibatch count mismatch")
        self.storage.clear()
        self._returns_ready = False
        return {**{name: value / count for name, value in totals.items()},
                "actor_learning_rate": self.actor_learning_rate,
                "critic_learning_rate": self.critic_learning_rate}

    def save(self):
        return {
            "learner_schema": LEARNER_SCHEMA,
            "actor_state_dict": self.actor.state_dict(), "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
        }

    @staticmethod
    def _validate_model_state(model, incoming, strict):
        current = model.state_dict()
        if strict and current.keys() != incoming.keys():
            raise ValueError("Official FADA model state keys mismatch")
        for name in current.keys() & incoming.keys():
            if current[name].shape != incoming[name].shape:
                raise ValueError(f"Official FADA model state shape mismatch: {name}")

    @staticmethod
    def _validate_optimizer_state(optimizer, incoming, lr):
        groups = incoming["param_groups"]
        if len(groups) != len(optimizer.param_groups) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Official FADA optimizer groups/lr mismatch")
        for saved, current in zip(groups, optimizer.param_groups, strict=True):
            if len(saved["params"]) != len(current["params"]) or saved["lr"] != lr:
                raise ValueError("Official FADA optimizer parameters/lr mismatch")
            for name in ("betas", "eps", "weight_decay", "amsgrad", "maximize"):
                if saved.get(name) != current.get(name):
                    raise ValueError(f"Official FADA optimizer setting mismatch: {name}")
            for key, parameter in zip(saved["params"], current["params"], strict=True):
                state = incoming["state"].get(key, {})
                for name in ("exp_avg", "exp_avg_sq"):
                    if state and (name not in state or state[name].shape != parameter.shape):
                        raise ValueError("Official FADA AdamW moment shape mismatch")

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if self.storage.step or self._returns_ready or self.transition.actions is not None:
            raise RuntimeError("Official FADA load requires an empty rollout and no pending transition")
        flags = {"actor": True, "critic": True, "optimizer": True, "iteration": True} if load_cfg is None else load_cfg
        if loaded_dict.get("learner_schema") != LEARNER_SCHEMA:
            raise ValueError("Not an Official FADA PPO learner state")
        required = {f"{name}_state_dict" for name in ("actor", "critic") if flags.get(name)}
        if flags.get("optimizer"):
            required.update(f"{name}_{suffix}" for name in ("actor", "critic")
                            for suffix in ("optimizer_state_dict", "learning_rate"))
        if not required.issubset(loaded_dict):
            raise ValueError(f"Official FADA learner state missing: {sorted(required - loaded_dict.keys())}")
        # Validate requested models and both optimizer payloads before changing
        # any live parameters. External Oracle lineage is owned by the adapter.
        for name in ("actor", "critic"):
            if flags.get(name):
                self._validate_model_state(getattr(self, name), loaded_dict[f"{name}_state_dict"], strict)
            if flags.get("optimizer"):
                self._validate_optimizer_state(getattr(self, f"{name}_optimizer"),
                                               loaded_dict[f"{name}_optimizer_state_dict"],
                                               loaded_dict[f"{name}_learning_rate"])
        for name in ("actor", "critic"):
            if flags.get(name):
                getattr(self, name).load_state_dict(loaded_dict[f"{name}_state_dict"], strict=strict)
            if flags.get("optimizer"):
                getattr(self, f"{name}_optimizer").load_state_dict(loaded_dict[f"{name}_optimizer_state_dict"])
                setattr(self, f"{name}_learning_rate", float(loaded_dict[f"{name}_learning_rate"]))
        self.learning_rate = self.actor_learning_rate
        return bool(flags.get("iteration", False))
