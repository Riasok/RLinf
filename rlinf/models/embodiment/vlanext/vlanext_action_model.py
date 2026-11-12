# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RLinf policy wrapper for VLANeXt (Qwen3-VL backbone + flow-matching DiT action head).

Mirrors the openpi/pi0.5 PPO path: chunk-level Flow-SDE sampling with per-step Gaussian
log-probabilities, a value head on the frozen VLM output, and only the action expert
(DiT head) plus the value head trainable.
"""

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor

logger = get_logger()

# Copied verbatim from VLANeXt/src/datasets/libero_act.py (that module imports tensorflow,
# which the RL runtime does not carry).
LIBERO_ACTION_STATS = {
    "original": {
        "spatial": (
            [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475],
            [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375],
        ),
        "object": (
            [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943],
            [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174],
        ),
        "goal": (
            [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.2871428430080414],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
        "10": (
            [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574],
            [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375],
        ),
        "mixed": (
            [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.3675000071525574],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
    },
    "legacy": {
        "spatial": (
            [-0.9375, -0.9375, -0.9375, -0.18857142329216003, -0.3675000071525574, -0.36000001430511475],
            [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375],
        ),
        "object": (
            [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943],
            [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174],
        ),
        "goal": (
            [-0.9375, -0.9375, -0.9375, -0.24214285612106323, -0.375, -0.2871428430080414],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
        "10": (
            [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3642857074737549],
            [0.9375, 0.9375, 0.9375, 0.32892856001853943, 0.36964285373687744, 0.375],
        ),
        "mixed": (
            [-0.9375, -0.9375, -0.9375, -0.24214285612106323, -0.375, -0.3642857074737549],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
    },
}


def _stats_suite_key(suite_name: str) -> str:
    name = suite_name.lower()
    for prefix in ("libero_",):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def get_libero_action_stats(suite_name: str, version: str = "original"):
    lo, hi = LIBERO_ACTION_STATS[version][_stats_suite_key(suite_name)]
    return np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)


@dataclass
class VLANeXtRLConfig:
    # paths
    repo_path: str = ""  # VLANeXt repo root (provides `src.models.VLANeXt`)
    checkpoint_path: str = ""  # .pt with {model_state_dict, config}
    lmm_path: str | None = None  # local Qwen3-VL dir; overrides the path stored in the checkpoint
    attn_implementation: str = "flash_attention_2"
    # flow sampling (same semantics as OpenPi0Config)
    noise_method: str = "flow_sde"  # flow_ode | flow_sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(default_factory=lambda: [0.7, 0.3, 400])
    num_steps: int = 5
    action_chunk: int = 4  # executed actions per chunk
    action_env_dim: int = 7
    ignore_last: bool = False
    joint_logprob: bool = False
    safe_get_logprob: bool = False
    # observation handling
    image_size: int = 256
    center_crop_ratio: float = 0.9
    history_len: int | None = None  # None -> take data.history_len from the checkpoint config
    max_text_len: int = 256  # fixed padding so stored token tensors stack across steps
    normalization_suite: str = "libero_spatial"
    normalization_stats_version: str = "original"
    # trainability
    train_expert_only: bool = True
    train_vlm_inputs: bool = False  # meta_queries / action_projector; needs grad through the VLM
    add_value_head: bool = False
    value_head_hidden_sizes: tuple = (512, 256, 128)
    value_head_activation: str = "relu"
    fp32_action_head: bool = True


def _import_vlanext(repo_path: str):
    repo_path = os.path.abspath(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from src.models.VLANeXt import VLANeXt  # noqa: E402

    return VLANeXt


def _load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "model_state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"VLANeXt checkpoint {path} must contain model_state_dict and config")
    return ckpt


class VLANeXtForRLActionPrediction(nn.Module, BasePolicy):
    """VLANeXt policy exposing RLinf's rollout/actor interface."""

    config: VLANeXtRLConfig

    @property
    def _no_split_modules(self) -> list[str]:
        # Keep the trainable fp32 DiT head in its own FSDP unit, separate from the frozen bf16 VLM.
        return ["ActionDiffusionTransformerMoE"]

    @property
    def _fsdp_ignored_states(self):
        """Frozen parameters FSDP must not flatten: the bf16 VLM and everything that feeds it.
        They run under no_grad inside forward, so FSDP's parameter views would break (Qwen3-VL
        reads self.visual.dtype through parameters())."""
        params = list(self.model.lmm.parameters()) + [self.model.meta_queries]
        for name in ("action_projector", "connector"):
            sub = getattr(self.model, name, None)
            if sub is not None:
                params += list(sub.parameters())
        return [p for p in params if not p.requires_grad]

    def __init__(self, config: VLANeXtRLConfig):
        super().__init__()
        self.config = config
        self.global_step = 0
        if not config.train_expert_only or config.train_vlm_inputs:
            raise NotImplementedError(
                "VLANeXt RL currently trains only the action head (+ value head). "
                "meta_queries/action_projector live in the frozen VLM FSDP unit."
            )

        VLANeXt = _import_vlanext(config.repo_path)
        ckpt = _load_checkpoint(config.checkpoint_path)
        train_config = ckpt["config"]
        mcfg = train_config["model"]
        dcfg = train_config["data"]
        if mcfg.get("loss_type", "diffusion") != "diffusion" or mcfg.get("scheduler_type") != "flow_match":
            raise ValueError("VLANeXt RL requires a flow_match diffusion checkpoint.")
        if mcfg.get("condition_type", "loose") not in ("tight", "soft"):
            raise ValueError("VLANeXt RL supports tight/soft conditioning only.")
        if mcfg.get("use_action_input_policy", False):
            raise ValueError("use_action_input_policy is not supported in RL.")

        self.history_len = int(config.history_len if config.history_len is not None else dcfg["history_len"])
        self.action_horizon = int(dcfg["future_len"])
        self.action_dim = int(mcfg["action_dim"])
        self.diffusion_loss_domain = mcfg.get("diffusion_loss_domain", "noise")
        self.train_config = train_config

        lmm_path = config.lmm_path or mcfg["lmm_path"]
        self.model = VLANeXt(
            lmm_path=lmm_path,
            vision_encoder_path=mcfg.get("vision_encoder_path", "google/siglip2-base-patch16-256"),
            use_pretrained_backbone=mcfg.get("use_pretrained_backbone", True),
            action_dim=self.action_dim,
            num_actions=self.action_horizon,
            num_queries=mcfg["num_queries"],
            num_history=self.history_len,
            loss_type="diffusion",
            future_image_loss_weight=float(mcfg.get("future_image_loss_weight", 0.0)),
            num_train_timesteps=mcfg.get("num_train_timesteps", 1000),
            num_inference_timesteps=config.num_steps,
            scheduler_type="flow_match",
            diffusion_loss_domain=self.diffusion_loss_domain,
            condition_type=mcfg.get("condition_type", "loose"),
            policy_hidden_size=mcfg.get("policy_hidden_size", 1024),
            policy_depth=mcfg.get("policy_depth", 29),
            policy_num_heads=mcfg.get("policy_num_heads", 12),
            policy_mlp_ratio=mcfg.get("policy_mlp_ratio", 4.0),
            policy_pos_embed=mcfg.get("policy_pos_embed", "absolute"),
            use_proprio_input_vlm=mcfg.get("use_proprio_input_vlm", True),
            use_action_input_policy=False,
            use_transformer_proprio_projector=mcfg.get("use_transformer_proprio_projector", False),
            projector_depth=mcfg.get("projector_depth", 2),
            projector_num_heads=mcfg.get("projector_num_heads", 4),
            use_transformer_connector=mcfg.get("use_transformer_connector", True),
            connector_depth=mcfg.get("connector_depth", 2),
            connector_num_heads=mcfg.get("connector_num_heads", 4),
            backbone_mode="frozen",
            gradient_checkpointing=False,
            num_bins=mcfg.get("num_bins", 256),
            classification_type=mcfg.get("classification_type", "parallel"),
            fast_action_tokenizer=None,
            generator_hidden_size=mcfg.get("generator_hidden_size", 768),
            generator_depth=mcfg.get("generator_depth", mcfg.get("policy_depth", 12)),
            generator_num_heads=mcfg.get("generator_num_heads", 12),
            generator_mlp_ratio=mcfg.get("generator_mlp_ratio", 4.0),
            generator_max_seq_len=mcfg.get("generator_max_seq_len", 1024),
            future_image_num_tokens=mcfg.get("future_image_num_tokens", 256),
            future_image_prediction_type=mcfg.get("future_image_prediction_type", "emu_token"),
            attn_implementation=config.attn_implementation,
            language_action_loss_weight=0.0,
            policy_gradient_stop_for_vlm=True,
            dct_loss_weight=mcfg.get("dct_loss_weight", 0.1),
            video_generation_loss_weight=0.0,
        )
        if self.model.model_family != "qwen":
            raise ValueError("VLANeXt RL wrapper supports the Qwen3-VL family only.")

        state_dict = ckpt["model_state_dict"]
        if next(iter(state_dict)).startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"[VLANeXt] loaded {config.checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            logger.warning(f"[VLANeXt] missing keys (first 10): {missing[:10]}")
        del ckpt, state_dict

        # trainability: action head only; everything feeding the VLM stays frozen
        self.model.requires_grad_(False)
        self.model.action_head.requires_grad_(True)

        # dtypes: frozen VLM side in bf16, trainable head in fp32 (actor) or bf16 (rollout)
        self.model.lmm.to(torch.bfloat16)
        self.model.meta_queries.data = self.model.meta_queries.data.to(torch.bfloat16)
        if getattr(self.model, "action_projector", None) is not None:
            self.model.action_projector.to(torch.bfloat16)
        if getattr(self.model, "connector", None) is not None:
            self.model.connector.to(torch.bfloat16)
        head_dtype = torch.float32 if config.fp32_action_head else torch.bfloat16
        self.model.action_head.to(head_dtype)

        self.use_vlm_value = bool(config.add_value_head)
        if self.use_vlm_value:
            self.value_head = ValueHead(
                input_dim=self.model.hidden_size,
                hidden_sizes=tuple(config.value_head_hidden_sizes),
                output_dim=1,
                activation=config.value_head_activation,
                bias_last=True,
            ).to(head_dtype)

        self.processor = self.model.processor
        self._prompt_cache: dict[str, str] = {}
        self.action_min, self.action_max = get_libero_action_stats(
            config.normalization_suite, config.normalization_stats_version
        )
        self.model.eval()

    # ------------------------------------------------------------------ misc hooks
    def set_global_step(self, global_step: int):
        self.global_step = int(global_step)

    def gradient_checkpointing_enable(self, **kwargs):
        logger.warning("[VLANeXt] gradient checkpointing is a no-op (VLM is frozen and run under no_grad).")

    def gradient_checkpointing_disable(self):
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        # the frozen VLM never needs train-mode behaviour (dropout off, etc.)
        self.model.lmm.eval()
        return self

    # ------------------------------------------------------------------ observation prep
    @property
    def _device(self):
        return self.model.action_head.input_proj.weight.device

    def _prep_images(self, images: torch.Tensor) -> np.ndarray:
        """env images [B,H,W,3] or [B,3,H,W] uint8 (already rotated like training data) ->
        center-crop(ratio) + resize(image_size) uint8 [B,S,S,3], matching VLANeXt eval."""
        arr = images.detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(0, 2, 3, 1)
        size = self.config.image_size
        ratio = float(self.config.center_crop_ratio)
        out = np.empty((arr.shape[0], size, size, 3), dtype=np.uint8)
        for i, img in enumerate(arr):
            h, w = img.shape[:2]
            if 0 < ratio < 1.0:
                side = min(h, w)
                crop = max(1, int(round(side * ratio)))
                top, left = (h - crop) // 2, (w - crop) // 2
                img = img[top : top + crop, left : left + crop]
            if img.shape[0] != size or img.shape[1] != size:
                img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((size, size), Image.Resampling.LANCZOS))
            out[i] = img
        return out

    def _prep_proprio(self, states: torch.Tensor) -> torch.Tensor:
        """env state [B,8] = eef_pos(3) + axis-angle(3) + gripper_qpos(2) -> VLANeXt proprio [B,hist,7]."""
        s = states.detach().float().cpu().numpy()
        gripper = np.clip(1.0 - np.mean(np.abs(s[:, 6:8]), axis=1, keepdims=True) / 0.04, 0.0, 1.0)
        proprio = np.concatenate([s[:, :6], gripper], axis=1).astype(np.float32)
        proprio = np.repeat(proprio[:, None, :], self.history_len, axis=1)
        return torch.from_numpy(proprio)

    def _prompt_text(self, task: str) -> str:
        text = self._prompt_cache.get(task)
        if text is None:
            dummy = Image.fromarray(np.zeros((self.config.image_size, self.config.image_size, 3), np.uint8))
            content = [{"type": "image", "image": dummy}, {"type": "image", "image": dummy}, {"type": "text", "text": task}]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            self._prompt_cache[task] = text
        return text

    def _flat_pil(self, main: np.ndarray, wrist: np.ndarray) -> list[Image.Image]:
        imgs = []
        for m, w in zip(main, wrist):
            imgs.append(Image.fromarray(m))
            imgs.append(Image.fromarray(w))
        return imgs

    def _tokenize(self, tasks: list[str], main: np.ndarray, wrist: np.ndarray) -> dict[str, torch.Tensor]:
        texts = [self._prompt_text(t) for t in tasks]
        enc = self.processor(
            text=texts,
            images=self._flat_pil(main, wrist),
            padding="max_length",
            max_length=self.config.max_text_len,
            return_tensors="pt",
        )
        if enc["input_ids"].shape[1] != self.config.max_text_len:
            raise ValueError(
                f"prompt longer than max_text_len={self.config.max_text_len}; got {enc['input_ids'].shape[1]}"
            )
        return {k: enc[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw")}

    def _image_features(self, main: np.ndarray, wrist: np.ndarray) -> dict[str, torch.Tensor]:
        enc = self.processor.image_processor(images=self._flat_pil(main, wrist), return_tensors="pt")
        return {"pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"]}

    # ------------------------------------------------------------------ VLM conditioning
    def _vlm_condition(self, inputs: dict[str, torch.Tensor], proprio: torch.Tensor):
        """Returns (hidden_states tuple, position_ids, last_hidden_state). Always no-grad: the VLM
        and everything feeding it are frozen."""
        dev = self._device
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.get_vlm_condition(
                input_ids=inputs["input_ids"].to(dev),
                attention_mask=inputs["attention_mask"].to(dev),
                proprioception=proprio.to(device=dev, dtype=torch.bfloat16),
                pixel_values=inputs["pixel_values"].to(device=dev, dtype=torch.bfloat16),
                image_grid_thw=inputs["image_grid_thw"].to(dev),
                mm_token_type_ids=inputs["mm_token_type_ids"].to(dev),
                return_last_hidden_state=True,
            )
        _, hidden_states, _, position_ids, last_hidden = out
        hidden_states = tuple(h.detach() for h in hidden_states)
        return hidden_states, position_ids, last_hidden.detach()

    def _value_from_vlm(self, last_hidden: torch.Tensor) -> torch.Tensor:
        # meta queries are appended after the (right-padded) text, so they are the last tokens.
        q = last_hidden[:, -self.model.num_queries :, :].float().mean(dim=1)
        return self.value_head(q.to(next(self.value_head.parameters()).dtype))[:, 0].float()

    # ------------------------------------------------------------------ flow matching
    def _velocity(self, x_t: torch.Tensor, t: torch.Tensor, hidden_states, position_ids) -> torch.Tensor:
        timesteps = t.float() * self.model.noise_scheduler.config.num_train_timesteps
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.action_head(
                x_t,
                timesteps,
                hidden_states,
                history_actions=None,
                gen_hidden_states=None,
                vlm_position_ids=position_ids,
            )
        out = out.float()
        if self.diffusion_loss_domain == "x0":
            sigma = t.float()[:, None, None].clamp_min(1e-6)
            out = (x_t.float() - out) / sigma
        return out

    def _get_timesteps(self, denoise_steps: int, device):
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        return torch.cat([timesteps, torch.zeros((1), device=device)])

    def _get_noise_level(self, device, sample_method: str | None = None) -> torch.Tensor:
        method = sample_method or self.config.noise_method
        if method == "flow_ode":
            return torch.zeros((), device=device)
        if self.config.noise_anneal:
            start, end, steps = self.config.noise_params
            level = start + (end - start) * min(self.global_step, steps) / steps
        else:
            level = self.config.noise_level
        return torch.full((), float(level), device=device)

    def sample_mean_var(self, x_t, idx, hidden_states, position_ids, sample_method, denoise_steps):
        """Same math as openpi's sample_mean_var_val (flow_ode / flow_sde)."""
        bsize, device = x_t.shape[0], x_t.device
        if isinstance(idx, int):
            idx = torch.full((bsize,), idx, device=device, dtype=torch.long)
        timesteps = self._get_timesteps(denoise_steps, device)
        noise_level = self._get_noise_level(device, sample_method)
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        v_t = self._velocity(x_t, t_input, hidden_states, position_ids)
        delta = delta[:, None, None].expand_as(x_t)
        t_in = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_in
        x1_pred = x_t + v_t * (1 - t_in)
        if sample_method == "flow_ode":
            x0_w = 1 - (t_in - delta)
            x1_w = t_in - delta
            std = torch.zeros_like(t_in)
        elif sample_method == "flow_sde":
            denom = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigma_ratio = timesteps / (1 - denom)
            sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_w = torch.ones_like(t_in) - (t_in - delta)
            x1_w = t_in - delta - sigma_i**2 * delta / (2 * t_in)
            std = torch.sqrt(delta) * sigma_i
        else:
            raise ValueError(f"Unsupported noise method {sample_method}")
        mean = x0_pred * x0_w + x1_pred * x1_w
        return mean, std, v_t

    def get_logprob_norm(self, sample, mu, sigma):
        if self.config.safe_get_logprob:
            return -torch.pow(sample - mu, 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        const = -torch.log(sigma_safe) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
        expo = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        return torch.where(mask, torch.zeros_like(expo), const + expo)

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def sample_actions(self, hidden_states, position_ids, last_hidden, mode="train", compute_values=True):
        device = self._device
        bsize = hidden_states[-1].shape[0]
        num_steps = self.config.num_steps
        x_t = torch.randn(bsize, self.action_horizon, self.action_dim, device=device, dtype=torch.float32)
        chains, log_probs = [x_t], []
        if mode == "train":
            hi = num_steps - 2 if self.config.ignore_last else num_steps - 1
            denoise_inds = torch.tensor([random.randint(0, hi)] * num_steps)
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1).to(device)
        for idx in range(num_steps):
            method = self.config.noise_method if idx == int(denoise_inds[0][idx]) else "flow_ode"
            mean, std, _ = self.sample_mean_var(x_t, idx, hidden_states, position_ids, method, num_steps)
            x_t = mean + torch.randn_like(x_t) * std
            log_probs.append(self.get_logprob_norm(x_t, mean, std))
            chains.append(x_t)
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[:, :, : self.config.action_chunk, : self.config.action_env_dim]
        log_probs = log_probs[torch.arange(bsize, device=device), denoise_inds[:, 0]]
        if self.use_vlm_value and compute_values:
            values = self._value_from_vlm(last_hidden)[:, None]
        else:
            values = torch.zeros((bsize, 1), device=device)
        return {"actions": x_t, "chains": chains, "prev_logprobs": log_probs, "prev_values": values, "denoise_inds": denoise_inds}

    def _denormalize(self, model_actions: torch.Tensor) -> torch.Tensor:
        a = model_actions.detach().float().cpu().numpy().copy()
        lo, hi = self.action_min, self.action_max
        a[..., :6] = (np.clip(a[..., :6], -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo
        a[..., 6] = np.where(a[..., 6] > 0, 1.0, -1.0)
        return torch.from_numpy(a)

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        main = self._prep_images(env_obs["main_images"])
        wrist = self._prep_images(env_obs["wrist_images"])
        proprio = self._prep_proprio(env_obs["states"])
        tasks = list(env_obs["task_descriptions"])
        tok = self._tokenize(tasks, main, wrist)
        hidden_states, position_ids, last_hidden = self._vlm_condition(tok, proprio)
        outputs = self.sample_actions(hidden_states, position_ids, last_hidden, mode=mode, compute_values=compute_values)
        env_actions = self._denormalize(outputs["actions"])[:, : self.config.action_chunk]
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "mm_token_type_ids": tok["mm_token_type_ids"],
            "proprio": proprio,
            "main_images": torch.from_numpy(main),
            "wrist_images": torch.from_numpy(wrist),
            "action": env_actions.reshape(env_actions.shape[0], -1).contiguous(),
            "model_action": outputs["actions"].reshape(outputs["actions"].shape[0], -1).contiguous(),
        }
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": copy_dict_tensor(forward_inputs),
            "model_actions": outputs["actions"],
        }
        return env_actions, result

    # ------------------------------------------------------------------ actor training
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"VLANeXt RL does not implement {forward_type}")

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs) -> dict[str, Any]:
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"].float()
        denoise_inds = forward_inputs["denoise_inds"].long()
        bsize, device = chains.shape[0], chains.device
        feats = self._image_features(
            forward_inputs["main_images"].cpu().numpy(), forward_inputs["wrist_images"].cpu().numpy()
        )
        tok = {
            "input_ids": forward_inputs["input_ids"],
            "attention_mask": forward_inputs["attention_mask"],
            "mm_token_type_ids": forward_inputs["mm_token_type_ids"],
            **feats,
        }
        hidden_states, position_ids, last_hidden = self._vlm_condition(tok, forward_inputs["proprio"])
        batch_idx = torch.arange(bsize, device=device)
        step_idx = denoise_inds[:, 0]
        x_pre = chains[batch_idx, step_idx]
        x_next = chains[batch_idx, step_idx + 1]
        mean, std, _ = self.sample_mean_var(
            x_pre, step_idx, hidden_states, position_ids, self.config.noise_method, self.config.num_steps
        )
        log_probs = self.get_logprob_norm(x_next, mean, std)[:, : self.config.action_chunk, : self.config.action_env_dim]
        entropy = torch.zeros((bsize, 1), device=device, dtype=log_probs.dtype)
        if self.use_vlm_value and compute_values:
            values = self._value_from_vlm(last_hidden)
        else:
            values = torch.zeros((bsize,), device=device)
        return {"logprobs": log_probs, "values": values, "entropy": entropy}
