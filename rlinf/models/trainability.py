# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurable parameter trainability masks for policy optimization."""

from dataclasses import asdict, dataclass

from torch import nn

TRAINABILITY_MASKS = frozenset({"all", "adarms_only", "exclude_adarms"})


@dataclass(frozen=True)
class TrainabilityMaskReport:
    """Summary of a parameter mask applied before distributed wrapping."""

    mode: str
    adarms_parameter_tensors: int
    adarms_parameters: int
    trainable_parameter_tensors: int
    trainable_parameters: int
    policy_trainable_parameters: int
    value_head_trainable_parameters: int
    adarms_parameter_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a logger-friendly representation."""
        return asdict(self)


def _is_adaptive_rmsnorm(module: nn.Module) -> bool:
    """Identify OpenPI AdaRMS modules by structure, without importing OpenPI."""
    return (
        module.__class__.__name__ == "GemmaRMSNorm"
        and getattr(module, "cond_dim", None) is not None
        and isinstance(getattr(module, "dense", None), nn.Linear)
    )


def collect_adarms_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Collect parameters owned by adaptive, but not ordinary, RMSNorm modules."""
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not _is_adaptive_rmsnorm(module):
            continue
        for local_name, _ in module.named_parameters(recurse=True):
            names.add(f"{module_name}.{local_name}" if module_name else local_name)
    return tuple(sorted(names))


def apply_trainability_mask(
    model: nn.Module,
    mode: str,
    *,
    keep_value_head: bool = True,
) -> TrainabilityMaskReport:
    """Apply an AdaRMS trainability mask before FSDP and optimizer creation.

    The mask controls policy-body parameters. PPO's value head remains trainable by
    default because actor-critic optimization otherwise freezes its critic.
    """
    mode = str(mode).lower()
    if mode not in TRAINABILITY_MASKS:
        raise ValueError(
            f"Unknown trainability_mask={mode!r}; expected one of "
            f"{sorted(TRAINABILITY_MASKS)}"
        )

    named_parameters = dict(model.named_parameters())
    adarms_names = collect_adarms_parameter_names(model)
    if mode != "all" and not adarms_names:
        raise ValueError(
            "AdaRMS trainability mask matched zero parameters. The mask is only "
            "valid for a model with adaptive GemmaRMSNorm dense modulators."
        )

    if mode == "adarms_only":
        for parameter in named_parameters.values():
            parameter.requires_grad = False
        for name in adarms_names:
            named_parameters[name].requires_grad = True
    elif mode == "exclude_adarms":
        for name in adarms_names:
            named_parameters[name].requires_grad = False

    if keep_value_head and hasattr(model, "value_head"):
        for parameter in model.value_head.parameters():
            parameter.requires_grad = True

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    value_parameters = sum(
        parameter.numel()
        for name, parameter in trainable.items()
        if name.startswith("value_head.") or ".value_head." in name
    )
    policy_parameters = (
        sum(parameter.numel() for parameter in trainable.values()) - value_parameters
    )
    if mode != "all" and policy_parameters == 0:
        raise ValueError(
            f"trainability_mask={mode!r} left no trainable policy parameters"
        )

    return TrainabilityMaskReport(
        mode=mode,
        adarms_parameter_tensors=len(adarms_names),
        adarms_parameters=sum(named_parameters[name].numel() for name in adarms_names),
        trainable_parameter_tensors=len(trainable),
        trainable_parameters=sum(parameter.numel() for parameter in trainable.values()),
        policy_trainable_parameters=policy_parameters,
        value_head_trainable_parameters=value_parameters,
        adarms_parameter_names=adarms_names,
    )
