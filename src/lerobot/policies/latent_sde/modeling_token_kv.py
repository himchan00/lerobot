# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope, get_intermediate_size
from lerobot.utils.import_utils import require_package

from .modeling_context import LayerKV, ObservationContext

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from .configuration_latent_sde import LatentSDEConfig


class SmolVLMTokenKVDrift(nn.Module):
    """Pointwise drift expert conditioned on the frozen VLM's layer-wise prefix K/V."""

    expert_width_multiplier = 0.75
    self_attn_every_n_layers = 2

    def __init__(
        self,
        config: "LatentSDEConfig",
        text_config: "PretrainedConfig",
        state_dim: int,
        output_dim: int,
        z_dim: int,
    ):
        super().__init__()
        require_package("transformers", extra="latent_sde")
        from transformers import AutoModel

        self.z_dim = z_dim
        self.z_mode = config.z_mode
        self.num_layers = config.vlm_num_layers

        expert_config = copy.deepcopy(text_config)
        vlm_hidden_size = expert_config.hidden_size
        expert_hidden_size = int(vlm_hidden_size * self.expert_width_multiplier)
        expert_config.hidden_size = expert_hidden_size
        expert_config.intermediate_size = get_intermediate_size(expert_hidden_size)
        expert_config.num_hidden_layers = self.num_layers
        self.expert = AutoModel.from_config(expert_config)
        self.expert.embed_tokens = None

        self.num_attention_heads = expert_config.num_attention_heads
        self.num_key_value_heads = expert_config.num_key_value_heads
        self.head_dim = expert_config.head_dim
        self.prefix_kv_width = text_config.num_key_value_heads * text_config.head_dim
        self.expert_kv_width = self.num_key_value_heads * self.head_dim

        for layer_index, layer in enumerate(self.expert.layers):
            if not self._is_self_attention_layer(layer_index):
                layer.self_attn.k_proj = nn.Linear(
                    self.prefix_kv_width,
                    self.expert_kv_width,
                    bias=expert_config.attention_bias,
                )
                layer.self_attn.v_proj = nn.Linear(
                    self.prefix_kv_width,
                    self.expert_kv_width,
                    bias=expert_config.attention_bias,
                )

        input_dim = state_dim + (z_dim if z_dim > 0 and self.z_mode == "input" else 0)
        self.input_proj = nn.Linear(input_dim, expert_hidden_size)
        self.z_proj = nn.Linear(z_dim, expert_hidden_size) if z_dim > 0 and self.z_mode == "cond" else None
        self.output_proj = nn.Linear(expert_hidden_size, output_dim)

    def _is_self_attention_layer(self, layer_index: int) -> bool:
        return layer_index % self.self_attn_every_n_layers == 0

    def _embed_queries(self, state: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if self.z_dim > 0 and self.z_mode == "input":
            assert z is not None
            z_queries = z[:, None, :].expand(-1, state.shape[1], -1)
            state = torch.cat([state, z_queries], dim=-1)
        hidden_states = self.input_proj(state.to(dtype=self.input_proj.weight.dtype))
        if self.z_proj is not None:
            assert z is not None
            hidden_states = hidden_states + self.z_proj(z.to(dtype=self.z_proj.weight.dtype)).unsqueeze(1)
        return hidden_states

    def _prefix_for_layer(
        self,
        context: ObservationContext,
        layer_index: int,
        batch_size: int,
        use_context: bool,
        device: torch.device,
    ) -> tuple[LayerKV | None, torch.Tensor]:
        if not use_context:
            return None, torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        assert context.layer_kv is not None and context.valid_mask is not None
        layer_kv = context.layer_kv[layer_index]
        valid_mask = context.valid_mask.to(device=device)
        return layer_kv, valid_mask

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_length = query.shape[:2]
        key_length = key.shape[1]
        if key_length == 0:
            return query.new_zeros((batch_size, query_length, self.num_attention_heads * self.head_dim))

        num_groups = self.num_attention_heads // self.num_key_value_heads
        key = (
            key[:, :, :, None, :]
            .expand(batch_size, key_length, self.num_key_value_heads, num_groups, self.head_dim)
            .reshape(batch_size, key_length, self.num_attention_heads, self.head_dim)
        )
        value = (
            value[:, :, :, None, :]
            .expand(batch_size, key_length, self.num_key_value_heads, num_groups, self.head_dim)
            .reshape(batch_size, key_length, self.num_attention_heads, self.head_dim)
        )
        weights = torch.matmul(
            query.to(torch.float32).transpose(1, 2),
            key.to(torch.float32).transpose(1, 2).transpose(2, 3),
        )
        weights *= self.head_dim**-0.5
        weights = torch.where(
            attention_mask[:, None],
            weights,
            torch.finfo(weights.dtype).min,
        )
        probabilities = F.softmax(weights, dim=-1).to(value.dtype)
        output = torch.matmul(probabilities, value.permute(0, 2, 1, 3))
        return output.permute(0, 2, 1, 3).reshape(
            batch_size, query_length, self.num_attention_heads * self.head_dim
        )

    def _self_attention(
        self,
        layer: nn.Module,
        normalized: torch.Tensor,
        prefix: LayerKV | None,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon = normalized.shape[:2]
        normalized = normalized.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query = layer.self_attn.q_proj(normalized).view(
            batch_size, horizon, self.num_attention_heads, self.head_dim
        )
        query_position = prefix_mask.sum(dim=1, keepdim=True).expand(-1, horizon)
        query = apply_rope(query, query_position)
        query_key = layer.self_attn.k_proj(normalized).view(
            batch_size, horizon, self.num_key_value_heads, self.head_dim
        )
        query_key = apply_rope(query_key, query_position)
        query_value = layer.self_attn.v_proj(normalized).view(
            batch_size, horizon, self.num_key_value_heads, self.head_dim
        )

        if prefix is None:
            prefix_key = query_key[:, :0]
            prefix_value = query_value[:, :0]
        else:
            prefix_key = prefix.key.to(device=query_key.device, dtype=query_key.dtype)
            prefix_value = prefix.value.to(device=query_value.device, dtype=query_value.dtype)
        key = torch.cat([prefix_key, query_key], dim=1)
        value = torch.cat([prefix_value, query_value], dim=1)
        diagonal = torch.eye(horizon, dtype=torch.bool, device=normalized.device)
        mask = torch.cat(
            [prefix_mask[:, None, :].expand(-1, horizon, -1), diagonal[None].expand(batch_size, -1, -1)],
            dim=-1,
        )
        return self._attention(query, key, value, mask)

    def _cross_attention(
        self,
        layer: nn.Module,
        normalized: torch.Tensor,
        prefix: LayerKV | None,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon = normalized.shape[:2]
        normalized = normalized.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query = layer.self_attn.q_proj(normalized).view(
            batch_size, horizon, self.num_attention_heads, self.head_dim
        )
        query = apply_rope(
            query,
            torch.zeros((batch_size, horizon), dtype=torch.long, device=normalized.device),
        )
        if prefix is None:
            return query.new_zeros((batch_size, horizon, self.num_attention_heads * self.head_dim))

        prefix_length = prefix.key.shape[1]
        flat_key = prefix.key.to(
            device=layer.self_attn.k_proj.weight.device,
            dtype=layer.self_attn.k_proj.weight.dtype,
        ).reshape(batch_size, prefix_length, self.prefix_kv_width)
        flat_value = prefix.value.to(
            device=layer.self_attn.v_proj.weight.device,
            dtype=layer.self_attn.v_proj.weight.dtype,
        ).reshape(batch_size, prefix_length, self.prefix_kv_width)
        key = layer.self_attn.k_proj(flat_key).view(
            batch_size, prefix_length, self.num_key_value_heads, self.head_dim
        )
        value = layer.self_attn.v_proj(flat_value).view(
            batch_size, prefix_length, self.num_key_value_heads, self.head_dim
        )
        mask = prefix_mask[:, None, :].expand(-1, horizon, -1)
        return self._attention(query, key, value, mask)

    def forward(
        self,
        state: torch.Tensor,
        context: ObservationContext,
        z: torch.Tensor | None,
        *,
        use_context: bool,
    ) -> torch.Tensor:
        squeeze_horizon = state.ndim == 2
        if squeeze_horizon:
            state = state.unsqueeze(1)
        hidden_states = self._embed_queries(state, z)
        batch_size = hidden_states.shape[0]

        for layer_index, layer in enumerate(self.expert.layers):
            prefix, prefix_mask = self._prefix_for_layer(
                context,
                layer_index,
                batch_size,
                use_context,
                hidden_states.device,
            )
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            if self._is_self_attention_layer(layer_index):
                attention_output = self._self_attention(layer, normalized, prefix, prefix_mask)
            else:
                attention_output = self._cross_attention(layer, normalized, prefix, prefix_mask)
            attention_output = attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
            hidden_states = residual + layer.self_attn.o_proj(attention_output)
            residual = hidden_states
            normalized = layer.post_attention_layernorm(hidden_states)
            # FP32 query projections keep the residual stream wider than the BF16 expert.
            normalized = normalized.to(dtype=layer.mlp.gate_proj.weight.dtype)
            hidden_states = residual + layer.mlp(normalized)

        hidden_states = self.expert.norm(hidden_states)
        output = self.output_proj(hidden_states.to(dtype=self.output_proj.weight.dtype))
        return output[:, 0] if squeeze_horizon else output
