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


class SmolVLMTokenKVPrior(nn.Module):
    """p(z | VLM tokens): one learnable query reads the frozen VLM's layer-wise prefix K/V through a
    SmolVLA-style action expert, then heads give (mu_p, sigma_p).

    Expert layer l reads VLM layer l: even layers self-attend over [prefix K/V, the query itself], odd
    layers cross-attend to the prefix K/V re-projected into the expert's K/V width.
    """

    expert_width_multiplier = 0.75
    self_attn_every_n_layers = 2

    def __init__(
        self,
        config: "LatentSDEConfig",
        text_config: "PretrainedConfig",
        z_dim: int,
        sigma_act,
        sigma_min: float,
    ):
        super().__init__()
        require_package("transformers", extra="latent_sde")
        from transformers import AutoModel

        expert_config = copy.deepcopy(text_config)
        expert_hidden_size = int(text_config.hidden_size * self.expert_width_multiplier)
        expert_config.hidden_size = expert_hidden_size
        expert_config.intermediate_size = get_intermediate_size(expert_hidden_size)
        expert_config.num_hidden_layers = config.vlm_num_layers
        self.expert = AutoModel.from_config(expert_config)
        self.expert.embed_tokens = None

        self.num_attention_heads = expert_config.num_attention_heads
        self.num_key_value_heads = expert_config.num_key_value_heads
        self.head_dim = expert_config.head_dim
        self.prefix_kv_width = text_config.num_key_value_heads * text_config.head_dim
        expert_kv_width = self.num_key_value_heads * self.head_dim
        for layer_index, layer in enumerate(self.expert.layers):
            if not self._is_self_attention_layer(layer_index):
                layer.self_attn.k_proj = nn.Linear(
                    self.prefix_kv_width, expert_kv_width, bias=expert_config.attention_bias
                )
                layer.self_attn.v_proj = nn.Linear(
                    self.prefix_kv_width, expert_kv_width, bias=expert_config.attention_bias
                )

        # FP32 query and heads keep the residual stream wider than the BF16 expert.
        self.query = nn.Parameter(torch.empty(expert_hidden_size).uniform_(-1, 1))
        self.output_proj = nn.Linear(expert_hidden_size, 2 * z_dim)
        self.sigma_act = sigma_act
        self.sigma_min = sigma_min
        # Wide prior at init (σ_p ≈ 1 for exp), as in LatentPrior.
        with torch.no_grad():
            self.output_proj.weight[z_dim:].zero_()
            self.output_proj.bias[z_dim:].zero_()

    def _is_self_attention_layer(self, layer_index: int) -> bool:
        return layer_index % self.self_attn_every_n_layers == 0

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_length = query.shape[:2]
        key_length = key.shape[1]
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
        weights = torch.where(attention_mask[:, None, None], weights, torch.finfo(weights.dtype).min)
        probabilities = F.softmax(weights, dim=-1).to(value.dtype)
        output = torch.matmul(probabilities, value.permute(0, 2, 1, 3))
        return output.permute(0, 2, 1, 3).reshape(
            batch_size, query_length, self.num_attention_heads * self.head_dim
        )

    def _self_attention(
        self,
        layer: nn.Module,
        normalized: torch.Tensor,
        prefix: LayerKV,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = normalized.shape[0]
        normalized = normalized.to(dtype=layer.self_attn.q_proj.weight.dtype)
        # The query sits right after the valid prefix, like a next token.
        position = prefix_mask.sum(dim=1, keepdim=True)
        query = layer.self_attn.q_proj(normalized).view(batch_size, 1, self.num_attention_heads, self.head_dim)
        query = apply_rope(query, position)
        own_key = layer.self_attn.k_proj(normalized).view(batch_size, 1, self.num_key_value_heads, self.head_dim)
        own_key = apply_rope(own_key, position)
        own_value = layer.self_attn.v_proj(normalized).view(
            batch_size, 1, self.num_key_value_heads, self.head_dim
        )
        key = torch.cat([prefix.key.to(dtype=own_key.dtype), own_key], dim=1)
        value = torch.cat([prefix.value.to(dtype=own_value.dtype), own_value], dim=1)
        mask = torch.cat([prefix_mask, prefix_mask.new_ones(batch_size, 1)], dim=1)
        return self._attention(query, key, value, mask)

    def _cross_attention(
        self,
        layer: nn.Module,
        normalized: torch.Tensor,
        prefix: LayerKV,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, prefix_length = prefix.key.shape[:2]
        normalized = normalized.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query = layer.self_attn.q_proj(normalized).view(batch_size, 1, self.num_attention_heads, self.head_dim)
        query = apply_rope(query, torch.zeros((batch_size, 1), dtype=torch.long, device=normalized.device))
        flat_key = prefix.key.to(dtype=layer.self_attn.k_proj.weight.dtype).reshape(
            batch_size, prefix_length, self.prefix_kv_width
        )
        flat_value = prefix.value.to(dtype=layer.self_attn.v_proj.weight.dtype).reshape(
            batch_size, prefix_length, self.prefix_kv_width
        )
        key = layer.self_attn.k_proj(flat_key).view(
            batch_size, prefix_length, self.num_key_value_heads, self.head_dim
        )
        value = layer.self_attn.v_proj(flat_value).view(
            batch_size, prefix_length, self.num_key_value_heads, self.head_dim
        )
        return self._attention(query, key, value, prefix_mask)

    def forward(self, context: ObservationContext) -> tuple[torch.Tensor, torch.Tensor]:
        assert context.layer_kv is not None and context.valid_mask is not None
        prefix_mask = context.valid_mask
        hidden_states = self.query.expand(prefix_mask.shape[0], 1, -1)
        for layer_index, layer in enumerate(self.expert.layers):
            prefix = context.layer_kv[layer_index]
            normalized = layer.input_layernorm(hidden_states)
            if self._is_self_attention_layer(layer_index):
                attention_output = self._self_attention(layer, normalized, prefix, prefix_mask)
            else:
                attention_output = self._cross_attention(layer, normalized, prefix, prefix_mask)
            attention_output = attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
            hidden_states = hidden_states + layer.self_attn.o_proj(attention_output)
            normalized = layer.post_attention_layernorm(hidden_states)
            normalized = normalized.to(dtype=layer.mlp.gate_proj.weight.dtype)
            hidden_states = hidden_states + layer.mlp(normalized)

        hidden_states = self.expert.norm(hidden_states)[:, 0]
        output = self.output_proj(hidden_states.to(dtype=self.output_proj.weight.dtype)).float()
        mu, sigma = output.chunk(2, dim=-1)
        return mu, self.sigma_act(sigma) + self.sigma_min
