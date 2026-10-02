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

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lerobot.utils.import_utils import require_package

from .modeling_context import LayerKV, ObservationContext

if TYPE_CHECKING:
    from .configuration_latent_sde import LatentSDEConfig


def _resize_with_pad(
    image: torch.Tensor,
    width: int,
    height: int,
    pad_value: float = -1,
) -> torch.Tensor:
    current_height, current_width = image.shape[2:]
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(image, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    return F.pad(resized, (pad_width, 0, pad_height, 0), value=pad_value)


def _preprocess_images(images: torch.Tensor, resize_shape: tuple[int, int]) -> torch.Tensor:
    images = _resize_with_pad(images, *resize_shape, pad_value=0)
    return images * 2.0 - 1.0


class SmolVLMContextEncoder(nn.Module):
    """Frozen generic SmolVLM2 with trainable state-history tokens; returns the layer-wise prefix K/V."""

    def __init__(self, config: "LatentSDEConfig", state_dim: int):
        super().__init__()
        require_package("transformers", extra="latent_sde")
        from transformers import AutoModelForImageTextToText

        self.resize_shape = tuple(config.vlm_resize_shape)
        self.num_layers = config.vlm_num_layers
        self.state_dim = state_dim
        self.padded_state_dim = max(32, state_dim)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.vlm_model_name,
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True,
        )

        text_model = self.vlm.model.text_model
        text_model.layers = text_model.layers[: self.num_layers]
        hidden_size = self.vlm.config.text_config.hidden_size

        for parameter in self.vlm.parameters():
            parameter.requires_grad_(False)
        self.vlm.eval()

        self.state_proj = nn.Linear(self.padded_state_dim, hidden_size)

    @property
    def text_config(self):
        return self.vlm.config.text_config

    def train(self, mode: bool = True) -> "SmolVLMContextEncoder":
        super().train(mode)
        self.vlm.eval()
        return self

    def _embed_images(self, images: torch.Tensor) -> torch.Tensor:
        num_frames, num_cameras = images.shape[1:3]
        vlm_model = self.vlm.model
        vision_model = vlm_model.vision_model
        vision_parameter = next(vision_model.parameters())
        connector = vlm_model.connector
        connector_parameter = next(connector.parameters())
        image_embeddings = []
        # Match SmolVLA's B-sized vision calls while keeping frame-major/camera order.
        for frame in range(num_frames):
            for camera in range(num_cameras):
                image = _preprocess_images(images[:, frame, camera], self.resize_shape)
                image_hidden_states = vision_model(
                    pixel_values=image.to(device=vision_parameter.device, dtype=vision_parameter.dtype),
                    patch_attention_mask=None,
                ).last_hidden_state
                image_embeddings.append(
                    connector(image_hidden_states.to(device=connector_parameter.device))
                )

        image_hidden_states = torch.cat(image_embeddings, dim=1)
        hidden_size = image_hidden_states.shape[-1]
        return image_hidden_states * torch.tensor(
            hidden_size**0.5, dtype=image_hidden_states.dtype, device=image_hidden_states.device
        )

    def _embed_language(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.vlm.model.text_model.get_input_embeddings()
        embedding_parameter = next(embeddings.parameters())
        hidden_states = embeddings(token_ids.to(device=embedding_parameter.device))
        return hidden_states * math.sqrt(hidden_states.shape[-1])

    def _run_text_model(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[LayerKV, ...]:
        from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

        text_model = self.vlm.model.text_model
        batch_size, sequence_length = hidden_states.shape[:2]
        layer_kv = []
        # SmolVLA bypasses HF's decoder: RoPE uses 10_000 and eager attention upcasts Q/K.
        for layer in text_model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attention = layer.self_attn
            hidden_states = hidden_states.to(dtype=attention.q_proj.weight.dtype)
            head_dim = attention.head_dim
            hidden_shape = (batch_size, sequence_length, -1, head_dim)
            query_states = apply_rope(attention.q_proj(hidden_states).view(hidden_shape), position_ids)
            key_states = apply_rope(attention.k_proj(hidden_states).view(hidden_shape), position_ids)
            value_states = attention.v_proj(hidden_states).view(hidden_shape)
            layer_kv.append(LayerKV(key=key_states, value=value_states))
            if len(layer_kv) == len(text_model.layers):
                break  # only the K/V are read; the last layer's output is never used
            num_heads = query_states.shape[2]
            num_kv_heads = key_states.shape[2]
            num_kv_groups = num_heads // num_kv_heads
            key_states = (
                key_states[:, :, :, None, :]
                .expand(batch_size, sequence_length, num_kv_heads, num_kv_groups, head_dim)
                .reshape(batch_size, sequence_length, num_heads, head_dim)
            )
            value_states = (
                value_states[:, :, :, None, :]
                .expand(batch_size, sequence_length, num_kv_heads, num_kv_groups, head_dim)
                .reshape(batch_size, sequence_length, num_heads, head_dim)
            )

            query_states = query_states.to(torch.float32).transpose(1, 2)
            key_states = key_states.to(torch.float32).transpose(1, 2)
            weights = torch.matmul(query_states, key_states.transpose(2, 3))
            weights *= head_dim**-0.5
            weights = torch.where(attention_mask[:, None], weights, torch.finfo(weights.dtype).min)
            probs = F.softmax(weights, dim=-1).to(value_states.dtype)
            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

            hidden_states = attention.o_proj(att_output.to(dtype=attention.o_proj.weight.dtype))
            hidden_states += residual
            residual = hidden_states.clone()
            hidden_states = layer.mlp(layer.post_attention_layernorm(hidden_states))
            hidden_states += residual

        return tuple(layer_kv)

    def _pad_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.padded_state_dim == self.state_dim:
            return state
        padded = state.new_zeros((*state.shape[:-1], self.padded_state_dim))
        padded[..., : self.state_dim] = state
        return padded

    def forward(
        self,
        images: torch.Tensor,
        language_token_ids: torch.Tensor,
        language_attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> ObservationContext:
        """Encode state frames ordered oldest to newest, accepting (B, T, D) or one (B, D) frame."""
        with torch.no_grad():
            image_hidden_states = self._embed_images(images)
            language_hidden_states = self._embed_language(language_token_ids)
        language_hidden_states = language_hidden_states.to(device=image_hidden_states.device)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        state_parameter = next(self.state_proj.parameters())
        state_hidden_states = self.state_proj(
            self._pad_state(state).to(device=state_parameter.device, dtype=state_parameter.dtype)
        )
        hidden_states = torch.cat(
            [
                image_hidden_states,
                language_hidden_states,
                state_hidden_states.to(device=image_hidden_states.device),
            ],
            dim=1,
        )

        image_mask = torch.ones(
            image_hidden_states.shape[:2],
            dtype=torch.bool,
            device=hidden_states.device,
        )
        num_state_tokens = state_hidden_states.shape[1]
        state_mask = torch.ones(state_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        valid_mask = torch.cat(
            [image_mask, language_attention_mask.to(device=hidden_states.device), state_mask],
            dim=1,
        )
        attention_mask = valid_mask[:, None, :] & valid_mask[:, :, None]
        # Image/text queries cannot read state; state queries read only themselves and earlier states.
        attention_mask[:, :-num_state_tokens, -num_state_tokens:] = False
        attention_mask[:, -num_state_tokens:, -num_state_tokens:] = torch.ones(
            (num_state_tokens, num_state_tokens), dtype=torch.bool, device=hidden_states.device
        ).tril()
        position_ids = torch.cumsum(valid_mask, dim=1) - 1
        layer_kv = self._run_text_model(hidden_states, attention_mask, position_ids)
        return ObservationContext(layer_kv=layer_kv, valid_mask=valid_mask)
