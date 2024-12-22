"""Utility functions for extracting attention components from Transformer blocks."""

import math
from collections.abc import Sequence
from typing import Any, Literal

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from PIL import Image

AffinityMethod = Literal[
    "query_cosine", "key_cosine", "value_cosine", "output_cosine", "attention_weight"
]


def generate_image_and_get_affinity(
    pipeline: StableDiffusion3Pipeline,
    inference_step: int,
    transformer_blocks: Sequence[int],
    prompt: str | Sequence[str],
    num_inference_steps: int,
    guidance_scale: float | Sequence[float],
    random_seed: int | Sequence[int],
    affinity_method: AffinityMethod,
) -> tuple[list[Image.Image], torch.Tensor]:
    """Generates images and gets the token affinity matrix using the specified
    method."""

    # Turn inputs into a batch.
    if not isinstance(prompt, str):
        batch_size = len(prompt)
    elif isinstance(guidance_scale, Sequence):
        batch_size = len(guidance_scale)
    elif isinstance(random_seed, Sequence):
        batch_size = len(random_seed)
    else:
        batch_size = 1

    if isinstance(prompt, str):
        prompts = [prompt] * batch_size
    else:
        prompts = prompt
    if isinstance(guidance_scale, Sequence):
        guidance_scales = guidance_scale
    else:
        guidance_scales = [guidance_scale] * batch_size
    if isinstance(random_seed, Sequence):
        random_seeds = random_seed
    else:
        random_seeds = [random_seed] * batch_size

    match affinity_method:
        case "query_cosine":
            component_names = ["query"]
        case "key_cosine":
            component_names = ["key"]
        case "value_cosine":
            component_names = ["value"]
        case "output_cosine":
            component_names = ["output"]
        case "attention_weight":
            component_names = ["query", "key"]
        case _:
            raise ValueError(f"Invalid affinity method: {affinity_method}")

    generated_images = []
    component_lists = {name: [] for name in component_names}

    for i in range(batch_size):
        generated_image, attention_components = (
            generate_image_and_get_attention_components(
                pipeline,
                inference_step,
                transformer_blocks,
                component_names,
                {
                    "prompt": prompts[i],
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scales[i],
                    "negative_prompt": "",
                    "generator": torch.Generator(pipeline.device).manual_seed(
                        random_seeds[i]
                    ),
                },
            )
        )
        generated_images.append(generated_image)
        for name, value in attention_components.items():
            component_lists[name].append(value)

    # component_name -> (num_heads, num_tokens, head_dim)
    attention_components = {
        name: torch.cat(component_list, dim=1)
        for name, component_list in component_lists.items()
    }

    if affinity_method != "attention_weight":
        component = attention_components[component_names[0]]
        # (num_heads, num_tokens, head_dim) -> (num_tokens, feature_dim)
        component = (
            component.transpose(0, 1)
            .flatten(start_dim=1)
            .to(pipeline.device, torch.float32)
        )
        component /= component.norm(dim=-1, keepdim=True)
        # [-1, 1] -> [0, 2]
        affinity = (component @ component.t()).add_(1)
    else:
        queries = attention_components["query"].to(pipeline.device)
        keys = attention_components["key"].to(pipeline.device)
        num_heads, num_tokens, head_dim = queries.size()
        scale_factor = 1 / math.sqrt(head_dim)

        attention_weight = torch.zeros((num_tokens, num_tokens), device=pipeline.device)
        for query, key in zip(queries, keys):
            attention_weight += (
                (query @ key.t()).mul_(scale_factor).softmax(dim=-1).div_(num_heads)
            )
        affinity = attention_weight

    return generated_images, affinity.cpu()


AttentionComponentName = Literal["query", "key", "value", "output"]


def generate_image_and_get_attention_components(
    pipeline: StableDiffusion3Pipeline,
    inference_step: int,
    transformer_blocks: Sequence[int],
    component_names: Sequence[AttentionComponentName],
    pipeline_kwargs: dict[str, Any],
) -> tuple[Image.Image, dict[AttentionComponentName, torch.Tensor]]:
    """Generates an image and gets the specified attention components from the
    Transformer blocks at a certain inference step."""

    original_joint_attn_processor_call = JointAttnProcessor2_0.__call__
    JointAttnProcessor2_0.__call__ = _joint_attn_processor_call

    def add_hooks():
        for block_index, block in enumerate(pipeline.transformer.transformer_blocks):
            if len(transformer_blocks) == 0 or block_index in transformer_blocks:
                for name in component_names:
                    setattr(block.attn.processor, f"saved_attention_{name}", None)

    attention_components: dict[AttentionComponentName, torch.Tensor] = {}

    def callback_on_step_end(
        _pipeline: StableDiffusion3Pipeline,
        i: int,
        t: torch.Tensor,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if i == inference_step:
            component_lists = {name: [] for name in component_names}
            for block in _pipeline.transformer.transformer_blocks:
                processor = block.attn.processor
                for name in component_names:
                    attribute_name = f"saved_attention_{name}"
                    if hasattr(processor, attribute_name):
                        component_lists[name].append(getattr(processor, attribute_name))
                        delattr(processor, attribute_name)
            for name, component_list in component_lists.items():
                attention_components[name] = torch.cat(component_list)

        if i == inference_step - 1:
            add_hooks()

        return {}

    if inference_step == 0:
        add_hooks()

    generated_image = pipeline(
        **pipeline_kwargs,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]

    JointAttnProcessor2_0.__call__ = original_joint_attn_processor_call

    return generated_image, attention_components


def _joint_attn_processor_call(
    self: JointAttnProcessor2_0,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: torch.FloatTensor | None = None,
    *args,
    **kwargs,
) -> torch.FloatTensor:
    """Modified `JointAttnProcessor2_0.__call__` method that saves attention
    components."""

    residual = hidden_states

    batch_size = hidden_states.shape[0]

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # `context` projections.
    if encoder_hidden_states is not None:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )

    for name, value in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("output", hidden_states),
    ):
        attribute_name = f"saved_attention_{name}"
        if hasattr(self, attribute_name):
            setattr(self, attribute_name, value[-1].cpu())

    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if encoder_hidden_states is not None:
        return hidden_states, encoder_hidden_states
    return hidden_states
