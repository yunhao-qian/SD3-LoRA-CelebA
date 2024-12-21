"""The `visualize-attention-weights` subcommand."""

import logging
import math
from collections.abc import Collection
from typing import Any, TypedDict

import click
import gradio as gr
import ncut_pytorch
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from matplotlib import pyplot as plt
from PIL import Image

from ..visualizer import TextTokenAndImageVisualizer

_logger = logging.getLogger(__name__)


class VisualizeAttentionWeightArgs(TypedDict):
    """Arguments to the `visualize-attention-weights` subcommand."""

    model_name: str
    model_revision: str
    lora_weight_dir: tuple[str, ...]
    share: bool


@click.command()
@click.option(
    "--model-name",
    default="stabilityai/stable-diffusion-3-medium-diffusers",
    help="Stable Diffusion 3 model name",
)
@click.option(
    "--model-revision", default="main", help="Stable Diffusion 3 model revision"
)
@click.option(
    "--lora-weight-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    help="zero, one, or more directories containing LoRA weights",
)
@click.option("--share/--no-share", default=False, help="share the Gradio app")
def visualize_attention_weights(**kwargs: VisualizeAttentionWeightArgs) -> None:
    """Gradio app to visualize attention weights."""

    _logger.info("Arguments to visualize-attention-weights: %s", kwargs)
    VisualizeAttentionWeight(kwargs).run()


class VisualizeAttentionWeight:
    """Implementation of the `visualize-attention-weights` subcommand."""

    def __init__(self, args: VisualizeAttentionWeightArgs) -> None:
        self.args = args

        self.pipeline: StableDiffusion3Pipeline | None = None
        self.visualizer: TextTokenAndImageVisualizer | None = None

        self.visualization_mode: str | None = None
        self.clip_tokens: list[int] | None = None
        self.t5_tokens: list[int] | None = None
        self.generated_image: Image.Image | None = None
        self.attention_weight: np.ndarray | None = None

    def run(self) -> None:
        """Runs the subcommand."""

        _logger.info(
            "Loading model '%s', revision '%s'",
            self.args["model_name"],
            self.args["model_revision"],
        )
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.args["model_name"],
            revision=self.args["model_revision"],
            torch_dtype=torch.float16,
            device_map="balanced",
        )

        self.visualizer = TextTokenAndImageVisualizer()

        title = "Attention Weight Visualization for Stable Diffusion 3"
        with gr.Blocks(title=title) as demo:
            gr.Markdown(f"# {title}")

            gr.Markdown("### Inference Settings")
            with gr.Group():
                prompt_input = gr.Textbox(label="Prompt")
                lora_weight_dir_input = gr.Dropdown(
                    choices=["None", *self.args["lora_weight_dir"]],
                    value="None",
                    label="LoRA Weights",
                )
                num_inference_steps_input = gr.Number(
                    value=28,
                    label="Inference Steps",
                    minimum=1,
                    maximum=1000,
                    precision=0,
                )
                guidance_scale_input = gr.Slider(
                    minimum=0, maximum=20, value=7, label="Guidance Scale"
                )
                random_seed_input = gr.Number(
                    value=42,
                    label="Random Seed",
                    minimum=0,
                    maximum=2**32 - 1,
                    precision=0,
                )

            gr.Markdown("### Visualization Settings")
            with gr.Group():
                visualization_mode_input = gr.Radio(
                    choices=["Heatmap", "NCUT"],
                    value="Heatmap",
                    label="Visualization Mode",
                )
                inference_step_input = gr.Number(
                    value=27,
                    label="Inference Step to Visualize",
                    minimum=0,
                    maximum=27,
                    precision=0,
                )
                transformer_blocks_input = gr.Dropdown(
                    choices=["All", *(f"Block {i}" for i in range(24))],
                    value="All",
                    label="Transformer Blocks to Visualize",
                )
                num_eigenvectors_input = gr.Number(
                    value=30,
                    label="Eigenvectors for NCUT",
                    minimum=1,
                    maximum=4096 + 77 + 256,
                    precision=0,
                    visible=False,
                )

            run_button = gr.Button(value="Run")

            figure_output = gr.Image(
                self.visualizer.visualize_empty_data(),
                label="Visualization",
                type="pil",
                interactive=False,
            )

            # Update the inference step to visualize based on the number of inference
            # steps.
            # pylint: disable-next=no-member
            num_inference_steps_input.change(
                lambda value, num_inference_steps: gr.update(
                    value=min(value, num_inference_steps - 1),
                    maximum=num_inference_steps - 1,
                ),
                inputs=[inference_step_input, num_inference_steps_input],
                outputs=inference_step_input,
            )

            # `num_eigenvectors` is applicable only to the NCUT mode.
            # pylint: disable-next=no-member
            visualization_mode_input.change(
                lambda value: gr.update(visible=value == "NCUT"),
                inputs=[visualization_mode_input],
                outputs=num_eigenvectors_input,
            )

            # pylint: disable-next=no-member
            run_button.click(
                self.on_run_button_click,
                inputs=[
                    prompt_input,
                    lora_weight_dir_input,
                    num_inference_steps_input,
                    guidance_scale_input,
                    random_seed_input,
                    visualization_mode_input,
                    inference_step_input,
                    transformer_blocks_input,
                    num_eigenvectors_input,
                ],
                outputs=figure_output,
            )

            # pylint: disable-next=no-member
            figure_output.select(
                self.on_figure_output_select,
                inputs=[figure_output],
                outputs=figure_output,
            )

        demo.launch(share=self.args["share"])

    def on_run_button_click(
        self,
        prompt: str,
        lora_weight_dir: str,
        num_inference_steps: int,
        guidance_scale: float,
        random_seed: int,
        visualization_mode: str,
        inference_step: int,
        transformer_blocks_str: str,
        num_eigenvectors: int,
    ) -> Image.Image:
        """Callback when the run button is clicked."""

        _logger.info("Run button clicked with prompt: %s", prompt)

        self.visualization_mode = visualization_mode
        self.get_text_tokens(prompt)

        if lora_weight_dir == "None":
            lora_weight_dir = None
        if transformer_blocks_str == "All":
            transformer_blocks = []
        else:
            # "Block {i}" -> i
            transformer_blocks = [int(transformer_blocks_str.split()[-1])]
        self.get_generated_image_and_attention_weight(
            lora_weight_dir,
            inference_step,
            transformer_blocks,
            {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": torch.Generator(self.pipeline.device).manual_seed(
                    random_seed
                ),
            },
        )

        if visualization_mode == "Heatmap":
            clip_token_colors = np.zeros((len(self.clip_tokens), 4), dtype=np.uint8)
            clip_token_colors[:, 3] = 16
            t5_token_colors = np.zeros((len(self.t5_tokens), 4), dtype=np.uint8)
            t5_token_colors[:, 3] = 16
            image_to_display = self.generated_image
        else:
            clip_token_colors, t5_token_colors, image_to_display = (
                self.apply_ncut_colors(num_eigenvectors)
            )

        return self.visualizer.visualize_data(
            self.clip_tokens,
            clip_token_colors,
            self.t5_tokens,
            t5_token_colors,
            image_to_display,
        )

    def on_figure_output_select(
        self, figure: Image.Image, event: gr.SelectData
    ) -> Image.Image:
        """Callback when the figure output is selected."""

        _logger.info("Figure output selected at %s", event.index)

        if self.visualization_mode != "Heatmap":
            return figure

        mouse_x, mouse_y = event.index
        token_index = self.visualizer.get_token_index(mouse_x, mouse_y)
        if token_index is None:
            return figure

        clip_token_colors = self.attention_weight_to_heatmap(
            self.attention_weight[token_index, 4096 : 4096 + len(self.clip_tokens)],
            kind="clip",
        )
        t5_token_colors = self.attention_weight_to_heatmap(
            self.attention_weight[
                token_index, 4096 + 77 : 4096 + 77 + len(self.t5_tokens)
            ],
            kind="t5",
        )

        image_overlay = self.attention_weight_to_heatmap(
            self.attention_weight[token_index, 0:4096], kind="image"
        ).reshape(64, 64, 4)
        image_to_display = self.apply_overlay(image_overlay)

        return self.visualizer.visualize_data(
            self.clip_tokens,
            clip_token_colors,
            self.t5_tokens,
            t5_token_colors,
            image_to_display,
        )

    def get_generated_image_and_attention_weight(
        self,
        lora_weight_dir: str | None,
        inference_step: int,
        transformer_blocks: Collection[int],
        pipeline_kwargs: dict[str, Any],
    ) -> None:
        """Gets the generated image and the attention weight at the specified inference
        step and transformer blocks."""

        original_joint_attn_processor_call = JointAttnProcessor2_0.__call__
        JointAttnProcessor2_0.__call__ = _joint_attn_processor_call
        if lora_weight_dir is not None:
            self.pipeline.load_lora_weights(lora_weight_dir)

        def add_hooks():
            for block_index, block in enumerate(
                self.pipeline.transformer.transformer_blocks
            ):
                if len(transformer_blocks) == 0 or block_index in transformer_blocks:
                    block.attn.processor.saved_attention_weight = None

        def callback_on_step_end(
            pipeline: StableDiffusion3Pipeline,
            i: int,
            t: torch.Tensor,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            if i == inference_step:
                attention_weights = []
                for block in pipeline.transformer.transformer_blocks:
                    processor = block.attn.processor
                    if not hasattr(processor, "saved_attention_weight"):
                        continue
                    attention_weights.append(processor.saved_attention_weight)
                    del processor.saved_attention_weight
                self.attention_weight = (
                    torch.stack(attention_weights).mean(dim=0).float().numpy()
                )

            if i == inference_step - 1:
                add_hooks()

            return {}

        if inference_step == 0:
            add_hooks()

        self.generated_image = self.pipeline(
            **pipeline_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=[],
        ).images[0]

        if lora_weight_dir is not None:
            self.pipeline.unload_lora_weights()
        JointAttnProcessor2_0.__call__ = original_joint_attn_processor_call

    def get_text_tokens(self, prompt: str) -> tuple[dict[int, str], dict[int, str]]:
        """Gets the mappings from CLIP/T5 token indices to the string representations of
        those tokens."""

        clip_token_ids = self.pipeline.tokenizer(
            prompt,
            padding=False,
            max_length=77,
            truncation=True,
        )["input_ids"]
        self.clip_tokens = self.pipeline.tokenizer.convert_ids_to_tokens(clip_token_ids)

        t5_token_ids = self.pipeline.tokenizer_3(
            prompt,
            padding=False,
            max_length=256,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"]
        self.t5_tokens = self.pipeline.tokenizer_3.convert_ids_to_tokens(t5_token_ids)

    @staticmethod
    def attention_weight_to_heatmap(weight: np.ndarray, kind: str) -> np.ndarray:
        """Converts the attention weight to a heatmap."""

        weight_min = weight.min()
        weight_max = weight.max()
        weight = ((weight - weight_min) / (weight_max - weight_min + 1e-6)).clip(0, 1)
        heatmap = plt.get_cmap("plasma")(weight)
        if kind == "image":
            heatmap[:, 3] = weight * 0.6 + 0.4  # alpha in [0.6, 1.0]
        return (heatmap * 255).astype(np.uint8)

    def apply_ncut_colors(
        self, num_eigenvectors: int
    ) -> tuple[np.ndarray, np.ndarray, Image.Image]:
        """Applies NCUT colors to the text tokens and the image."""

        attention_weight = torch.from_numpy(self.attention_weight).to(
            self.pipeline.device
        )
        attention_weight = attention_weight + attention_weight.t()
        attention_weight /= 2
        sqrt_diagonal = attention_weight.sum(dim=1).sqrt_()
        attention_weight /= sqrt_diagonal[:, None]
        attention_weight /= sqrt_diagonal[None, :]

        eigenvectors, eigenvalues, _ = torch.svd_lowrank(
            attention_weight, q=num_eigenvectors
        )
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        sorted_indices = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Correct the flipping signs of the eigenvectors.
        eigenvector_signs = eigenvectors.sum(dim=0).sign()
        eigenvectors *= eigenvector_signs

        _, ncut_colors = ncut_pytorch.rgb_from_tsne_3d(eigenvectors)
        ncut_colors = ncut_colors.numpy()

        clip_token_colors = ncut_colors[4096 : 4096 + len(self.clip_tokens)]
        t5_token_colors = ncut_colors[4096 + 77 : 4096 + 77 + len(self.t5_tokens)]

        image_overlay = ncut_colors[:4096].reshape(64, 64, 3)
        image_overlay = (image_overlay * 255).astype(np.uint8)
        # Add alpha = 180 to the image overlay.
        image_overlay = np.concatenate(
            (image_overlay, np.full((64, 64, 1), 180, dtype=np.uint8)), axis=2
        )
        image_to_display = self.apply_overlay(image_overlay)

        return clip_token_colors, t5_token_colors, image_to_display

    def apply_overlay(self, overlay: np.ndarray) -> Image.Image:
        """Alpha composites the overlay on the generated image."""

        assert self.generated_image.size == (1024, 1024)
        assert overlay.shape == (64, 64, 4)

        # Upsample (64, 64, 4) -> (1024, 1024, 4)
        overlay = np.repeat(overlay, 16, axis=0)
        overlay = np.repeat(overlay, 16, axis=1)

        return Image.alpha_composite(
            self.generated_image.convert("RGBA"), Image.fromarray(overlay, "RGBA")
        )


def _joint_attn_processor_call(
    self: JointAttnProcessor2_0,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: torch.FloatTensor | None = None,
    *args,
    **kwargs,
) -> torch.FloatTensor:
    """Modified `JointAttnProcessor2_0.__call__` method that saves the attention
    weights."""

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

        if hasattr(self, "saved_attention_weight"):
            # When classifier-free guidance is enabled, the batch size is 2, and we use
            # the later example which uses the positive prompt.
            scale_factor = 1 / math.sqrt(head_dim)
            # (24, 4429, 64) @ (24, 64, 4429) -> (24, 4429, 4429)
            attn_weight = query[-1] @ key[-1].transpose(-2, -1) * scale_factor
            attn_weight = attn_weight.softmax(dim=-1)
            # (24, 4429, 4429) -> (4429, 4429)
            self.saved_attention_weight = attn_weight.mean(dim=0).cpu()

    # pylint: disable=not-callable
    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
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
