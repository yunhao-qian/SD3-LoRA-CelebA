"""The `visualize-token-affinities` subcommand."""

import logging
from typing import Any, TypedDict

import click
import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from matplotlib import pyplot as plt
from ncut_pytorch import rgb_from_tsne_3d
from PIL import Image

from ..attention import generate_image_and_get_affinity
from ..ncut import get_ncut_eigenvectors
from ..prompt import get_prompt_tokens
from ..visualizer import MultiPromptAndImageVisualizer, VisualizerData

_logger = logging.getLogger(__name__)


class VisualizeTokenAffinitiesArgs(TypedDict):
    """Arguments to the `visualize-token-affinities` subcommand."""

    model_name: str
    model_path: str
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
def visualize_token_affinities(**kwargs: VisualizeTokenAffinitiesArgs) -> None:
    """Gradio app that visualizes affinities between image and text tokens."""

    _logger.info("Arguments to visualize-token-affinities: %s", kwargs)
    VisualizeTokenAffinities(kwargs).run()


class VisualizeTokenAffinities:
    """Implementation of the `visualize-token-affinities` subcommand."""

    _AFFINITY_METHODS = {
        "Attention Weights": "attention_weight",
        "Attention Output Cosine Similarities": "output_cosine",
        "Attention Query Cosine Similarities": "query_cosine",
        "Attention Key Cosine Similarities": "key_cosine",
        "Attention Value Cosine Similarities": "value_cosine",
    }

    def __init__(self, args: VisualizeTokenAffinitiesArgs) -> None:
        self.args = args

        self.pipeline: StableDiffusion3Pipeline | None = None
        self.visualizer: MultiPromptAndImageVisualizer | None = None

        self.text_tokens: list[tuple[list[str], list[str]]] = []
        self.generated_images: list[Image.Image] = []
        self.affinity: torch.Tensor | None = None
        self.ncut_eigenvectors: torch.Tensor | None = None

        self._figure_output_select_event: gr.SelectData | None = None

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

        self.visualizer = MultiPromptAndImageVisualizer()

        title = "Token Affinity Visualization for Stable Diffusion 3"
        with gr.Blocks(title=title) as demo:
            gr.Markdown(f"# {title}")

            gr.Markdown("## Prompts")
            num_images_input = gr.Dropdown(
                choices=[str(i) for i in range(1, 4)],
                value="1",
                label="Number of Images to Generate",
            )
            prompt_inputs: list[gr.Textbox] = []
            with gr.Group():
                for i in range(1, 4):
                    prompt_inputs.append(
                        gr.Textbox(label=f"Prompt {i}", visible=i == 1)
                    )

            gr.Markdown("## Image Generation Settings")
            with gr.Group():
                lora_weight_dir_input = gr.Dropdown(
                    choices=["None", *self.args["lora_weight_dir"]],
                    value="None",
                    label="LoRA Weights",
                )
                num_inference_steps_input = gr.Number(
                    value=28,
                    label="Number of Inference Steps",
                    minimum=1,
                    maximum=1000,
                    precision=0,
                )
                guidance_scale_input = gr.Number(
                    minimum=0,
                    maximum=20,
                    value=7,
                    label="Classifier-Free Guidance Scale",
                )
                random_seed_input = gr.Number(
                    value=42,
                    label="Random Seed",
                    minimum=0,
                    maximum=2**32 - 1,
                    precision=0,
                )

            gr.Markdown("## Visualization Settings")
            with gr.Group():
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
                affinity_method_input = gr.Dropdown(
                    choices=list(self._AFFINITY_METHODS.keys()),
                    value="Attention Weights",
                    label="Affinity Method",
                )
                num_eigenvectors_input = gr.Number(
                    value=30,
                    label="Number of Eigenvectors for NCUT",
                    minimum=1,
                    maximum=1000,
                    precision=0,
                )

            generate_button = gr.Button(value="Generate")

            with gr.Row():
                clear_overlay_button = gr.Button(value="Clear Overlay")
                run_ncut_button = gr.Button(value="Run NCUT")

            figure_output = gr.Image(
                value=None, label="Visualization", type="pil", interactive=False
            )

            # Show/hide prompt inputs based on the number of images to generate.
            def on_num_images_change(value: str) -> list[dict[str, Any]]:
                num_images = int(value)
                updates = []
                for i in range(3):
                    if i < num_images:
                        updates.append(gr.update(visible=True))
                    else:
                        updates.append(gr.update(visible=False, value=""))
                return updates

            num_images_input.change(
                on_num_images_change, inputs=[num_images_input], outputs=prompt_inputs
            )

            # Update the inference step to visualize based on the number of inference steps.
            num_inference_steps_input.change(
                lambda inference_step, num_inference_steps: gr.update(
                    value=min(inference_step, num_inference_steps - 1),
                    maximum=num_inference_steps - 1,
                ),
                inputs=[inference_step_input, num_inference_steps_input],
                outputs=inference_step_input,
            )

            buttons = [generate_button, clear_overlay_button, run_ncut_button]

            def disable_buttons() -> list[dict[str, Any]]:
                return [gr.update(interactive=False) for _ in buttons]

            def enable_buttons() -> list[dict[str, Any]]:
                return [gr.update(interactive=True) for _ in buttons]

            def figure_output_select_start(
                event: gr.SelectData,
            ) -> list[dict[str, Any]]:
                self._figure_output_select_event = event
                return [gr.update(interactive=False) for _ in buttons]

            generate_button.click(disable_buttons, outputs=buttons).then(
                self.on_generate_button_click,
                inputs=[
                    num_images_input,
                    *prompt_inputs,
                    lora_weight_dir_input,
                    num_inference_steps_input,
                    guidance_scale_input,
                    random_seed_input,
                    inference_step_input,
                    transformer_blocks_input,
                    affinity_method_input,
                    num_eigenvectors_input,
                ],
                outputs=figure_output,
            ).then(enable_buttons, outputs=buttons)

            clear_overlay_button.click(disable_buttons, outputs=buttons).then(
                self.on_clear_overlay_button_click, outputs=figure_output
            ).then(enable_buttons, outputs=buttons)

            run_ncut_button.click(disable_buttons, outputs=buttons).then(
                self.on_run_ncut_button_click, outputs=figure_output
            ).then(enable_buttons, outputs=buttons)

            figure_output.select(figure_output_select_start, outputs=buttons).then(
                self.on_figure_output_select,
                inputs=[figure_output],
                outputs=figure_output,
            ).then(enable_buttons, outputs=buttons)

        demo.launch(share=self.args["share"])

    def on_generate_button_click(
        self,
        num_images_str: str,
        prompt_1: str,
        prompt_2: str,
        prompt_3: str,
        lora_weight_dir: str,
        num_inference_steps: int,
        guidance_scale: float,
        random_seed: int,
        inference_step: int,
        transformer_blocks_str: str,
        affinity_method: str,
        num_eigenvectors: int,
    ) -> Image.Image:
        """Callback when the generate button is clicked."""

        num_images = int(num_images_str)
        prompts = [
            prompt
            for i, prompt in enumerate((prompt_1, prompt_2, prompt_3))
            if i < num_images
        ]
        _logger.info("Run button clicked with prompts: %s", prompts)

        self.text_tokens = [
            get_prompt_tokens(self.pipeline, prompt) for prompt in prompts
        ]

        if transformer_blocks_str == "All":
            transformer_blocks = []
        else:
            # "Block {i}" -> i
            transformer_blocks = [int(transformer_blocks_str.split()[-1])]

        if lora_weight_dir != "None":
            self.pipeline.load_lora_weights(lora_weight_dir)
        self.generated_images, self.affinity = generate_image_and_get_affinity(
            self.pipeline,
            inference_step,
            transformer_blocks,
            prompts,
            num_inference_steps,
            guidance_scale,
            random_seed,
            self._AFFINITY_METHODS[affinity_method],
        )
        if lora_weight_dir != "None":
            self.pipeline.unload_lora_weights()

        self.ncut_eigenvectors = get_ncut_eigenvectors(
            self.affinity, num_eigenvectors, self.pipeline.device
        )

        return self.visualize_without_overlay()

    def on_clear_overlay_button_click(self) -> Image.Image | None:
        """Callback when the clear overlay button is clicked."""

        if self.affinity is None:
            return None
        _logger.info("Clear overlay button clicked")

        return self.visualize_without_overlay()

    def on_run_ncut_button_click(self) -> Image.Image | None:
        """Callback when the run NCUT button is clicked."""

        if self.affinity is None:
            return None
        _logger.info("Run NCUT button clicked")

        _, tsne_rgb = rgb_from_tsne_3d(
            self.ncut_eigenvectors, device=self.pipeline.device
        )
        ncut_colors = np.split(tsne_rgb.numpy(), len(self.text_tokens))

        visualizer_data = []
        for (clip_tokens, t5_tokens), example_ncut_colors in zip(
            self.text_tokens, ncut_colors
        ):
            clip_token_colors = example_ncut_colors[4096 : 4096 + len(clip_tokens)]
            t5_token_colors = example_ncut_colors[
                4096 + 77 : 4096 + 77 + len(t5_tokens)
            ]
            overlay_colors = example_ncut_colors[0:4096].reshape(64, 64, 3)

            visualizer_data.append(
                VisualizerData(
                    clip_tokens,
                    clip_token_colors,
                    t5_tokens,
                    t5_token_colors,
                    image=None,
                    overlay_colors=overlay_colors,
                )
            )

        return self.visualizer.visualize_data(visualizer_data)

    def on_figure_output_select(self, figure: Image.Image) -> Image.Image:
        """Callback when the figure output is selected."""

        event = self._figure_output_select_event
        self._figure_output_select_event = None

        assert self.affinity is not None
        _logger.info("Figure output selected at %s", event.index)

        click_x, click_y = event.index
        subfigure_and_token_index = self.visualizer.get_subfigure_and_token_index(
            click_x, click_y
        )
        if subfigure_and_token_index is None:
            return figure
        subfigure_index, local_token_index = subfigure_and_token_index
        global_token_index = subfigure_index * (4096 + 77 + 256) + local_token_index
        affinities = self.affinity[global_token_index].chunk(len(self.text_tokens))

        visualizer_data = []
        for (clip_tokens, t5_tokens), generated_image, affinity in zip(
            self.text_tokens, self.generated_images, affinities
        ):
            clip_token_colors = self.affinity_to_heatmap_color(
                affinity[4096 : 4096 + len(clip_tokens)]
            )
            t5_token_colors = self.affinity_to_heatmap_color(
                affinity[4096 + 77 : 4096 + 77 + len(t5_tokens)]
            )
            overlay_colors = self.affinity_to_heatmap_color(
                affinity[0:4096], alpha=True
            ).reshape(64, 64, 4)
            visualizer_data.append(
                VisualizerData(
                    clip_tokens,
                    clip_token_colors,
                    t5_tokens,
                    t5_token_colors,
                    generated_image,
                    overlay_colors,
                )
            )

        return self.visualizer.visualize_data(visualizer_data)

    def visualize_without_overlay(self) -> Image.Image:
        """Visualizes the prompts and images without any overlay."""

        visualizer_data = []
        for (clip_tokens, t5_tokens), generated_image in zip(
            self.text_tokens, self.generated_images
        ):
            clip_token_colors = np.zeros((len(clip_tokens), 4), dtype=np.uint8)
            clip_token_colors[:, 3] = 16
            t5_token_colors = np.zeros((len(t5_tokens), 4), dtype=np.uint8)
            t5_token_colors[:, 3] = 16
            visualizer_data.append(
                VisualizerData(
                    clip_tokens,
                    clip_token_colors,
                    t5_tokens,
                    t5_token_colors,
                    generated_image,
                    overlay_colors=None,
                )
            )

        return self.visualizer.visualize_data(visualizer_data)

    @staticmethod
    def affinity_to_heatmap_color(
        affinity_values: torch.Tensor, alpha: bool = False
    ) -> np.ndarray:
        """Converts affinity values to heatmap colors."""

        affinity_values = affinity_values.numpy()
        affinity_values -= affinity_values.min()
        affinity_values /= affinity_values.max() + 1e-6
        colors = plt.get_cmap("plasma")(affinity_values)
        if alpha:
            colors[:, 3] = affinity_values * 0.6 + 0.4  # 0.6 <= alpha <= 1.0
        return colors
