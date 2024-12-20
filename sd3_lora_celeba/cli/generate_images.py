"""The `generate-images` subcommand."""

import logging
from pathlib import Path
from typing import TypedDict

import click
import torch
from diffusers import StableDiffusion3Pipeline
from tqdm.auto import tqdm

from ..dataset import load_prompt_embeds

_logger = logging.getLogger(__name__)


class GenerateImagesArgs(TypedDict):
    """Arguments to the `generate-images` subcommand."""

    dataset_dir: Path
    empty_prompt_dir: Path
    prompt_variant: str
    output_dir: Path
    model_name: str
    model_revision: str
    lora_weight_dir: Path | None
    compile_model: bool
    batch_size: int
    num_workers: int
    num_inference_steps: int
    guidance_scale: float
    random_seed: int | None


@click.command()
@click.argument(
    "dataset_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument(
    "empty_prompt_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument("prompt_variant", type=click.Choice(["blip2", "llama3"]))
@click.argument(
    "output_dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
)
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
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    default=None,
    help="directory containing LoRA weights, if any",
)
@click.option(
    "--compile-model/--no-compile-model", default=False, help="compile the model"
)
@click.option(
    "--batch-size", type=int, default=1, help="batch size for generating images"
)
@click.option(
    "--num-workers", type=int, default=1, help="number of workers for data loading"
)
@click.option(
    "--num-inference-steps", type=int, default=28, help="number of inference steps"
)
@click.option(
    "--guidance-scale",
    type=float,
    default=7.0,
    help="guidance scale for classifier-free guidance",
)
@click.option(
    "--random-seed", type=int, default=None, help="random seed for generating images"
)
def generate_images(**kwargs: GenerateImagesArgs) -> None:
    """Generate images from prompt embeddings.

    DATASET_DIR is the directory containing the prompt embeddings.

    EMPTY_PROMPT_DIR is the directory containing the text embeddings of the empty
    prompt.

    PROMPT_VARIANT (one of 'blip2' or 'llama3') is the variant of prompts to use.

    OUTPUT_DIR is the directory to save the generated images to.
    """

    _logger.info("Arguments to generate-images: %s", kwargs)
    GenerateImages(kwargs).run()


class PromptDataset(torch.utils.data.Dataset):
    """Dataset for loading prompt embeddings."""

    class Example(TypedDict):
        """An example from PromptDataset."""

        name: str
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: torch.Tensor

    class ExampleBatch(TypedDict):
        """A batch of examples from PromptDataset."""

        name: list[str]
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: torch.Tensor

    def __init__(self, dataset_dir: Path, variant: str, dtype: torch.dtype) -> None:
        _logger.info("Searching for examples in '%s'", dataset_dir)
        self.prompt_dirs: list[Path] = []
        for prompt_embeds_file_path in dataset_dir.rglob("prompt_embeds_1.safetensors"):
            self.prompt_dirs.append(prompt_embeds_file_path.parent)
        _logger.info("Found %d examples in the dataset", len(self.prompt_dirs))

        self.variant = variant
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.prompt_dirs)

    def __getitem__(self, index: int) -> Example:
        prompt_dir = self.prompt_dirs[index]
        prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(
            prompt_dir, self.variant, self.dtype
        )
        return {
            "name": prompt_dir.name,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }


class GenerateImages:
    """Implementation of the `generate-images` subcommand."""

    def __init__(self, args: GenerateImagesArgs) -> None:
        self.args = args

        self.pipeline: StableDiffusion3Pipeline | None = None
        self.generator: torch.Generator | None = None
        self.empty_prompt_embeds: torch.Tensor | None = None
        self.empty_pooled_prompt_embeds: torch.Tensor | None = None

    def run(self) -> None:
        """Runs the subcommand."""

        self.args["output_dir"].mkdir(exist_ok=True)

        self.load_pipeline()
        self.generator = torch.Generator(self.pipeline.device)
        self.load_empty_prompt_embeds()

        dataset = PromptDataset(
            self.args["dataset_dir"], self.args["prompt_variant"], torch.float16
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            pin_memory=self.pipeline.device.type == "cuda",
        )

        progress = tqdm(desc="Generating images", total=len(dataset))
        for examples in dataloader:
            self.process_batch(examples)
            progress.update(len(examples["name"]))

    def load_pipeline(self) -> None:
        """Loads the Stable Diffusion 3 pipeline."""

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.args["model_name"],
            revision=self.args["model_revision"],
            torch_dtype=torch.float16,
            device_map="balanced",
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            text_encoder_3=None,
            tokenizer_3=None,
        )
        if self.args["lora_weight_dir"] is not None:
            self.pipeline.load_lora_weights(self.args["lora_weight_dir"])
        self.pipeline.transformer.requires_grad_(False)
        if self.args["compile_model"]:
            _logger.info("Compiling the transformer")
            self.pipeline.transformer = torch.compile(
                self.pipeline.transformer, fullgraph=True, mode="max-autotune"
            )
            _logger.info("Compiling the VAE decoder")
            self.pipeline.vae.decoder = torch.compile(
                self.pipeline.vae.decoder, fullgraph=True, mode="max-autotune"
            )

        # Turn off progress bars.
        self.pipeline.set_progress_bar_config(disable=True)

    def load_empty_prompt_embeds(self) -> None:
        """Loads the text embeddings of the empty prompt."""

        empty_prompt_embeds, empty_pooled_prompt_embeds = load_prompt_embeds(
            self.args["empty_prompt_dir"], "empty", torch.float16
        )
        self.empty_prompt_embeds = empty_prompt_embeds.to(self.pipeline.device)
        self.empty_pooled_prompt_embeds = empty_pooled_prompt_embeds.to(
            self.pipeline.device
        )

    def process_batch(self, examples: PromptDataset.ExampleBatch) -> None:
        """Generates images for a batch of examples."""

        prompt_embeds = examples["prompt_embeds"].to(self.pipeline.device)
        pooled_prompt_embeds = examples["pooled_prompt_embeds"].to(self.pipeline.device)

        if self.args["random_seed"] is not None:
            self.generator.manual_seed(self.args["random_seed"])

        images = self.pipeline(
            num_inference_steps=self.args["num_inference_steps"],
            guidance_scale=self.args["guidance_scale"],
            generator=self.generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=self.empty_prompt_embeds.expand(
                (prompt_embeds.size(0), -1, -1)
            ),
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=self.empty_pooled_prompt_embeds.expand(
                (prompt_embeds.size(0), -1)
            ),
        ).images

        for name, image in zip(examples["name"], images):
            image.save(self.args["output_dir"] / f"{name}.png")
