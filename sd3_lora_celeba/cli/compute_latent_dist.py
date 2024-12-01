"""The `compute-latent-dist` subcommand."""

import logging
from pathlib import Path
from typing import TypedDict

import click
import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

from ..dataset import ImageDataset

_logger = logging.getLogger(__name__)


class ComputeLatentDistArgs(TypedDict):
    """Arguments to the `compute-latent-dist` subcommand."""

    dataset_dir: Path
    overwrite: bool
    latent_dist_filename: str
    model_name: str
    model_revision: str
    compile_model: bool
    image_size: int
    batch_size: int
    num_workers: int


@click.command()
@click.argument(
    "dataset_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--latent-dist-filename", default="latent_dist.safetensors")
@click.option("--model-name", default="stabilityai/stable-diffusion-3-medium-diffusers")
@click.option("--model-revision", default="main")
@click.option("--compile-model/--no-compile-model", default=False)
@click.option("--image-size", type=int, default=1024)
@click.option("--batch-size", type=int, default=1)
@click.option("--num-workers", type=int, default=1)
def compute_latent_dist(**kwargs: ComputeLatentDistArgs) -> None:
    """Compute and save the latent distributions of images."""

    _logger.info("Arguments to compute-latent-dist: %s", kwargs)
    ComputeLatentDist(kwargs).run()


class ComputeLatentDist:
    """Implementation of the `compute-latent-dist` subcommand."""

    def __init__(self, args: ComputeLatentDistArgs) -> None:
        self.args = args

        self.vae: AutoencoderKL | None = None

    def run(self) -> None:
        """Run the subcommand."""

        self.load_model()

        dataset = self.create_dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            collate_fn=ImageDataset.collate,
            pin_memory=self.vae.device.type == "cuda",
        )

        progress = tqdm(desc="Computing latent distributions", total=len(dataset))
        for examples in dataloader:
            self.process_batch(examples)
            progress.update(len(examples["example_dirs"]))

    def load_model(self) -> None:
        """Load the VAE encoder model."""

        _logger.info(
            "Loading model '%s', revision '%s'",
            self.args["model_name"],
            self.args["model_revision"],
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.args["model_name"],
            revision=self.args["model_revision"],
            subfolder="vae",
            device_map={"": 0},
        ).requires_grad_(False)

        if self.args["compile_model"]:
            _logger.info("Compiling the VAE encoder")
            self.vae.encoder = torch.compile(
                self.vae.encoder, fullgraph=True, mode="max-autotune"
            )

    def create_dataset(self) -> ImageDataset:
        """Create the dataset of images to process."""

        transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Resize((self.args["image_size"], self.args["image_size"])),
            ]
        )
        return ImageDataset(
            self.args["dataset_dir"],
            skip_filename=(
                None if self.args["overwrite"] else self.args["latent_dist_filename"]
            ),
            transform=transform,
        )

    def process_batch(self, examples: ImageDataset.ExampleBatch) -> None:
        """Compute and save the latent distributions for a batch of images."""

        pixel_values = examples["pixel_values"].to(self.vae.device)

        # The internal implementation of encode() processes images sequentially, so
        # there is no real benefit in batching here.

        latent_dists_original = self.vae.encode(
            pixel_values
        ).latent_dist.parameters.cpu()
        latent_dists_flipped = self.vae.encode(
            pixel_values.flip([-1])
        ).latent_dist.parameters.cpu()

        for example_dir, latent_dist_original, latent_dist_flipped in zip(
            examples["example_dirs"], latent_dists_original, latent_dists_flipped
        ):
            save_file(
                {
                    "shift_factor": torch.tensor(
                        self.vae.config.shift_factor, dtype=torch.float32
                    ),
                    "scaling_factor": torch.tensor(
                        self.vae.config.scaling_factor, dtype=torch.float32
                    ),
                    "original": latent_dist_original,
                    "flipped": latent_dist_flipped,
                },
                example_dir / self.args["latent_dist_filename"],
            )
