"""The `sd3` command line tool."""

import logging

import click
from rich.logging import RichHandler

from .caption_blip2 import caption_blip2
from .caption_llama3 import caption_llama3
from .compute_latent_dist import compute_latent_dist
from .compute_prompt_embeds import compute_prompt_embeds
from .copy_images import copy_images
from .fine_tune import fine_tune
from .generate_images import generate_images
from .prepare_celeba import prepare_celeba
from .prepare_celeba_hq import prepare_celeba_hq
from .visualize_token_affinities import visualize_token_affinities


@click.group()
@click.option("--quiet", is_flag=True, help="suppress non-warning log messages")
@click.option(
    "--logging-style",
    type=click.Choice(["rich", "plain"]),
    default="rich",
    help="use rich text or plain text for logging",
)
def sd3(quiet: bool, logging_style: str) -> None:
    """Stable Diffusion 3 fine-tuning and feature visualization."""
    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        level=logging.WARNING if quiet else logging.INFO,
        **({"handlers": [RichHandler()]} if logging_style == "rich" else {}),
    )


sd3.add_command(prepare_celeba_hq)
sd3.add_command(prepare_celeba)
sd3.add_command(caption_blip2)
sd3.add_command(caption_llama3)
sd3.add_command(compute_latent_dist)
sd3.add_command(compute_prompt_embeds)
sd3.add_command(fine_tune)
sd3.add_command(copy_images)
sd3.add_command(generate_images)
sd3.add_command(visualize_token_affinities)
