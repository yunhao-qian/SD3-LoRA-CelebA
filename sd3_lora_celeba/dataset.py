"""Shared PyTorch dataset classes."""

import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import cvxpy
import rich
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from safetensors import safe_open
from typing_extensions import NotRequired

from .celeba import create_attribute_frequency_table, read_attribute_json_files

_logger = logging.getLogger(__name__)


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for loading images to process."""

    class Example(TypedDict):
        """An example from ImageDataset."""

        example_dir: Path
        image: NotRequired[Image.Image]
        pixel_values: NotRequired[torch.Tensor]

    class ExampleBatch(TypedDict):
        """A batch of examples from ImageDataset."""

        example_dirs: list[Path]
        images: NotRequired[list[Image.Image]]
        pixel_values: NotRequired[torch.Tensor]

    def __init__(
        self,
        dataset_dir: Path,
        skip_filename: str | None,
        transform: Callable[[Image.Image], torch.Tensor] | None,
    ) -> None:
        _logger.info("Searching for examples in '%s'", dataset_dir)

        self.example_dirs: list[Path] = []
        num_skipped = 0
        for image_path_file_path in dataset_dir.rglob("image_path.txt"):
            example_dir = image_path_file_path.parent
            if skip_filename is not None and (example_dir / skip_filename).exists():
                num_skipped += 1
                continue
            self.example_dirs.append(example_dir)

        if num_skipped > 0:
            _logger.info("Skipped %d examples with existing files", num_skipped)
        _logger.info("Found %d examples to process", len(self.example_dirs))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.example_dirs)

    def __getitem__(self, index: int) -> Example:
        example_dir = self.example_dirs[index]
        image_path = (
            example_dir
            / (example_dir / "image_path.txt").read_text(encoding="utf-8").strip()
        )
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is None:
            return {"example_dir": example_dir, "image": image}
        pixel_values = self.transform(image)
        return {"example_dir": example_dir, "pixel_values": pixel_values}

    @staticmethod
    def collate(examples: list[Example]) -> ExampleBatch:
        """Collate a list of examples into a batch."""

        example_dirs: list[Path] = []
        images: list[Image.Image] = []
        pixel_values: list[torch.Tensor] = []

        for example in examples:
            example_dirs.append(example["example_dir"])
            if "image" in example:
                images.append(example["image"])
            else:
                pixel_values.append(example["pixel_values"])

        batch = {"example_dirs": example_dirs}
        if len(images) > 0:
            batch["images"] = images
        if len(pixel_values) > 0:
            batch["pixel_values"] = torch.stack(pixel_values)

        return batch


def load_prompt_embeds(
    prompt_dir: Path, variant: str | None, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads the prompt embeddings from a directory of safetensors files.

    Args:
        prompt_dir: The directory containing the prompt embeddings.
        variant: The variant of the prompt to load, or `None` to choose randomly.
        dtype: The data type to convert the prompt embeddings to.

    Returns:
        A tuple containing the prompt embeddings and pooled prompt embeddings.
    """

    def load_file(file_path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
        nonlocal variant  # Modify variant in the outer scope if it is `None`.

        with safe_open(file_path, framework="pt") as file:
            if variant is None:
                variants = set(key.split(".")[0] for key in file.keys())
                variant = random.choice(list(variants))

            prompt_embeds = file.get_tensor(f"{variant}.prompt_embeds").to(dtype)

            pooled_key = f"{variant}.pooled_prompt_embeds"
            if pooled_key in file.keys():
                pooled_prompt_embeds = file.get_tensor(pooled_key).to(dtype)
            else:
                pooled_prompt_embeds = None

        return prompt_embeds, pooled_prompt_embeds

    prompt_embeds_1, pooled_prompt_embeds_1 = load_file(
        prompt_dir / "prompt_embeds_1.safetensors"
    )
    prompt_embeds_2, pooled_prompt_embeds_2 = load_file(
        prompt_dir / "prompt_embeds_2.safetensors"
    )
    t5_prompt_embeds, _ = load_file(prompt_dir / "prompt_embeds_3.safetensors")

    clip_prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)
    pooled_prompt_embeds = torch.cat(
        (pooled_prompt_embeds_1, pooled_prompt_embeds_2), dim=-1
    )
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embeds.size(-1) - clip_prompt_embeds.size(-1)),
    )
    prompt_embeds = torch.cat((clip_prompt_embeds, t5_prompt_embeds), dim=-2)

    return prompt_embeds, pooled_prompt_embeds


class ImageAndPromptDataset(torch.utils.data.Dataset):
    """Dataset for loading latents of images and text embeddings of prompts."""

    class Example(TypedDict):
        """An example from ImageAndPromptDataset."""

        model_input: torch.Tensor
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: torch.Tensor

    ExampleBatch = Example
    """A batch of examples from ImageAndPromptDataset."""

    def __init__(
        self, dataset_dir: Path, empty_prompt_dir: Path, dtype: torch.dtype
    ) -> None:
        _logger.info("Searching for examples in '%s'", dataset_dir)
        self.example_dirs: list[Path] = []
        for latent_dist_file_path in dataset_dir.rglob("latent_dist.safetensors"):
            self.example_dirs.append(latent_dist_file_path.parent)
        _logger.info("Found %d examples in the dataset", len(self.example_dirs))

        self.empty_prompt_dir = empty_prompt_dir
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.example_dirs)

    def __getitem__(self, index: int) -> Example:
        example_dir = self.example_dirs[index]

        model_input = self.load_latent_dist_file(
            example_dir / "latent_dist.safetensors"
        )
        prompt_dir = self.empty_prompt_dir if random.random() < 0.2 else example_dir
        prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(
            prompt_dir, None, self.dtype
        )

        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    def load_latent_dist_file(self, file_path: Path) -> torch.Tensor:
        """Load the latent distribution from a file and sample from it."""

        with safe_open(file_path, framework="pt") as file:
            latent_dist = file.get_tensor(random.choice(["original", "flipped"]))
            shift_factor = file.get_tensor("shift_factor")
            scaling_factor = file.get_tensor("scaling_factor")
        model_input = (
            DiagonalGaussianDistribution(latent_dist.unsqueeze(0)).sample().squeeze(0)
        )
        model_input = ((model_input - shift_factor) * scaling_factor).to(self.dtype)
        return model_input

    def get_sampling_weights(self) -> list[float]:
        """Compute the sampling weights to balance attribute frequencies."""

        attribute_names, attribute_values = read_attribute_json_files(self.example_dirs)

        # Solve the optimization problem of balancing attribute frequencies.
        _logger.info("Optimizing sampling weights")

        x = cvxpy.Variable(len(self))
        # Cost 1: Attribute frequencies should be close to 0.5.
        attribute_frequency_cost = cvxpy.sum_squares(
            x @ attribute_values / len(self) - 0.5
        )
        # Cost 2: We want sum(x) == len(self), but this constraint is not convex, so we
        # approximate it with a penalty term.
        sum_constraint_cost = (cvxpy.mean(x) - 1) ** 2
        # Cost 3: Sampling weights should be close to 1.
        weight_cost = cvxpy.sum_squares(x - 1) / len(self)
        problem = cvxpy.Problem(
            cvxpy.Minimize(
                attribute_frequency_cost
                + sum_constraint_cost * 1000
                + weight_cost * 1.3
            ),
            [x >= 0.2, x <= 5],
        )
        problem.solve(solver=cvxpy.OSQP)

        _logger.info("Optimization status: %s", problem.status)
        sampling_weights = x.value
        sampling_weights /= sampling_weights.sum() / len(self)

        table = create_attribute_frequency_table(
            "Attribute Frequencies",
            attribute_names,
            {
                "Before": attribute_values,
                "After": attribute_values * sampling_weights[:, None],
            },
        )
        rich.print(table)

        return sampling_weights
