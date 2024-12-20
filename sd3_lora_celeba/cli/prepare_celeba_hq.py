"""The `prepare-celeba-hq` subcommand."""

import json
import logging
from pathlib import Path
from typing import TypedDict

import click
import numpy as np
import rich
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from ..celeba import (
    CelebAAttributeData,
    create_attribute_frequency_table,
    read_attribute_file,
)

_logger = logging.getLogger(__name__)


class PrepareCelebAHQArgs(TypedDict):
    """Arguments to the `prepare-celeba-hq` subcommand."""

    celeba_mask_hq_dir: Path
    output_dir: Path
    relative_path: bool
    random_seed: int


@click.command()
@click.argument(
    "celeba_mask_hq_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument(
    "output_dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
)
@click.option(
    "--relative-path/--no-relative-path",
    default=False,
    help="use relative paths in image_path.txt files",
)
@click.option(
    "--random-seed", type=int, default=42, help="random seed for splitting the dataset"
)
def prepare_celeba_hq(**kwargs: PrepareCelebAHQArgs) -> None:
    """Prepare images and boolean attributes in the CelebA-HQ dataset.

    CELEBA_MASK_HQ_DIR is the directory containing the CelebA-HQ dataset.

    OUTPUT_DIR is the directory to write the prepared dataset to.
    """

    _logger.info("Arguments to prepare-celeba-hq: %s", kwargs)
    PrepareCelebAHQ(kwargs).run()


class PrepareCelebAHQ:
    """Implementation of the `prepare-celeba-hq` subcommand."""

    def __init__(self, args: PrepareCelebAHQArgs) -> None:
        self.args = args

        self.attribute_data: CelebAAttributeData | None = None
        self.split_indices: dict[str, np.ndarray] | None = None

    def run(self) -> None:
        """Runs the subcommand."""

        self.args["output_dir"].mkdir(exist_ok=True)
        self.read_attribute_file()
        self.split_data()
        self.print_attribute_frequencies()

        progress = tqdm(
            desc="Writing examples", total=len(self.attribute_data["image_filenames"])
        )
        for split in "train", "val", "test":
            split_dir = self.args["output_dir"] / split / "CelebA-HQ"
            split_dir.mkdir(exist_ok=True)
            for example_index in self.split_indices[split]:
                self.write_example(split_dir, example_index)
                progress.update()

    def read_attribute_file(self) -> None:
        """Reads the attribute file."""

        attribute_file_path = (
            self.args["celeba_mask_hq_dir"] / "CelebAMask-HQ-attribute-anno.txt"
        )
        _logger.info("Reading attribute file: %s", attribute_file_path)
        with attribute_file_path.open(encoding="utf-8") as file:
            self.attribute_data = read_attribute_file(file)

    def split_data(self) -> None:
        """Splits the examples into training, validation, and test sets."""

        # Use only the following attributes for stratified sampling. Adding more
        # attributes results in classes with a single example.
        attribute_indices = [
            self.attribute_data["attribute_names"].index(name)
            for name in ("Male", "Wearing_Lipstick", "Smiling")
        ]
        attribute_values = self.attribute_data["attribute_values"][:, attribute_indices]

        train_indices, val_test_indices = train_test_split(
            np.arange(attribute_values.shape[0]),
            test_size=0.2,
            random_state=self.args["random_seed"],
            stratify=attribute_values,
        )
        val_indices, test_indices = train_test_split(
            val_test_indices,
            test_size=0.5,
            random_state=self.args["random_seed"],
            stratify=attribute_values[val_test_indices],
        )
        _logger.info("Training set: %d examples", len(train_indices))
        _logger.info("Validation set: %d examples", len(val_indices))
        _logger.info("Test set: %d examples", len(test_indices))

        self.split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    def print_attribute_frequencies(self) -> None:
        """Prints the attribute frequencies of each split."""

        attribute_values: dict[str, np.ndarray] = {
            "All": self.attribute_data["attribute_values"]
        }
        for split, column_name in (
            ("train", "Training"),
            ("val", "Validation"),
            ("test", "Test"),
        ):
            attribute_values[column_name] = self.attribute_data["attribute_values"][
                self.split_indices[split]
            ]

        table = create_attribute_frequency_table(
            "Attribute Frequencies",
            self.attribute_data["attribute_names"],
            attribute_values,
        )
        rich.print(table)

    def write_example(self, split_dir: Path, example_index: int) -> None:
        """Writes an example to the output directory."""

        image_filename = self.attribute_data["image_filenames"][example_index]

        example_dir = split_dir / Path(image_filename).stem
        example_dir.mkdir(exist_ok=True)

        # Write image_path.txt.
        image_path = self.args["celeba_mask_hq_dir"] / "CelebA-HQ-img" / image_filename
        assert image_path.is_file(), f"'{image_path}' is not a file."
        if self.args["relative_path"]:
            image_path = image_path.relative_to(example_dir)
        else:
            image_path = image_path.resolve()
        (example_dir / "image_path.txt").write_text(str(image_path), encoding="utf-8")

        # Write attributes.json.
        attributes = {
            name: bool(value)
            for name, value in zip(
                self.attribute_data["attribute_names"],
                self.attribute_data["attribute_values"][example_index],
            )
        }
        with (example_dir / "attributes.json").open(mode="w", encoding="utf-8") as file:
            json.dump(attributes, file, indent=4)
