"""Functions for processing the CelebA and CelebA-HQ datasets."""

import json
import logging
from io import TextIOBase
from pathlib import Path
from typing import TypedDict

import numpy as np
from rich.table import Table

_logger = logging.getLogger(__name__)


class CelebAAttributeData(TypedDict):
    """Attribute data in the CelebA or CelebA-HQ dataset."""

    attribute_names: list[str]
    image_filenames: list[str]
    attribute_values: np.ndarray


def read_attribute_file(file: TextIOBase) -> CelebAAttributeData:
    """Read data from an attribute file in the CelebA or CelebA-HQ dataset."""

    num_examples = int(file.readline())
    _logger.info("Number of examples: %d", num_examples)

    attribute_names = file.readline().strip().split()
    _logger.info("Attribute names: %s", attribute_names)

    image_filenames: list[str] = []
    attribute_values = np.empty((num_examples, len(attribute_names)), dtype=bool)

    for example_index, line in enumerate(file):
        parts = line.strip().split()
        assert len(parts) == len(attribute_names) + 1
        image_filenames.append(parts.pop(0))
        for attribute_index, part in enumerate(parts):
            attribute_values[example_index, attribute_index] = {"1": True, "-1": False}[
                part
            ]

    assert len(image_filenames) == num_examples

    return {
        "attribute_names": attribute_names,
        "image_filenames": image_filenames,
        "attribute_values": attribute_values,
    }


def read_attribute_json_files(example_dirs: list[Path]) -> tuple[list[str], np.ndarray]:
    """Read CelebA attribute data from JSON files."""

    _logger.info("Reading attribute JSON files from %d examples", len(example_dirs))

    attribute_names: list[str] | None = None
    attribute_values: np.ndarray | None = None

    for example_index, example_dir in enumerate(example_dirs):
        with (example_dir / "attributes.json").open(encoding="utf-8") as file:
            example_attributes = json.load(file)

        if example_index == 0:
            attribute_names = list(example_attributes.keys())
            attribute_values = np.empty(
                (len(example_dirs), len(attribute_names)), dtype=bool
            )

        for attribute_index, attribute_name in enumerate(attribute_names):
            attribute_values[example_index, attribute_index] = example_attributes[
                attribute_name
            ]

    return attribute_names, attribute_values


def create_attribute_frequency_table(
    title: str, attribute_names: list[str], attribute_values: dict[str, np.ndarray]
) -> Table:
    """Create a table of attribute frequencies."""

    table = Table(title=title)
    table.add_column("Attribute")
    for column_name in attribute_values:
        table.add_column(column_name)
    for attribute_index, attribute_name in enumerate(attribute_names):
        frequencies = (
            f"{column_values[:, attribute_index].mean():.4f}"
            for column_values in attribute_values.values()
        )
        table.add_row(attribute_name, *frequencies)
    return table
