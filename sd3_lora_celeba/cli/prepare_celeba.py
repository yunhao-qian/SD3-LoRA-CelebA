"""The `prepare-celeba` subcommand."""

import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import TypedDict

import click
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from ..celeba import CelebAAttributeData, read_attribute_file

_logger = logging.getLogger(__name__)


class PrepareCelebAArgs(TypedDict):
    """Arguments to the `prepare-celeba` subcommand."""

    celeba_dir: Path
    celeba_mask_hq_dir: Path
    output_dir: Path
    min_image_size: int
    num_workers: int


@click.command()
@click.argument(
    "celeba_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
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
    "--min-image-size",
    type=int,
    default=768,
    help="minimum image size to keep an CelebA image",
)
@click.option(
    "--num-workers",
    type=int,
    default=1,
    help="number of workers for processing examples",
)
def prepare_celeba(**kwargs: PrepareCelebAArgs) -> None:
    """Prepare examples in the CelebA dataset that are not in CelebA-HQ and have
    sufficient image resolution.

    CELEBA_DIR is the directory containing the CelebA dataset.

    CELEBA_MASK_HQ_DIR is the directory containing the CelebA-HQ dataset.

    OUTPUT_DIR is the directory to write the prepared dataset to.
    """

    _logger.info("Arguments to prepare-celeba: %s", kwargs)
    PrepareCelebA(kwargs).run()


class PrepareCelebA:
    """Implementation of the `prepare-celeba` subcommand."""

    def __init__(self, args: PrepareCelebAArgs) -> None:
        self.args = args

        self.attribute_data: CelebAAttributeData | None = None
        self.bbox_data: dict[str, tuple[int, int, int, int]] | None = None

    def run(self) -> None:
        """Runs the subcommand."""

        self.args["output_dir"].mkdir(exist_ok=True)

        self.read_attribute_file()
        filenames_to_remove = self.read_mapping_file()
        self.remove_examples_by_filename(filenames_to_remove)
        self.read_bbox_file()

        num_passed = 0
        num_failed = 0
        num_examples = len(self.attribute_data["image_filenames"])
        with Pool(self.args["num_workers"], _initialize_worker, (self,)) as pool:
            progress = tqdm(
                pool.imap_unordered(_process_example, range(num_examples)),
                desc="Processing examples",
                total=num_examples,
            )
            for passed in progress:
                if passed:
                    num_passed += 1
                else:
                    num_failed += 1
                progress.set_postfix(passed=num_passed, failed=num_failed)
        _logger.info(
            "Removed %d examples with insufficient resolution, remaining %d examples.",
            num_failed,
            num_passed,
        )

    def read_attribute_file(self) -> None:
        """Reads the attribute file."""

        attribute_file_path = self.args["celeba_dir"] / "Anno/list_attr_celeba.txt"
        _logger.info("Reading attribute file: %s", attribute_file_path)

        with attribute_file_path.open(encoding="utf-8") as file:
            self.attribute_data = read_attribute_file(file)

    def read_mapping_file(self) -> list[str]:
        """From the mapping file, reads the filenames in CelebA that are in
        CelebA-HQ."""

        mapping_file_path = (
            self.args["celeba_mask_hq_dir"] / "CelebA-HQ-to-CelebA-mapping.txt"
        )
        _logger.info("Reading mapping file: %s", mapping_file_path)

        filenames: list[str] = []
        with mapping_file_path.open(encoding="utf-8") as file:
            column_names = file.readline().strip().split()
            assert column_names == ["idx", "orig_idx", "orig_file"]

            for line in file:
                _, _, orig_file = line.strip().split()
                filenames.append(orig_file)

        return filenames

    def remove_examples_by_filename(self, filenames_to_remove: list[str]) -> None:
        """Removes examples from `attribute_data` that are in CelebA-HQ."""

        old_num_examples = len(self.attribute_data["image_filenames"])

        filename_to_index = {
            filename: index
            for index, filename in enumerate(self.attribute_data["image_filenames"])
        }
        example_mask = np.ones(old_num_examples, dtype=bool)
        for filename in filenames_to_remove:
            example_mask[filename_to_index[filename]] = False
        self.attribute_data["image_filenames"] = [
            filename
            for filename, mask in zip(
                self.attribute_data["image_filenames"], example_mask
            )
            if mask
        ]
        self.attribute_data["attribute_values"] = self.attribute_data[
            "attribute_values"
        ][example_mask]

        new_num_examples = len(self.attribute_data["image_filenames"])
        _logger.info(
            "Removed %d examples in CelebA-HQ, remaining %d examples.",
            old_num_examples - new_num_examples,
            new_num_examples,
        )

    def read_bbox_file(self) -> None:
        """Reads the file of face bounding boxes."""

        bbox_file_path = self.args["celeba_dir"] / "Anno/list_bbox_celeba.txt"
        _logger.info("Reading bounding box file: %s", bbox_file_path)

        self.bbox_data = {}
        with bbox_file_path.open(encoding="utf-8") as file:
            num_examples = int(file.readline())

            column_names = file.readline().strip().split()
            assert column_names == ["image_id", "x_1", "y_1", "width", "height"]

            for line in file:
                filename, x1, y1, width, height = line.strip().split()
                self.bbox_data[filename] = (int(x1), int(y1), int(width), int(height))

        assert len(self.bbox_data) == num_examples

    def process_example(self, example_index: int) -> bool:
        """Processes an example, and if it passes the criteria, writes it to the output
        directory."""

        input_image_filename = self.attribute_data["image_filenames"][example_index]

        # Load and crop the image.
        with Image.open(
            self.args["celeba_dir"] / "Img/img_celeba" / input_image_filename
        ) as image_file:
            if min(image_file.size) < self.args["min_image_size"]:
                return False
            image = image_file.convert("RGB")
        image = image.crop(
            self.compute_crop_box(image.size, self.bbox_data[input_image_filename])
        )

        # Save the cropped image and its attributes.

        image_filename_stem = Path(input_image_filename).stem

        example_dir = self.args["output_dir"] / "train/CelebA" / image_filename_stem
        example_dir.mkdir(exist_ok=True, parents=True)

        output_image_filename = f"{image_filename_stem}.png"
        image.save(example_dir / output_image_filename)

        (example_dir / "image_path.txt").write_text(
            output_image_filename, encoding="utf-8"
        )

        attributes = {
            name: bool(value)
            for name, value in zip(
                self.attribute_data["attribute_names"],
                self.attribute_data["attribute_values"][example_index],
            )
        }
        with (example_dir / "attributes.json").open(mode="w", encoding="utf-8") as file:
            json.dump(attributes, file, indent=4)

        return True

    @staticmethod
    def compute_crop_box(
        image_size: tuple[int, int], bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Crops the image, centered around the face if possible."""

        image_width, image_height = image_size
        bbox_x1, bbox_y1, bbox_width, bbox_height = bbox

        image_min_size = min(image_size)

        x_min = bbox_x1 - (image_min_size - bbox_width) // 2
        x_min = max(0, min(image_width - image_min_size, x_min))
        x_max = x_min + image_min_size

        y_min = bbox_y1 - (image_min_size - bbox_height) // 2
        y_min = max(0, min(image_height - image_min_size, y_min))
        y_max = y_min + image_min_size

        return x_min, y_min, x_max, y_max


_worker_context: PrepareCelebA | None = None


def _initialize_worker(context: PrepareCelebA) -> None:
    """Initializes the worker process."""

    global _worker_context
    _worker_context = context


def _process_example(example_index: int) -> bool:
    """Processes an example in the worker process."""

    return _worker_context.process_example(example_index)
