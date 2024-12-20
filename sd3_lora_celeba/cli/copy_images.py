"""The `copy-images` subcommand."""

import logging
import shutil
from pathlib import Path
from typing import TypedDict

import click
from tqdm.auto import tqdm

_logger = logging.getLogger(__name__)


class CopyImagesArgs(TypedDict):
    """Arguments to the `copy-images` subcommand."""

    dataset_dir: Path
    output_dir: Path
    symlink: bool


@click.command()
@click.argument(
    "dataset_dir",
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
    "--symlink/--no-symlink", default=False, help="create symlinks instead of copying"
)
def copy_images(**kwargs: CopyImagesArgs) -> None:
    """Copy images from one directory to another.

    DATASET_DIR is the source directory containing the images to copy.

    OUTPUT_DIR is the destination directory to copy the images to.
    """

    _logger.info("Arguments to copy-images: %s", kwargs)
    CopyImages(kwargs).run()


class CopyImages:
    """Implementation of the `copy-images` subcommand."""

    def __init__(self, args: CopyImagesArgs) -> None:
        self.args = args

    def run(self) -> None:
        """Runs the subcommand."""

        self.args["output_dir"].mkdir(exist_ok=True)

        _logger.info("Searching for examples in '%s'", self.args["dataset_dir"])

        example_dirs: list[Path] = []
        for image_path_file_path in self.args["dataset_dir"].rglob("image_path.txt"):
            example_dir = image_path_file_path.parent
            example_dirs.append(example_dir)

        _logger.info("Found %d examples to copy", len(example_dirs))

        for example_dir in tqdm(example_dirs, desc="Copying images"):
            self.copy_image(example_dir)

    def copy_image(self, example_dir: Path) -> None:
        """Copies an image from the example directory."""

        input_image_path = (
            example_dir
            / (example_dir / "image_path.txt").read_text(encoding="utf-8").strip()
        )
        output_image_path = self.args["output_dir"] / input_image_path.name

        if self.args["symlink"]:
            output_image_path.symlink_to(input_image_path)
        else:
            shutil.copy(input_image_path, output_image_path)
