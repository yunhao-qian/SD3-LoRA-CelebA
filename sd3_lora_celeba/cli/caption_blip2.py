"""The `caption-blip2` subcommand."""

import logging
from pathlib import Path
from typing import TypedDict

import click
import torch
from tqdm.auto import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from ..dataset import ImageDataset

_logger = logging.getLogger(__name__)


class CaptionBLIP2Args(TypedDict):
    """Arguments to the `caption-blip2` subcommand."""

    dataset_dir: Path
    overwrite: bool
    caption_filename: str
    model_name: str
    model_revision: str
    precision: str
    compile_model: bool
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
@click.option("--caption-filename", default="prompt_blip2.txt")
@click.option("--model-name", default="Salesforce/blip2-flan-t5-xl-coco")
@click.option("--model-revision", default="main")
@click.option(
    "--precision", type=click.Choice(["float32", "float16", "8bit"]), default="float32"
)
@click.option("--compile-model/--no-compile-model", default=False)
@click.option("--batch-size", type=int, default=1)
@click.option("--num-workers", type=int, default=1)
def caption_blip2(**kwargs: CaptionBLIP2Args) -> None:
    """Caption images using a BLIP-2 model."""

    _logger.info("Arguments to caption-blip2: %s", kwargs)
    CaptionBLIP2(kwargs).run()


class CaptionBLIP2:
    """Implementation of the `caption-blip2` subcommand."""

    def __init__(self, args: CaptionBLIP2Args) -> None:
        self.args = args

        self.input_dtype: torch.dtype | None = None
        self.processor: Blip2Processor | None = None
        self.model: Blip2ForConditionalGeneration | None = None

    def run(self) -> None:
        """Run the subcommand."""

        self.load_model()

        dataset = ImageDataset(
            self.args["dataset_dir"],
            skip_filename=(
                None if self.args["overwrite"] else self.args["caption_filename"]
            ),
            transform=None,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            collate_fn=ImageDataset.collate,
            pin_memory=self.model.device.type == "cuda",
        )

        progress = tqdm(desc="Captioning images", total=len(dataset))
        for examples in dataloader:
            self.process_batch(examples)
            progress.update(len(examples["example_dirs"]))

    def load_model(self) -> None:
        """Load the BLIP-2 processor and model."""

        match self.args["precision"]:
            case "float32":
                model_args = {}
                self.input_dtype = torch.float32
            case "float16":
                model_args = {"torch_dtype": torch.float16}
                self.input_dtype = torch.float16
            case "8bit":
                model_args = {
                    "quantization_config": {"load_in_8bit": True},
                    "torch_dtype": torch.float16,
                }
                self.input_dtype = torch.float16
            case _:
                raise ValueError(f"Unexpected precision: {self.args['precision']}")

        _logger.info(
            "Loading model '%s', revision '%s'",
            self.args["model_name"],
            self.args["model_revision"],
        )

        self.processor = Blip2Processor.from_pretrained(
            self.args["model_name"], revision=self.args["model_revision"]
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.args["model_name"],
            revision=self.args["model_revision"],
            **model_args,
            device_map={"": 0},
        )

        if self.args["compile_model"]:
            _logger.info("Compiling the model")
            self.model = torch.compile(self.model, fullgraph=True, mode="max-autotune")

    def process_batch(self, examples: ImageDataset.ExampleBatch) -> None:
        """Caption a batch of examples and save the results."""

        inputs = self.processor(examples["images"], return_tensors="pt").to(
            self.model.device, self.input_dtype
        )

        generated_ids = self.model.generate(**inputs, max_length=100)
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        for example_dir, generated_text in zip(
            examples["example_dirs"], generated_texts
        ):
            generated_text = generated_text.strip()

            if len(generated_text) > 100:
                _logger.warning(
                    "The image caption for '%s' has %d characters. The model probably "
                    "had neural text degeneration.",
                    example_dir,
                    len(generated_text),
                )

            (example_dir / self.args["caption_filename"]).write_text(
                generated_text, encoding="utf-8"
            )
