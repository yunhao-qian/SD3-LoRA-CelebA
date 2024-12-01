"""The `compute-prompt-embeds` subcommand."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypedDict, TypeVar

import click
import more_itertools
import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

_logger = logging.getLogger(__name__)


class ComputePromptEmbedsArgs(TypedDict):
    """Arguments to the `compute-prompt-embeds` subcommand."""

    dataset_dir: Path
    text_encoder_id: str
    overwrite: bool
    prompt_embed_filename: str | None
    model_name: str
    model_revision: str
    compile_model: bool
    batch_size: int


@click.command()
@click.argument(
    "dataset_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument("text_encoder_id", type=click.Choice(["1", "2", "3"]))
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--prompt-embed-filename", default=None)
@click.option("--model-name", default="stabilityai/stable-diffusion-3-medium-diffusers")
@click.option("--model-revision", default="main")
@click.option("--compile-model/--no-compile-model", default=False)
@click.option("--batch-size", type=int, default=1)
def compute_prompt_embeds(**kwargs: ComputePromptEmbedsArgs) -> None:
    """Compute and save the text embeddings of prompts."""

    _logger.info("Arguments to compute-prompt-embeds: %s", kwargs)
    ComputePromptEmbeds(kwargs).run()


class ComputePromptEmbeds:
    """Implementation of the `compute-prompt-embeds` subcommand."""

    def __init__(self, args: ComputePromptEmbedsArgs) -> None:
        self.args = args

        self.example_dirs: list[Path] | None = None
        self.model: ModelBase | None = None

    def run(self) -> None:
        """Run the subcommand."""

        self.infer_prompt_embed_filename()
        self.find_example_dirs_to_process()
        self.load_model()

        progress = tqdm(
            desc="Computing prompt embeddings", total=len(self.example_dirs)
        )
        for example_dirs in more_itertools.chunked(
            self.example_dirs, self.args["batch_size"]
        ):
            prompt_embeds = self.model.embed_example_dirs(example_dirs)
            for example_dir, example_prompt_embeds in zip(example_dirs, prompt_embeds):
                save_file(
                    example_prompt_embeds,
                    example_dir / self.args["prompt_embed_filename"],
                )
            progress.update(len(example_dirs))

    def infer_prompt_embed_filename(self) -> None:
        """Infer the prompt embedding filename."""

        if self.args["prompt_embed_filename"] is None:
            self.args["prompt_embed_filename"] = (
                f"prompt_embeds_{self.args['text_encoder_id']}.safetensors"
            )
        _logger.info(
            "Using the prompt embedding filename '%s'",
            self.args["prompt_embed_filename"],
        )

    def find_example_dirs_to_process(self) -> None:
        """Find the example directories to process."""

        _logger.info("Searching for examples in '%s'", self.args["dataset_dir"])

        example_dirs: set[Path] = set()
        num_skipped = 0
        for prompt_file_path in self.args["dataset_dir"].rglob("prompt_*.txt"):
            example_dir = prompt_file_path.parent
            if example_dir in example_dirs:
                continue
            if (
                not self.args["overwrite"]
                and (example_dir / self.args["prompt_embed_filename"]).exists()
            ):
                num_skipped += 1
                continue
            example_dirs.add(example_dir)

        if num_skipped > 0:
            _logger.info(
                "Skipped %d examples with existing prompt embedding files", num_skipped
            )
        _logger.info("Found %d examples to process", len(example_dirs))

        self.example_dirs = list(example_dirs)

    def load_model(self) -> None:
        """Load the text encoder and tokenizer."""

        _logger.info(
            "Loading model '%s', revision '%s'",
            self.args["model_name"],
            self.args["model_revision"],
        )

        match self.args["text_encoder_id"]:
            case "1":
                self.model = CLIPModel(
                    text_encoder=CLIPTextModelWithProjection.from_pretrained(
                        self.args["model_name"],
                        revision=self.args["model_revision"],
                        subfolder="text_encoder",
                        device_map={"": 0},
                    ).requires_grad_(False),
                    tokenizer=CLIPTokenizer.from_pretrained(
                        self.args["model_name"],
                        revision=self.args["model_revision"],
                        subfolder="tokenizer",
                    ),
                    compile_model=self.args["compile_model"],
                )
            case "2":
                self.model = CLIPModel(
                    text_encoder=CLIPTextModelWithProjection.from_pretrained(
                        self.args["model_name"],
                        revision=self.args["model_revision"],
                        subfolder="text_encoder_2",
                        device_map={"": 0},
                    ).requires_grad_(False),
                    tokenizer=CLIPTokenizer.from_pretrained(
                        self.args["model_name"],
                        revision=self.args["model_revision"],
                        subfolder="tokenizer_2",
                    ),
                    compile_model=self.args["compile_model"],
                )
            case "3":
                self.model = T5Model(
                    text_encoder=T5EncoderModel.from_pretrained(
                        self.args["model_name"],
                        revision=self.args["model_revision"],
                        subfolder="text_encoder_3",
                        device_map={"": 0},
                    ).requires_grad_(False),
                    tokenizer=T5TokenizerFast.from_pretrained(
                        self.args["model_name"],
                        revision=self.args["model_revision"],
                        subfolder="tokenizer_3",
                    ),
                    compile_model=self.args["compile_model"],
                )
            case _:
                raise ValueError(
                    f"Invalid text encoder ID: {self.args['text_encoder_id']}"
                )


TextEncoderClass = TypeVar("TextEncoderClass")
TokenizerClass = TypeVar("TokenizerClass")


class ModelBase(ABC, Generic[TextEncoderClass, TokenizerClass]):
    """Base class for the text encoder and tokenizer of a model."""

    MAX_PROMPT_LENGTH = 77

    def __init__(
        self,
        text_encoder: TextEncoderClass,
        tokenizer: TokenizerClass,
        compile_model: bool,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        if compile_model:
            _logger.info("Compiling the text encoder")
            self.text_encoder = torch.compile(
                self.text_encoder, fullgraph=True, mode="max-autotune"
            )

    def embed_example_dirs(
        self, example_dirs: list[Path]
    ) -> list[dict[str, torch.Tensor]]:
        """Embed the prompts of the given example directories."""

        prompt_example_dirs: list[Path] = []
        prompt_variants: list[str] = []
        prompt_texts: list[str] = []

        for example_dir in example_dirs:
            for prompt_file_path in example_dir.glob("prompt_*.txt"):
                prompt_example_dirs.append(example_dir)
                prompt_variants.append(prompt_file_path.stem.split("_", maxsplit=1)[1])
                prompt_texts.append(
                    prompt_file_path.read_text(encoding="utf-8").strip()
                )

        prompt_embed_dict = self.embed_texts(prompt_texts)

        results: list[dict[str, torch.Tensor]] = [{} for _ in example_dirs]
        for prompt_index, example_dir in enumerate(prompt_example_dirs):
            example_index = example_dirs.index(example_dir)
            variant = prompt_variants[prompt_index]
            for key, value in prompt_embed_dict.items():
                results[example_index][f"{variant}.{key}"] = value[prompt_index]

        return results

    @abstractmethod
    def embed_texts(self, _: list[str]) -> dict[str, torch.Tensor]:
        """Compute the embeddings of the given texts."""


class CLIPModel(ModelBase[CLIPTextModelWithProjection, CLIPTokenizer]):
    """Text encoder and tokenizer of a CLIP model."""

    def embed_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:

        text_input_ids = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.MAX_PROMPT_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.text_encoder.device)

        outputs = self.text_encoder(text_input_ids, output_hidden_states=True)
        return {
            "prompt_embeds": outputs.hidden_states[-2].cpu(),
            "pooled_prompt_embeds": outputs.text_embeds.cpu(),
        }


class T5Model(ModelBase[T5EncoderModel, T5TokenizerFast]):
    """Text encoder and tokenizer of a T5 model."""

    def embed_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:

        text_input_ids = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.MAX_PROMPT_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.to(self.text_encoder.device)

        prompt_embeds = self.text_encoder(text_input_ids).last_hidden_state.cpu()
        return {"prompt_embeds": prompt_embeds}
