"""The `caption-llama3` subcommand."""

import json
import logging
from pathlib import Path
from typing import TypedDict

import click
import torch
import transformers
from tqdm.auto import tqdm
from transformers import Pipeline

_logger = logging.getLogger(__name__)


class CaptionLlama3Args(TypedDict):
    """Arguments to the `caption-llama3` subcommand."""

    dataset_dir: Path
    overwrite: bool
    caption_filename: str
    blip2_caption_filename: str
    model_name: str
    model_revision: str
    precision: str


@click.command()
@click.argument(
    "dataset_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.option(
    "--overwrite/--no-overwrite", default=False, help="overwrite existing caption files"
)
@click.option(
    "--caption-filename", default="prompt_llama3.txt", help="filename for captions"
)
@click.option(
    "--blip2-caption-filename",
    default="prompt_blip2.txt",
    help="filename for BLIP-2 captions to extend",
)
@click.option(
    "--model-name",
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Llama 3 model name",
)
@click.option("--model-revision", default="main", help="Llama 3 model revision")
@click.option(
    "--precision",
    type=click.Choice(["bfloat16", "float16", "8bit", "4bit"]),
    default="bfloat16",
    help="data type to run the model in",
)
def caption_llama3(**kwargs: CaptionLlama3Args) -> None:
    """Extend BLIP-2 captions with CelebA attributes using a Llama 3 model.

    DATASET_DIR is the directory containing the captions to process.
    """

    _logger.info("Arguments to caption-llama3: %s", kwargs)
    CaptionLlama3(kwargs).run()


class CaptionLlama3:
    """Implementation of the `caption-llama3` subcommand."""

    def __init__(self, args: CaptionLlama3Args) -> None:
        self.args = args

        self.example_dirs: list[Path] | None = None
        self.pipeline: Pipeline | None = None

    def run(self) -> None:
        """Runs the subcommand."""

        self.find_example_dirs_to_caption()
        self.load_model()

        for example_dir in tqdm(self.example_dirs, desc="Generating captions"):
            self.process_example(example_dir)

    def find_example_dirs_to_caption(self) -> None:
        """Finds the example directories to caption."""

        _logger.info("Searching for examples in '%s'", self.args["dataset_dir"])

        self.example_dirs = []
        num_skipped = 0
        for blip2_caption_file_path in self.args["dataset_dir"].rglob(
            self.args["blip2_caption_filename"]
        ):
            example_dir = blip2_caption_file_path.parent
            if (
                not self.args["overwrite"]
                and (example_dir / self.args["caption_filename"]).exists()
            ):
                num_skipped += 1
                continue
            self.example_dirs.append(example_dir)

        if num_skipped:
            _logger.info("Skipped %d examples with existing caption files", num_skipped)
        _logger.info("Found %d examples to caption", len(self.example_dirs))

    def load_model(self) -> None:
        """Loads the Llama 3 model."""

        _logger.info(
            "Loading model '%s', revision '%s'",
            self.args["model_name"],
            self.args["model_revision"],
        )

        match self.args["precision"]:
            case "bfloat16":
                model_kwargs = {"torch_dtype": torch.bfloat16}
            case "float16":
                model_kwargs = {"torch_dtype": torch.float16}
            case "8bit":
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "quantization_config": {"load_in_8bit": True},
                }
            case "4bit":
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "quantization_config": {"load_in_4bit": True},
                }
            case _:
                raise ValueError(f"Unexpected precision: {self.args['precision']}")

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.args["model_name"],
            revision=self.args["model_revision"],
            device_map={"": 0},
            model_kwargs=model_kwargs,
        )

    def process_example(self, example_dir: Path) -> None:
        """Generates and saves the caption for an example."""

        blip2_caption = (
            (example_dir / self.args["blip2_caption_filename"])
            .read_text(encoding="utf-8")
            .strip()
        )

        with (example_dir / "attributes.json").open(encoding="utf-8") as file:
            attributes = json.load(file)

        messages = self.create_chat_messages(blip2_caption, attributes)
        outputs = self.pipeline(
            messages,
            max_new_tokens=100,
            # To mute the message:
            # Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )

        # Postprocessing: Take only the last line of the response.
        llama3_caption = (
            outputs[0]["generated_text"][-1]["content"].strip().splitlines()[-1]
        )

        (example_dir / self.args["caption_filename"]).write_text(
            llama3_caption, encoding="utf-8"
        )

    @staticmethod
    def create_chat_messages(
        blip2_caption: str, attributes: dict[str, bool]
    ) -> list[dict[str, str]]:
        """Creates chat messages to pass to the Llama 3 model."""

        # For attributes except "Male" and "No_Beard", the attribute is mentioned only
        # if its value is True.
        attribute_to_prompt = {
            "5_o_Clock_Shadow": "There is a 5 o'clock shadow.",
            "Arched_Eyebrows": "The eyebrows are arched.",
            "Attractive": "The person is attractive.",
            "Bags_Under_Eyes": "There are bags under the eyes.",
            "Bald": "The person is bald.",
            "Bangs": "The person has bangs.",
            "Big_Lips": "The person has big lips.",
            "Big_Nose": "The person has a big nose.",
            "Black_Hair": "The person has black hair.",
            "Blond_Hair": "The person has blond hair.",
            "Blurry": "The image is blurry.",
            "Brown_Hair": "The person has brown hair.",
            "Bushy_Eyebrows": "The person has bushy eyebrows.",
            "Chubby": "The person is chubby.",
            "Double_Chin": "The person has a double chin.",
            "Eyeglasses": "The person is wearing eyeglasses.",
            "Goatee": "The person has a goatee.",
            "Gray_Hair": "The person has gray hair.",
            "Heavy_Makeup": "The person is wearing heavy makeup.",
            "High_Cheekbones": "The person has high cheekbones.",
            "Male": "",
            "Mouth_Slightly_Open": "The mouth is slightly open.",
            "Mustache": "The person has a mustache.",
            "Narrow_Eyes": "The person has narrow eyes.",
            "No_Beard": "",
            "Oval_Face": "The person has an oval face.",
            "Pale_Skin": "The person has pale skin.",
            "Pointy_Nose": "The person has a pointy nose.",
            "Receding_Hairline": "The person has a receding hairline.",
            "Rosy_Cheeks": "The person has rosy cheeks.",
            "Sideburns": "The person has sideburns.",
            "Smiling": "The person is smiling.",
            "Straight_Hair": "The person has straight hair.",
            "Wavy_Hair": "The person has wavy hair.",
            "Wearing_Earrings": "The person is wearing earrings.",
            "Wearing_Hat": "The person is wearing a hat.",
            "Wearing_Lipstick": "The person is wearing lipstick.",
            "Wearing_Necklace": "The person is wearing a necklace.",
            "Wearing_Necktie": "The person is wearing a necktie.",
            "Young": "The person is young.",
        }

        # A list of fact sentences from binary attributes.
        facts: list[str] = []
        for name, value in attributes.items():
            match name:
                case "No_Beard":
                    if value:
                        continue
                    fact = "The person has a beard."
                case "Male":
                    fact = f"The person is {'male' if value else 'female'}."
                case _:
                    if not value:
                        continue
                    fact = attribute_to_prompt[name]
            facts.append(fact)

        # Create chat messages.

        system_prompt = (
            "You are a helpful assistant who provides a concise, single-line, and "
            "single-sentence description of a person's image while avoiding any "
            "subjective comments."
        )
        user_prompt = (
            "Fix and extend the following description of a person's image using "
            "several facts about the person.\n"
            f"Original image description: {blip2_caption}\n"
            "Facts about the person:"
        )
        for fact in facts:
            user_prompt += f"\n- {fact}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
