"""Utility functions related to prompts."""

from diffusers import StableDiffusion3Pipeline


def get_prompt_tokens(
    pipeline: StableDiffusion3Pipeline,
    prompt: str,
) -> tuple[list[str], list[str]]:
    """Gets the CLIP and T5 tokens for a given prompt."""

    clip_token_ids = pipeline.tokenizer(
        prompt,
        padding=False,
        max_length=77,
        truncation=True,
    )["input_ids"]
    clip_tokens = pipeline.tokenizer.convert_ids_to_tokens(clip_token_ids)

    t5_token_ids = pipeline.tokenizer_3(
        prompt,
        padding=False,
        max_length=256,
        truncation=True,
        add_special_tokens=True,
    )["input_ids"]
    t5_tokens = pipeline.tokenizer_3.convert_ids_to_tokens(t5_token_ids)

    return clip_tokens, t5_tokens
