"""Implementation of the TextTokenAndImageVisualizer class."""

import itertools
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from .ncut import get_ncut_colors

BBox = tuple[int, int, int, int]
TokenID = Literal["clip_heading", "t5_heading"] | int
TypesetInput = tuple[str, ImageFont.FreeTypeFont, TokenID] | Literal["\n", " "]
TypesetOutput = tuple[BBox, TokenID]


class TextTokenAndImageVisualizer:
    """A class to visualize text tokens and an image together in a single figure."""

    class Config(NamedTuple):
        """Configuration for the visualizer class."""

        figure_width: int = 2560
        figure_padding: int = 80
        token_spacing: int = 30
        line_spacing: int = 90
        font_path: str | None = None
        heading_font_size: int = 64
        text_token_font_size: int = 56
        text_token_corner_radius: int = 10

    def __init__(self, config: Config = Config()) -> None:
        self._config = config

        font_path = config.font_path
        if font_path is None:
            font_path = str(Path(__file__).parent / "assets/RobotoMono-Regular.ttf")
        self._heading_font = ImageFont.truetype(font_path, config.heading_font_size)
        self._text_token_font = ImageFont.truetype(
            font_path, config.text_token_font_size
        )

        self._text_token_bboxes: dict[int, BBox] = {}
        self._image_offset: int = 0

    @property
    def image_size(self) -> int:
        """The size to resize the image to."""

        return self._config.figure_width - 2 * self._config.figure_padding

    def visualize_data(
        self,
        clip_tokens: Sequence[str],
        clip_token_colors: np.ndarray,
        t5_tokens: Sequence[str],
        t5_token_colors: np.ndarray,
        image: Image.Image | None = None,
        overlay_colors: np.ndarray | None = None,
    ) -> None:
        """Visualizes the text tokens and the image.

        Args:
            clip_tokens: A sequence of CLIP token strings.
            clip_token_colors: RGB/RGBA colors of shape `(num_tokens, 3|4)`.
            t5_tokens: A sequence of T5 token strings.
            t5_token_colors: RGB/RGBA colors of shape `(num_tokens, 3|4)`.
            image: An 1024x1024 image to visualize.
            overlay_colors: An 64x64 overlay to the image.

        Returns:
            The visualized figure.
        """

        inputs: list[TypesetInput] = []

        clip_heading = "CLIP tokens:"
        inputs.append((clip_heading, self._heading_font, "clip_heading"))
        inputs.append("\n")
        for i, token_str in enumerate(clip_tokens):
            if i > 0:
                inputs.append(" ")
            token_index = 4096 + i
            inputs.append((token_str, self._text_token_font, token_index))
        inputs.append("\n")

        t5_heading = "T5 tokens:"
        inputs.append((t5_heading, self._heading_font, "t5_heading"))
        inputs.append("\n")
        for i, token_str in enumerate(t5_tokens):
            if i > 0:
                inputs.append(" ")
            token_index = 4096 + 77 + i
            inputs.append((token_str, self._text_token_font, token_index))
        inputs.append("\n")

        typeset_iter = self._typeset(inputs)

        self._text_token_bboxes = {}
        clip_heading_bbox: BBox | None = None
        t5_heading_bbox: BBox | None = None

        # Read the typeset outputs and store the bounding boxes.
        while True:
            try:
                typeset_output = next(typeset_iter)
            except StopIteration as e:
                self._image_offset = e.value
                break

            bbox, token_id = typeset_output
            if token_id == "clip_heading":
                clip_heading_bbox = bbox
            elif token_id == "t5_heading":
                t5_heading_bbox = bbox
            else:
                self._text_token_bboxes[token_id] = bbox

        if overlay_colors is None:
            overlay = None
        else:
            assert overlay_colors.shape[0:2] == (64, 64)
            overlay_colors = self._to_uint8_colors(overlay_colors)

            # Upsample (64, 64, 3|4) -> (1024, 1024, 3|4)
            overlay_colors = np.repeat(overlay_colors, 16, axis=0)
            overlay_colors = np.repeat(overlay_colors, 16, axis=1)

            overlay = Image.fromarray(overlay_colors)

        # Apply the overlay to the image.
        if image is not None:
            assert image.size == (1024, 1024)
            if overlay is not None:
                image = Image.alpha_composite(
                    image.convert("RGBA"), overlay.convert("RGBA")
                )
        elif overlay is not None:
            image = overlay
        else:
            image = Image.new("RGB", (1024, 1024), color="lightgray")

        image_size = self.image_size
        image = image.resize((image_size, image_size))

        figure = Image.new(
            "RGB",
            (
                self._config.figure_width,
                self._image_offset + image_size + self._config.figure_padding,
            ),
            color="white",
        )

        figure.paste(image, (self._config.figure_padding, self._image_offset))

        draw = ImageDraw.Draw(figure, "RGBA")

        # Draw the headings.
        draw.text(
            clip_heading_bbox[:2], clip_heading, fill="black", font=self._heading_font
        )
        draw.text(
            t5_heading_bbox[:2], t5_heading, fill="black", font=self._heading_font
        )

        # Pillow accepts only 8-bit color values.
        clip_token_colors = self._to_uint8_colors(clip_token_colors)
        t5_token_colors = self._to_uint8_colors(t5_token_colors)

        # Draw the tokens with colored backgrounds.
        for token_index, (token_str, token_color) in itertools.chain(
            enumerate(zip(clip_tokens, clip_token_colors), start=4096),
            enumerate(zip(t5_tokens, t5_token_colors), start=4096 + 77),
        ):
            bbox = self._text_token_bboxes[token_index]
            draw.rounded_rectangle(
                bbox,
                radius=self._config.text_token_corner_radius,
                fill=tuple(token_color),
            )
            draw.text(
                bbox[:2],
                token_str,
                fill=self._pick_text_color(token_color),
                font=self._text_token_font,
            )

        return figure

    def visualize_empty_data(self) -> Image.Image:
        """Visualizes an empty figure."""

        return self.visualize_data(
            clip_tokens=[],
            clip_token_colors=np.empty((0, 3)),
            t5_tokens=[],
            t5_token_colors=np.empty((0, 3)),
        )

    def visualize_affinity_heatmap(
        self,
        clip_tokens: Sequence[str],
        t5_tokens: Sequence[str],
        affinity: torch.Tensor,
        image: Image.Image | None = None,
    ) -> Image.Image:
        """Visualizes the affinities between tokens using a heatmap.

        `affinity` has shape `(num_tokens,)` and is the row of the selected token in the
        affinity matrix.
        """

        affinity = affinity.cpu().numpy()

        def affinity_to_color(
            affinity_values: np.ndarray, alpha: bool = False
        ) -> np.ndarray:
            """Converts the affinity values to colors."""

            affinity_values -= affinity_values.min()
            affinity_values /= affinity_values.max() + 1e-6
            colors = plt.get_cmap("plasma")(affinity_values)
            if alpha:
                colors[:, 3] = affinity_values * 0.6 + 0.4  # 0.6 <= alpha <= 1.0
            return (colors * 255).astype(np.uint8)

        clip_token_colors = affinity_to_color(affinity[4096 : 4096 + len(clip_tokens)])
        t5_token_colors = affinity_to_color(
            affinity[4096 + 77 : 4096 + 77 + len(t5_tokens)]
        )
        overlay_colors = affinity_to_color(
            affinity[0:4096], alpha=image is not None
        ).reshape(64, 64, 4)

        return self.visualize_data(
            clip_tokens,
            clip_token_colors,
            t5_tokens,
            t5_token_colors,
            image,
            overlay_colors,
        )

    def visualize_affinity_ncut(
        self,
        clip_tokens: Sequence[str],
        t5_tokens: Sequence[str],
        affinity: torch.Tensor,
        image: Image.Image | None = None,
        num_eigenvectors: int = 30,
        device: str | torch.device | None = None,
    ) -> Image.Image:
        """Visualizes the affinities between tokens using NCUT.

        `affinity` is the affinity matrix of shape `(num_tokens, num_tokens)`.
        """

        ncut_colors = get_ncut_colors(affinity, num_eigenvectors, device)

        clip_token_colors = ncut_colors[4096 : 4096 + len(clip_tokens)]
        t5_token_colors = ncut_colors[4096 + 77 : 4096 + 77 + len(t5_tokens)]

        overlay_colors = ncut_colors[0:4096].reshape(64, 64, 3)
        if image is not None:
            # Alpha = 0.7 for the overlay
            overlay_colors = np.concatenate(
                (overlay_colors, np.full((64, 64, 1), 0.7)), axis=2
            )

        return self.visualize_data(
            clip_tokens,
            clip_token_colors,
            t5_tokens,
            t5_token_colors,
            image,
            overlay_colors,
        )

    def get_token_index(self, x: float, y: float) -> int | None:
        """Gets the index of the text of image token at the given position.

        Args:
            x: The x coordinate of the position.
            y: The y coordinate of the position.

        Returns:
            The token index if the position is within the bounding box of a text token
            or an image patch, or `None` if otherwise.
        """

        for token_index, (
            x_min,
            y_min,
            x_max,
            y_max,
        ) in self._text_token_bboxes.items():
            if x_min <= x < x_max and y_min <= y < y_max:
                return token_index

        x -= self._config.figure_padding
        y -= self._image_offset
        image_size = self.image_size
        x /= image_size
        y /= image_size

        if not (0 <= x < 1 and 0 <= y < 1):
            return None

        # 64 image patches per side.
        patch_x = max(0, min(63, int(x * 64)))
        patch_y = max(0, min(63, int(y * 64)))
        return patch_y * 64 + patch_x

    def _typeset(
        self, inputs: Iterable[TypesetInput]
    ) -> Generator[TypesetOutput, None, int]:
        x = self._config.figure_padding
        y = self._config.figure_padding

        for typeset_input in inputs:
            if typeset_input == "\n":
                x = self._config.figure_padding
                y += self._config.line_spacing
                continue

            if typeset_input == " ":
                x += self._config.token_spacing
                continue

            text, font, token_id = typeset_input

            bbox_width = font.getlength(text)
            ascent, descent = font.getmetrics()
            bbox_height = ascent + descent

            # Breaks the line if we are not at the beginning of the line and the next
            # text does not fit in the current line.
            if (
                x != self._config.figure_padding
                and x + bbox_width
                > self._config.figure_width - self._config.figure_padding
            ):
                x = self._config.figure_padding
                y += self._config.line_spacing

            yield (x, y, x + bbox_width, y + bbox_height), token_id
            x += bbox_width

        return y + self._config.line_spacing + self._config.figure_padding

    @staticmethod
    def _to_uint8_colors(colors: np.ndarray) -> np.ndarray:
        if np.issubdtype(colors.dtype, np.integer):
            return colors.astype(np.uint8)
        if np.issubdtype(colors.dtype, np.floating):
            return (colors.clip(0, 1) * 255).astype(np.uint8)
        raise ValueError(f"Unsupported dtype: {colors.dtype}")

    @staticmethod
    def _pick_text_color(background_color: np.ndarray) -> str:
        background_color = background_color / 255

        brightness = np.dot(background_color[:3], [0.2126, 0.7152, 0.0722])
        if background_color.size > 3:
            alpha = background_color[3]
            brightness = brightness * alpha + (1 - alpha)

        # Use black for light backgrounds and white for dark backgrounds.
        return "black" if brightness >= 0.5 else "white"
