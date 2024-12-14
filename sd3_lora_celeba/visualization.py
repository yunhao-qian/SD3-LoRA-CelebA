"""Utility classes for visualizing the Stable Diffusion 3 model."""

import itertools
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BBox = tuple[int, int, int, int]


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

    class _TypesetInput(NamedTuple):
        input_type: Literal["text", "newline", "space"]
        font: ImageFont.FreeTypeFont | None = None
        text: str = ""
        input_id: str = ""

        def get_bbox_size(self) -> tuple[int, int]:
            """Returns the bounding box size of the text."""

            bbox_width = self.font.getlength(self.text)
            ascent, descent = self.font.getmetrics()
            bbox_height = ascent + descent
            return bbox_width, bbox_height

    class _TypesetOutput(NamedTuple):
        bbox: BBox
        input_id: str

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
        self._figure: Image.Image | None = None

    def set_data(
        self,
        clip_tokens: dict[int, str],
        clip_token_colors: np.ndarray,
        t5_tokens: dict[int, str],
        t5_token_colors: np.ndarray,
        image: Image.Image,
    ) -> None:
        """Sets the data to visualize.

        Args:
            clip_tokens: A dictionary mapping token indices to token strings.
            clip_token_colors: `uint8` RGB/RGBA colors of shape `(num_tokens, 3|4)`.
            t5_tokens: A dictionary mapping token indices to token strings.
            t5_token_colors: `uint8` RGB/RGBA colors of shape `(num_tokens, 3|4)`.
            image: An image to visualize.
        """

        # Construct the typeset inputs.
        inputs: list[self._TypesetInput] = []
        clip_heading = "CLIP tokens:"
        inputs.append(
            self._TypesetInput("text", self._heading_font, clip_heading, "clip_heading")
        )
        inputs.append(self._TypesetInput("newline"))
        for i, (token_index, token_str) in enumerate(clip_tokens.items()):
            if i > 0:
                inputs.append(self._TypesetInput("space"))
            inputs.append(
                self._TypesetInput(
                    "text", self._text_token_font, token_str, str(token_index)
                )
            )
        inputs.append(self._TypesetInput("newline"))
        t5_heading = "T5 tokens:"
        inputs.append(
            self._TypesetInput("text", self._heading_font, t5_heading, "t5_heading")
        )
        inputs.append(self._TypesetInput("newline"))
        for i, (token_index, token_str) in enumerate(t5_tokens.items()):
            if i > 0:
                inputs.append(self._TypesetInput("space"))
            inputs.append(
                self._TypesetInput(
                    "text", self._text_token_font, token_str, str(token_index)
                )
            )

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
            if typeset_output.input_id == "clip_heading":
                clip_heading_bbox = typeset_output.bbox
            elif typeset_output.input_id == "t5_heading":
                t5_heading_bbox = typeset_output.bbox
            else:
                token_index = int(typeset_output.input_id)
                self._text_token_bboxes[token_index] = typeset_output.bbox

        image_size = self._config.figure_width - 2 * self._config.figure_padding
        image = image.resize((image_size, image_size))

        figure = Image.new(
            "RGB",
            (
                self._config.figure_width,
                self._image_offset + image_size + self._config.figure_padding,
            ),
            "white",
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
        for (token_index, token_str), token_color in itertools.chain(
            zip(clip_tokens.items(), clip_token_colors),
            zip(t5_tokens.items(), t5_token_colors),
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

        self._figure = figure

    def set_empty_data(self) -> None:
        """Sets empty data to visualize."""

        self.set_data(
            clip_tokens={},
            clip_token_colors=np.empty((0, 4)),
            t5_tokens={},
            t5_token_colors=np.empty((0, 4)),
            image=Image.new("RGB", (1024, 1024), "lightgray"),
        )

    def get_figure(self) -> Image.Image:
        """Returns the visualized figure."""

        if self._figure is None:
            raise ValueError("No data is set")
        return self._figure

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
        image_size = self._config.figure_width - 2 * self._config.figure_padding
        x /= image_size
        y /= image_size

        if not (0 <= x < 1 and 0 <= y < 1):
            return None

        # 64 image patches per side.
        patch_x = max(0, min(63, int(x * 64)))
        patch_y = max(0, min(63, int(y * 64)))
        return patch_y * 64 + patch_x

    def _typeset(
        self, inputs: Iterable[_TypesetInput]
    ) -> Generator[_TypesetOutput, None, int]:
        x = self._config.figure_padding
        y = self._config.figure_padding

        for typeset_input in inputs:
            if typeset_input.input_type == "newline":
                x = self._config.figure_padding
                y += self._config.line_spacing
                continue

            if typeset_input.input_type == "space":
                x += self._config.token_spacing
                continue

            bbox_width, bbox_height = typeset_input.get_bbox_size()

            # Breaks the line if we are not at the beginning of the line and the next
            # text does not fit in the current line.
            if (
                x != self._config.figure_padding
                and x + bbox_width
                > self._config.figure_width - self._config.figure_padding
            ):
                x = self._config.figure_padding
                y += self._config.line_spacing

            yield self._TypesetOutput(
                (x, y, x + bbox_width, y + bbox_height), typeset_input.input_id
            )
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
