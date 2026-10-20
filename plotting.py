from pathlib import Path
from typing import Any

import plotly.express as px
from IPython.display import HTML, display

from configurations import (
    OUTPUT_DIR,
    PLOTLY_CONFIG,
    PLOTLY_HEIGHT,
    PLOTLY_TEMPLATE,
    PLOTLY_WIDTH,
)


px.defaults.template = PLOTLY_TEMPLATE
px.defaults.width = PLOTLY_WIDTH
px.defaults.height = PLOTLY_HEIGHT


def _output_path(filename: str | Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Build an output path and create its parent directory."""
    path = output_dir / Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def display_html(fig: Any, config: dict[str, Any] | None = None) -> None:
    """Display an interactive Plotly figure in the notebook output."""
    html = fig.to_html(
        config=config or PLOTLY_CONFIG,
        include_plotlyjs="cdn",
        full_html=False,
    )
    display(HTML(html))


def save_html(
    fig: Any,
    filename: str | Path,
    output_dir: Path = OUTPUT_DIR,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save an interactive Plotly figure as an HTML file."""
    path = _output_path(filename, output_dir)
    fig.write_html(path, include_plotlyjs="cdn", config=config or PLOTLY_CONFIG)
    return path


def save_image(
    fig: Any,
    filename: str | Path,
    output_dir: Path = OUTPUT_DIR,
    scale: int = 2,
) -> Path:
    """Save a Plotly figure using the image format from the file extension."""
    path = _output_path(filename, output_dir)
    image_format = path.suffix.lstrip(".").lower()
    if not image_format:
        raise ValueError("The image filename must include an extension.")
    fig.write_image(path, format=image_format, scale=scale)
    return path


def show_and_save(
    fig: Any,
    image_filename: str | Path,
    output_dir: Path = OUTPUT_DIR,
    scale: int = 2,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save a figure as an image and display its interactive HTML version."""
    image_path = save_image(fig, image_filename, output_dir=output_dir, scale=scale)
    display_html(fig, config=config)
    print(f"Figure saved: {image_path}")
    return image_path
