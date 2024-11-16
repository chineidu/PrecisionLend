import logging
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.theme import Theme
from sklearn.pipeline import Pipeline
from typeguard import typechecked


@typechecked
def get_rich_logger(name: str | None = "richLogger") -> logging.Logger:
    """This is used to create a logger object with RichHandler."""

    # Create logger if it doesn't exist
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create console handler with formatting
        console_handler = RichHandler(
            rich_tracebacks=True,
            level=logging.DEBUG,
            log_time_format="%y-%m-%d %H:%M:%S",
        )
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)

        # Add console handler to the logger
        logger.addHandler(console_handler)

    return logger


logger = get_rich_logger()
custom_theme = Theme(
    {
        "white": "#FFFFFF",  # Bright white
        "info": "#00FF00",  # Bright green
        "warning": "#FFD700",  # Bright gold
        "error": "#FF1493",  # Deep pink
        "success": "#00FFFF",  # Cyan
        "highlight": "#FF4500",  # Orange-red
    }
)

console = Console(theme=custom_theme)


@typechecked
def fancy_print(
    object: Any,
    title: str = "Result",
    border_style: str = "bright_green",
    content_style: str | None = None,
    show_type: bool = True,
    expand: bool = False,
    return_panel: bool = False,
) -> Panel | None:
    if isinstance(object, dict):
        content = Table(show_header=False, box=box.SIMPLE)
        for key, value in object.items():
            content.add_row(
                Text(str(key), style="cyan"),
                Text(str(value), style=content_style or "white"),
            )
    elif isinstance(object, (list, tuple)):
        content = Table(show_header=False, box=box.SIMPLE)
        for i, item in enumerate(object):
            content.add_row(
                Text(str(i), style="cyan"),
                Text(str(item), style=content_style or "white"),
            )
    else:
        content = Text(str(object), style=content_style or "white")

    if show_type:
        title = f"{title} ({type(object).__name__})"

    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style=border_style,
        expand=expand,
    )
    if return_panel:
        return panel
    else:
        console.print(panel)
        return None


@typechecked
def transform_array_to_lazyframe(
    array: np.ndarray, processor_pipe: Pipeline
) -> pl.LazyFrame:
    """Transform a numpy array using a scikit-learn Pipeline and convert to a
    Polars LazyFrame.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (n_samples, n_features)
    processor_pipe : Pipeline
        Fitted scikit-learn Pipeline for data transformation

    Returns
    -------
    pl.LazyFrame
        Transformed data as a Polars LazyFrame with feature names from the pipeline
    """
    array = processor_pipe.transform(array)
    data: pl.LazyFrame = pl.LazyFrame(
        array, schema=processor_pipe.get_feature_names_out()
    )
    return data
