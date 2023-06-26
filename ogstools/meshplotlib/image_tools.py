"""Image processing utilities."""

import io

import matplotlib.figure as mfigure
import numpy as np
import PIL.Image as Image


def trim(im: Image.Image, margin: int) -> Image.Image:
    """
    Trim the input image by removing excess white space.

    :param im: The input image to be trimmed.
    :param margin: The margin size to add around the trimmed image.

    :returns: The trimmed image.

    """
    w, h = (im.width, im.height)
    img = Image.new("RGB", (w + 2 * margin, h + 2 * margin), color="white")
    img.paste(im, np.array([0, 0, w, h]) + int(margin))
    coords = np.argwhere(np.any(np.array(img) != [255, 255, 255], axis=2))
    y0, x0 = np.min(coords, axis=0)
    y1, x1 = np.max(coords, axis=0)
    bbox = (x0 - margin, y0 - margin, x1 + 1 + margin, y1 + 1 + margin)
    return img.crop(bbox)


def save_plot(fig: mfigure.Figure, out_name: str, margin: int = 10) -> None:
    """
    Save a matplotlib figure as an image file, trimming excess white space.

    :param fig: The matplotlib figure to be saved.
    :param out_name: The output filename for the saved image.
    :param margin: The margin to add around the trimmed image. Defaults to 10.

    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    im = trim(Image.open(buf), margin)
    im.save(out_name)
