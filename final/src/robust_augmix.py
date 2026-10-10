"""Reflected PSF convolution used by HardShift blur profiles."""
import numpy as np
from PIL import Image


def _image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 255)).astype(np.uint8))

def _filter_psf(image, kernel):
    """Small RGB convolution using reflected boundaries; no scipy dependency."""
    array = np.asarray(image, dtype=np.float32)
    radius = kernel.shape[0] // 2
    padded = np.pad(array, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    result = np.zeros_like(array)
    height, width = array.shape[:2]
    for y, x in np.argwhere(kernel > 0):
        result += kernel[y, x] * padded[y:y + height, x:x + width]
    return _image(result)
