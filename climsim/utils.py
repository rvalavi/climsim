from __future__ import annotations

from typing import Union, Sequence, Tuple
import numpy as np
import rasterio
from os import PathLike

Path = Union[str, PathLike[str]]


def is_geographic(src):
    """Guess CRS type from transform and bounds when CRS is missing."""
    transform = src.transform
    xres = abs(transform.a)
    yres = abs(transform.e)
    bounds = src.bounds

    # Heuristic 1: resolution
    if xres < 1 and yres < 1:
        return True  # likely geographic (degrees)
    if xres > 1 and yres > 1:
        return False  # likely projected (metres)

    # Heuristic 2: extent ranges
    if (-180 <= bounds.left <= 180 and
        -180 <= bounds.right <= 180 and
        -90 <= bounds.bottom <= 90 and
        -90 <= bounds.top <= 90):
        return True

    return False  # default to projected if unsure


def read_rast(files: Union[Path, Sequence[Path]]) -> Tuple[np.ndarray, bool]:
    """Read raster(s) with nodata->nan.

    Output shape invariant:
        - Always returns a 3D array
        - Axis 0 = band index (single file) OR file index (multi-file input)
    is_geo : bool
        Whether CRS is geographic (lon/lat).
    """
    def _is_geographic(src) -> bool:
        try:
            if src.crs is None:
                return is_geographic(src)
            return src.crs.is_geographic
        except Exception as e:
            raise RuntimeError(f"Error reading CRS info: {e}")

    def _read_one(path: PathLike) -> Tuple[np.ndarray, bool]:
        with rasterio.open(path) as src:
            is_geo = _is_geographic(src)
            data = src.read(masked=True)  # (bands, rows, cols)
            arr = np.where(data.mask, np.nan, data).astype(np.float32)
            return arr, is_geo

    # single path
    if isinstance(files, (str, bytes)) or hasattr(files, "__fspath__"):
        return _read_one(files)

    # list/tuple of paths
    if not isinstance(files, (list, tuple)) or len(files) == 0:
        raise ValueError("files must be a path or a non-empty list/tuple of paths")

    arrays = []
    is_geo0 = None
    shape0 = None

    for f in files:
        arr, is_geo = _read_one(f)

        # For multi-file stacking, enforce exactly 1 band per file
        if arr.ndim != 3 or arr.shape[0] != 1:
            raise ValueError(
                f"Multi-file input expects single-band rasters; '{f}' has shape {arr.shape}"
            )

        # Drop the band axis (since it's guaranteed 1) before stacking files
        arr2d = arr[0]  # (rows, cols)

        if is_geo0 is None:
            is_geo0 = is_geo
            shape0 = arr2d.shape
        else:
            if is_geo != is_geo0:
                raise ValueError("Input rasters disagree on CRS type (geographic vs projected).")
            if arr2d.shape != shape0:
                raise ValueError(
                    f"Input rasters must have identical shapes. Expected {shape0}, got {arr2d.shape} for '{f}'."
                )

        arrays.append(arr2d)

    stacked = np.stack(arrays, axis=0).astype(np.float32)  # (n_files, rows, cols)
    return stacked, bool(is_geo0)

