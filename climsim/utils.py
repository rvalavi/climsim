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
        - Always returns a 2D array with: (cell, bands)
        - Single file: bands = raster bands
        - Multiple files (single-band each): bands = file index

    is_geo : bool
        Whether CRS is geographic (lon/lat).
    """
    def _is_geographic(src) -> bool:
        try:
            if src.crs is None:
                return is_geographic(src)  # your existing heuristic fn
            return src.crs.is_geographic
        except Exception as e:
            raise RuntimeError(f"Error reading CRS info: {e}")

    def _read_one(path: Path) -> Tuple[np.ndarray, bool]:
        with rasterio.open(path) as src:
            is_geo = _is_geographic(src)
            data = src.read(masked=True)  # (bands, rows, cols)
            arr = np.where(data.mask, np.nan, data).astype(np.float32)
            # Make it (rows, cols, bands) for fast per-cell band vectors in Rust
            arr = np.moveaxis(arr, 0, -1)
            return arr, is_geo

    # single path
    if isinstance(files, (str, bytes)) or hasattr(files, "__fspath__"):
        arr, is_geo = _read_one(files)
        # Reshape the array to (cell, band); this way be consistant for Rust
        rows, cols, n_bands = arr.shape
        flat = arr.reshape(rows * cols, n_bands)
        return flat, bool(is_geo), (rows, cols)

    # list/tuple of paths
    if not isinstance(files, (list, tuple)) or len(files) == 0:
        raise ValueError("files must be a path or a non-empty list/tuple of paths")

    arrays = []
    is_geo0 = None
    shape0 = None

    for f in files:
        arr, is_geo = _read_one(f)  # (rows, cols, bands)

        # For multi-file stacking, enforce exactly 1 band per file
        if arr.ndim != 3 or arr.shape[2] != 1:
            raise ValueError(
                f"Multi-file input expects single-band rasters; '{f}' has shape {arr.shape}"
            )

        # Drop the singleton band axis -> (rows, cols)
        arr2d = arr[..., 0]

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

    # Stack files into the *band* axis (bands last): (rows, cols, n_files)
    stacked = np.stack(arrays, axis=-1).astype(np.float32)
    # Reshape the array to (cell, band); this way be consistant for Rust
    rows, cols, n_bands = stacked.shape
    flat = stacked.reshape(rows * cols, n_bands)
    return flat, bool(is_geo0), (rows, cols)

