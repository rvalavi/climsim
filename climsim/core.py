from .utils import read_rast
from climsim_rust import dissimpy
import os

def dissim(
        files: str,
        radius: float | None = None,
        bandwidth: float = 1.0,
        n_sample: int | None = 10_000,
        seed: int = 42,
        n_threads: int | None = None,
    ):
    """
    Calculate climate dissimilarity for each grid cell.
    
    Args:
        files: Path to gridded climate data (one multi-band or several files)
        radius: Maximum distance (km) for neighbor sampling (None = all cells)
        bandwidth: Gaussian kernel bandwidth for distance weighting
        n_sample: Number of non-NA cells to sample as neighbors (None = all valid cells)
        seed: Random seed for sampling
        n_threads: Number of parallel threads (None = auto)
    
    Returns:
        A numpy array of dissimilarity scores per cell
    """

    if n_threads is None:
        n_threads = os.cpu_count() or 1

    if n_sample is None or n_sample < 0:
        n_sample = 0
    
    if radius is not None and radius <= 0:
        radius = None
    
    r, t, geo, dim = read_rast(files)

    outarray = dissimpy(
        arr = r,
        trans = t,
        is_geo = geo,
        nrows = dim[0],
        ncols = dim[1],
        bandwidth = bandwidth,
        nsample = n_sample,
        seed = seed,
        n_cores = n_threads,
        radius = radius,
    )

    return outarray