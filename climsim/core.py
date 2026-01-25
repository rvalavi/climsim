from .utils import read_rast
from climsim_rust import similaritypy
import os

def simialrity(
        files: str,
        raduis: float | None = None,
        band_width: float = 1.0,
        n_sample: int | None = 10_000,
        seed = 42,
        n_threads = None,
    ):
    """
    Docstring for simialrity
    
    :param file: Description
    :type file: str
    """

    if n_threads is None:
        n_threads = os.cpu_count() or 1

    if n_sample is None or n_sample < 0:
        n_sample = 0
    
    r, geo, dim = read_rast(files)

    outarray = similaritypy(
        arr = r,
        nrows = dim[0],
        ncols = dim[1],
        bandwidth = band_width,
        nsample = n_sample,
        seed = seed,
        n_cores = n_threads,
    )

    return outarray