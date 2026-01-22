from .utils import read_rast
from climsim_rust import similaritypy


def simialrity(
        files: str, 
        # local: bool = True,
        # raduis: float | None = None,
        # exact: bool = False # use f32 for more exact calc, or bf16 for faster calc
        n_threads,
    ):
    """
    Docstring for simialrity
    
    :param file: Description
    :type file: str
    """

    r, geo, dim = read_rast(files)

    outarray = similaritypy(
        arr = r,
        nrows = dim[0],
        ncols = dim[1],
        n_cores = n_threads,
    )

    return outarray