from utils import read_rast

def simialrity(
        file: str, 
        local: bool = True,
        raduis: float | None = None,
        exact: bool = False # use f32 for more exact calc, or bf16 for faster calc
    ):
    """
    Docstring for simialrity
    
    :param file: Description
    :type file: str
    """

    r = read_rast(file)

    outarray = similaritypy(
        arr = r,
    )

    return outarray