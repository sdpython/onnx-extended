from typing import Any, List


def pick(*args: List[Any]) -> Any:
    """
    Picks the value value not None.
    """
    for a in args:
        if a is not None:
            return a
    raise ValueError("All values are None.")
