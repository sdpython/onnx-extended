import os
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from timeit import Timer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy
from numpy.testing import assert_allclose


def is_azure() -> bool:
    "Tells if the job is running on Azure DevOps."
    return os.environ.get("AZURE_HTTP_USER_AGENT", "undefined") != "undefined"


def is_windows() -> bool:
    return sys.platform == "win32"


def is_apple() -> bool:
    return sys.platform == "darwin"


def skipif_ci_windows(msg) -> Callable:
    """
    Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`.
    """
    if is_windows() and is_azure():
        msg = f"Test does not work on azure pipeline (Windows). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_apple(msg) -> Callable:
    """
    Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`.
    """
    if is_apple() and is_azure():
        msg = f"Test does not work on azure pipeline (Apple). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_unstable(msg) -> Callable:
    """
    Skips a unit test if the environment variable `SKIP_UNSTABLE` is set to 1.
    """
    value = os.environ.get("SKIP_UNSTABLE", "0")
    if value in ("1", 1):
        msg = f"Test is unstable. Disabling it. {msg}"
        return unittest.skip(msg)
    return lambda x: x


def unit_test_going():
    """
    Enables a flag telling the script is running while testing it.
    Avois unit tests to be very long.
    """
    going = int(os.environ.get("UNITTEST_GOING", 0))
    return going == 1


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        assert warns is not None, f"warns cannot be None for '{fct}'."

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def measure_time(
    stmt: Union[str, Callable],
    context: Optional[Dict[str, Any]] = None,
    repeat: int = 10,
    number: int = 50,
    warmup: int = 1,
    div_by_number: bool = True,
    max_time: Optional[float] = None,
) -> Dict[str, Union[str, int, float]]:
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string or callable
    :param context: variable to know in a dictionary
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param warmup: number of iteration to do before starting the
        real measurement
    :param div_by_number: divide by the number of executions
    :param max_time: execute the statement until the total goes
        beyond this time (approximatively), *repeat* is ignored,
        *div_by_number* must be set to True
    :return: dictionary

    .. runpython::
        :showcode:

        from pprint import pprint
        from math import cos
        from onnx_extended.ext_test_case import measure_time

        res = measure_time(lambda: cos(0.5))
        pprint(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    assert callable(stmt) or isinstance(
        stmt, str
    ), f"stmt is not callable or a string but is of type {type(stmt)!r}."
    if context is None:
        context = {}

    if isinstance(stmt, str):
        tim = Timer(stmt, globals=context)
    else:
        tim = Timer(stmt)

    if warmup > 0:
        warmup_time = tim.timeit(warmup)
    else:
        warmup_time = 0

    if max_time is not None:
        assert (
            div_by_number
        ), "div_by_number must be set to True of max_time is defined."
        i = 1
        total_time = 0.0
        results = []
        while True:
            for j in (1, 2):
                number = i * j
                time_taken = tim.timeit(number)
                results.append((number, time_taken))
                total_time += time_taken
                if total_time >= max_time:
                    break
            if total_time >= max_time:
                break
            ratio = (max_time - total_time) / total_time
            ratio = max(ratio, 1)
            i = int(i * ratio)

        res = numpy.array(results)
        tw = res[:, 0].sum()
        ttime = res[:, 1].sum()
        mean = ttime / tw
        ave = res[:, 1] / res[:, 0]
        dev = (((ave - mean) ** 2 * res[:, 0]).sum() / tw) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(ave),
            max_exec=numpy.max(ave),
            repeat=1,
            number=tw,
            ttime=ttime,
        )
    else:
        res = numpy.array(tim.repeat(repeat=repeat, number=number))
        if div_by_number:
            res /= number

        mean = numpy.mean(res)
        dev = numpy.mean(res**2)
        dev = (dev - mean**2) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(res),
            max_exec=numpy.max(res),
            repeat=repeat,
            number=number,
            ttime=res.sum(),
        )

    if "values" in context:
        if hasattr(context["values"], "shape"):
            mes["size"] = context["values"].shape[0]
        else:
            mes["size"] = len(context["values"])
    else:
        mes["context_size"] = sys.getsizeof(context)
    mes["warmup_time"] = warmup_time
    return mes


class ExtTestCase(unittest.TestCase):
    _warns: List[Tuple[str, int, Warning]] = []

    def assertExists(self, name):
        assert os.path.exists(name), f"File or folder {name!r} does not exists."

    def assertIns(self, sub: Tuple[Any, ...], s: str):
        """
        Checks that one of the substrings in sub is part of s.
        """
        for t in sub:
            if t in s:
                return
        raise AssertionError(f"None of the substring in {sub} is part of {s!r}.")

    def assertEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[str] = None,
    ):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        try:
            assert_allclose(expected, value, atol=atol, rtol=rtol)
        except AssertionError as e:
            raise AssertionError(msg) from e

    def assertAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertNotAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        try:
            self.assertEqualArray(expected, value, atol=atol, rtol=rtol)
            raise AssertionError("Arrays are equal.")
        except AssertionError:
            pass

    def assertRaise(self, fct: Callable, exc_type: type[Exception]):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if not value:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if not value:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}")

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout):
            with redirect_stderr(serr):
                try:
                    res = fct()
                except Exception as e:
                    raise AssertionError(
                        f"function {fct} failed, stdout="
                        f"\n{sout.getvalue()}\n---\nstderr=\n{serr.getvalue()}"
                    ) from e
        return res, sout.getvalue(), serr.getvalue()

    def tryCall(
        self, fct: Callable, msg: Optional[str] = None, none_if: Optional[str] = None
    ) -> Optional[Any]:
        """
        Calls the function, catch any error.

        :param fct: function to call
        :param msg: error message to display if failing
        :param none_if: returns None if this substring is found in the error message
        :return: output of *fct*
        """
        try:
            return fct()
        except Exception as e:
            if none_if is not None and none_if in str(e):
                return None
            if msg is None:
                raise e
            raise AssertionError(msg) from e
