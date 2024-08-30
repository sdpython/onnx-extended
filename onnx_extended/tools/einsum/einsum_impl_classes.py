from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy
from onnx import helper, numpy_helper, ModelProto, NodeProto, TensorProto
from .einsum_config import (
    DEFAULT_IR_VERSION,
    DEFAULT_OPSET,
    guess_proto_dtype,
)
from .einsum_impl_ext import (
    numpy_extended_dot,
    numpy_diagonal,
    _numpy_extended_dot_equation,
    numpy_extended_dot_python,
    numpy_extended_dot_matrix,
)
from .blas_lapack import gemm_dot


def single_axes(axes: Tuple[int, ...]) -> Optional[List[int]]:
    """
    *axes* contains positive values, then it is the position
    of this axis in the original matrix, otherwise it is -1
    meaning this axis is an added single dimension to align
    all the dimensions based on the einsum equation.

    :param axes: axes described above
    :return: list of integer in set `{1, 2}`, 1 for
        a single axis, 2 otherwise
    """
    if axes is None:
        return axes
    return [(1 if a == -1 else 2) for a in axes]


class EinsumSubOp:
    """
    Defines a sub operation used in Einsum decomposition.

    :param full_dim: dimension of the result
    :param name: name (reshape, transpose, reduce_sum, matmul, id,
        squeeze, diagonal, mul, batch_dot)
    :param inputs: inputs
    :param kwargs: arguments

    Operator suffixed by `_mm` (*transpose_mm*, *reduce_sum_mm*)
    are equivalent to the same operator without the suffix
    but takes two inputs and only changes the first one.

    Attributes `_info` summarizes the known information
    about dimensions. Many of them are empty because inserted.
    Value `1` means it was the case, `2` means it is a plain dimension.
    """

    _allowed = {
        "expand_dims",
        "transpose",
        "reduce_sum",
        "matmul",
        "id",
        "squeeze",
        "diagonal",
        "mul",
        "batch_dot",
        "transpose_mm",
        "reduce_sum_mm",
    }

    def __init__(
        self,
        full_dim: int,
        name: str,
        *inputs: List["EinsumSubOp"],
        **kwargs: Dict[str, Any],
    ):
        self.full_dim = full_dim
        self.name = name
        self.inputs = inputs
        self.kwargs = kwargs
        self._info: Dict[str, Any] = {}
        assert (
            name in EinsumSubOp._allowed
        ), f"Unexpected name {name!r}. It should be in {EinsumSubOp._allowed!r}."
        assert len(inputs) in (
            1,
            2,
        ), f"Inputs must contains 1 or 2 inputs not {len(inputs)}."
        if name == "matmul" and len(inputs) != 2:
            raise RuntimeError(
                "Inputs must contains 2 inputs not %d for operator 'matmul'."
                "" % len(inputs)
            )
        for i, inp in enumerate(inputs):
            assert isinstance(inp, (int, EinsumSubOp)), (
                "Input %d has type %r, int or EinsumSubOp is expected."
                % (  # noqa: ISC001
                    i,
                    type(inp),
                )
            )
        self._check_()

    def _check_(self):
        if self.name == "transpose":
            self._check_arg_("perm", tuple)
            perm = self.kwargs["perm"]
            assert len(perm) == len(
                set(perm)
            ), f"perm has duplicated values {perm!r} (name={self.name!r})."
            assert list(perm) != list(
                range(len(perm))
            ), f"Transpose = identity perm={perm}. It must be removed."
        elif self.name == "matmul":
            self._check_arg_("axes", tuple)
            self._check_arg_("left", tuple)
            self._check_arg_("right", tuple)
            axes = self.kwargs["axes"]
            left = self.kwargs["left"]
            right = self.kwargs["right"]
            for a in axes:
                if a in left and a in right:
                    raise RuntimeError(
                        "One axis belongs to every set (axes, left, right). "
                        "axes=%r, left=%r, right=%r." % (axes, left, right)
                    )

    def __repr__(self) -> str:
        inps = ", ".join(map(str, self.inputs))
        kw = ", ".join(f"{k}={w!r}" for k, w in self.kwargs.items())
        m = f"{self.__class__.__name__}({self.name!r}, {inps}, {kw})"
        return m

    def dot_label(self) -> Optional[str]:
        """
        Displays some informations useful to understand the operator.
        """
        if self.name == "matmul":
            ndim = self.kwargs["ndim"]
            axes = self.kwargs["axes"]
            left = self.kwargs["left"]
            right = self.kwargs["right"]
            eq = _numpy_extended_dot_equation(ndim, ndim, axes, left, right)
            eq = eq.replace(">", "\\\\=")
            return "~" + eq
        return None

    def _check_arg_(self, name: str, typ: type, empty: bool = False):
        assert (
            name in self.kwargs
        ), f"Parameter {name!r} not found for operator {self.name!r}."
        if empty and self.kwargs[name] is None:
            return
        assert isinstance(
            self.kwargs[name], typ
        ), "Unexpected type %r for parameter %r and parameter %r." % (  # noqa: ISC001
            type(self.kwargs[name]),
            name,
            self.name,
        )

    def _check_row_(
        self,
        row: numpy.ndarray,
        inp: bool = False,
        verbose: bool = False,
    ):
        """
        Checks input or output is valid.
        """
        if verbose:
            if inp:
                print("<<" if inp else ">>", self.name, row, self.kwargs)
            else:
                print("<<" if inp else ">>", self.name, row)

    def _compute_output_row_id(
        self,
        row: numpy.ndarray,
        row2: numpy.ndarray,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert not ab, "ab option not allowed."
        self._check_row_(row, True, verbose=verbose)
        row[:] = row2[:]
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_transpose(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        if ab:
            assert row2 is not None
            self._compute_output_row_transpose(row2, verbose=verbose)
            return
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_("perm", tuple)
        assert len(self.kwargs["perm"]) == len(
            row
        ), f"Unexpected permutation {self.kwargs['perm']!r} (row={row!r})."
        perm = self.kwargs["perm"]
        cpy = row.copy()
        for i, p in enumerate(perm):
            row[i] = cpy[p]
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_transpose_mm(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert ab, "ab must be True."
        self._check_row_(row, True, verbose=verbose)
        assert row2 is not None, "transpose_mm expects a second input."
        self._compute_output_row_transpose(row, row2=None, verbose=verbose)

    def _compute_output_row_expand_dims(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert not ab, "ab option not allowed."
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_("axes", tuple)
        axes = self.kwargs["axes"]
        for axis in axes:
            assert isinstance(axis, tuple), (
                "Parameter axes of expand_dims should be a tuple of "
                "tuple, axes=%r." % axes
            )
            assert row[axis[1]] == -1, "Dimension should be -1 in row %r axis=%r." % (
                row,
                self.kwargs["axis"],
            )
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_reduce_sum(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert not ab, "ab option not allowed."
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_("axes", tuple)
        for a in self.kwargs["axes"]:
            row[a] = -1
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_reduce_sum_mm(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert ab, "ab must be true."
        self._check_row_(row2, True, verbose=verbose)
        assert row2 is not None, "reduce_sum_mm expects a second input."
        self._compute_output_row_reduce_sum(row, row2=None, verbose=verbose)

    def _compute_output_row_squeeze(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert not ab, "ab option not allowed."
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_("axes", tuple)
        for a in self.kwargs["axes"]:
            row[a] = -1
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_diagonal(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert not ab, "ab option not allowed."
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_("diag", list)
        to_remove = []
        for choice, choices in self.kwargs["diag"]:
            for ch in choices:
                if ch != choice:
                    to_remove.append(ch)
            for i in range(len(row)):
                if row[i] in choices:
                    if row[i] != choice:
                        row[i] = choice
        to_remove.sort()
        for r in to_remove:
            for i in range(len(row)):
                assert row[i] != r, (
                    "Unexpected result r=%r row=%r to_remove=%r diag=%r."
                    % (  # noqa: ISC001
                        r,
                        row,
                        to_remove,
                        self.kwargs["diag"],
                    )
                )
                if row[i] > r:
                    row[i] -= 1
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_matmul(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert ab, "ab must be True."
        assert row2 is not None, "row2 must be defined"
        self._check_row_(row, True, verbose=verbose)
        self._check_row_(row2, True, verbose=verbose)
        self._check_arg_("axes", tuple)
        self._check_arg_("left", tuple)
        self._check_arg_("right", tuple)
        self._check_arg_("ndim", int)
        assert row2 is not None, "matmul expects two inputs."
        if verbose:
            ndim = self.kwargs["ndim"]
            axes = self.kwargs["axes"]
            left = self.kwargs["left"]
            right = self.kwargs["right"]
            print(
                "    MATMUL %r @ %r axes=%r left=%r right=%r - eq=%s"
                % (
                    row,
                    row2,
                    axes,
                    left,
                    right,
                    _numpy_extended_dot_equation(ndim, ndim, axes, left, right),
                )
            )
        row2[:] = numpy.maximum(row, row2)
        for a in self.kwargs["axes"]:
            if a not in self.kwargs["right"]:
                row2[a] = -1
        self._check_row_(row2, verbose=verbose)

    def _compute_output_row_batch_dot(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert ab, "ab must be True."
        self._check_row_(row, True, verbose=verbose)
        self._check_row_(row2, True, verbose=verbose)
        self._check_arg_("batch_axes", tuple)
        self._check_arg_("keep_axes", tuple, empty=True)
        self._check_arg_("sum_axes", tuple)
        self._check_arg_("left", tuple)
        self._check_arg_("right", tuple)
        self._check_arg_("ndim", int)
        assert row2 is not None, "batch_dot expects two inputs."
        if verbose:
            batch_axes = self.kwargs["batch_axes"]
            keep_axes = self.kwargs["keep_axes"]
            sum_axes = self.kwargs["sum_axes"]
            left = self.kwargs["left"]
            right = self.kwargs["right"]
            ndim = self.kwargs["ndim"]
            print(
                "    BATCH_DOT batch_axes=%r keep_axes=%r sum_axes=%r "
                "left=%r right=%r eq=%r"
                % (
                    batch_axes,
                    keep_axes,
                    sum_axes,
                    left,
                    right,
                    _numpy_extended_dot_equation(ndim, ndim, sum_axes, left, right),
                )
            )
        row2[:] = numpy.maximum(row, row2)
        for a in self.kwargs["sum_axes"]:
            if a not in self.kwargs["right"]:
                row2[a] = -1
        self._check_row_(row2, verbose=verbose)

    def _compute_output_row_mul(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        assert ab, "ab must be True."
        self._check_row_(row, True, verbose=verbose)
        self._check_row_(row2, True, verbose=verbose)
        assert row2 is not None, "mul expects two inputs."
        if verbose:
            print(f"    MUL {row!r} @ {row2!r}")
        row2[:] = numpy.maximum(row, row2)
        self._check_row_(row2, verbose=verbose)

    def compute_output_row(
        self,
        row: numpy.ndarray,
        row2: Optional[numpy.ndarray] = None,
        ab: bool = False,
        verbose: bool = False,
    ):
        """
        Updates *row* based on the operator.
        """
        method_name = f"_compute_output_row_{self.name}"
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(
                f"compute_output_row not implemented for {self.name!r}."
            )
        if verbose and ab:
            print("  -- called as a binary operator")
        self.add_info(i_row=single_axes(row), i_row2=single_axes(row2))
        meth(row, row2=row2, ab=ab, verbose=verbose)
        self.add_info(o_row=single_axes(row), o_row2=single_axes(row2))

    def add_info(self, **kwargs: Dict[str, Any]):
        """
        Adds information to the node.

        :param kwargs: dictionary
        """
        for k, v in kwargs.items():
            assert (
                k not in self._info
            ), f"Key {k!r} already added (operator {self.name!r})."
            self._info[k] = v

    def _check_inputs_(self, n_expected: int, check_dim: bool = False):
        assert (
            len(self.inputs) == n_expected
        ), "Number of inputs must be %d not %d for operator %r." % (  # noqa: ISC001
            n_expected,
            len(self.inputs),
            self.name,
        )

    def _check_shape_(self, m: numpy.ndarray):
        assert len(m.shape) == self.full_dim, (
            "Number of dimensions %r is different from expected value "
            "%d."
            % (  # noqa: ISC001
                m.shape,
                self.full_dim,
            )
        )

    def _get_data(self, data: Dict[int, Any], key: Union[int, "EinsumSubOp"]) -> Any:
        if isinstance(key, int):
            assert key in data, f"Unable to find key {key!r} in {list(sorted(data))}"
            return data[key]
        if isinstance(key, EinsumSubOp):
            skey = id(key)
            assert skey in data, f"Unable to find key {skey!r} in {list(sorted(data))}"
            return data[skey]
        raise TypeError(f"Unexpected input type {type(key)!r}.")

    def _apply_id(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(1)
        inp = self.inputs[0]
        output = self._get_data(data, inp)
        return output

    def _apply_diagonal(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        if verbose:
            print(f"- {self.name}, shape={m.shape!r} diag={self.kwargs['diag']!r}")
        diag = self.kwargs["diag"]
        if len(diag) != 1:
            raise NotImplementedError(
                f"Not implemented with more than one duplicated indice {diag!r}."
            )
        diag0 = diag[0]
        output = numpy_diagonal(m, axis=diag0[0], axes=diag0[1])
        return output

    def _apply_expand_dims(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        if verbose:
            print(f"- {self.name}, shape={m.shape!r} axes={self.kwargs['axes']!r}")
        output = m
        for axis in reversed(self.kwargs["axes"]):
            output = numpy.expand_dims(output, axis[0])
        return output

    def _apply_transpose(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(1, True)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        if verbose:
            print(f"- {self.name}, shape={m.shape!r} perm={self.kwargs['perm']!r}")
        output = numpy.transpose(m, self.kwargs["perm"])
        self._check_shape_(output)
        return output

    def _apply_transpose_mm(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(2, True)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        if verbose:
            print(f"- {self.name}, shape={m.shape!r} perm={self.kwargs['perm']!r}")
        output = numpy.transpose(m, self.kwargs["perm"])
        self._check_shape_(output)
        return output

    def _apply_matmul(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        self._check_shape_(m1)
        self._check_shape_(m2)
        axes = self.kwargs["axes"]
        left = self.kwargs["left"]
        right = self.kwargs["right"]

        if verbose:
            print(
                "- %s, shapes=%r @ %r axes=%r left=%r right=%r"
                % (self.name, m1.shape, m2.shape, axes, left, right)
            )

        impl = kwargs.get("matmul_impl", None)
        if impl == "pyf":
            output = numpy_extended_dot_matrix(
                m1, m2, axes, left, right, verbose=verbose
            )
        elif impl == "py":
            output = numpy_extended_dot_python(
                m1, m2, axes, left, right, verbose=verbose
            )
        elif impl is None:
            output = numpy_extended_dot(m1, m2, axes, left, right, verbose=verbose)
        else:
            raise ValueError(f"Unknown implementation of numpy_extended_dot ({impl}).")
        self._check_shape_(output)
        return output

    def _apply_mul(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        self._check_shape_(m1)
        self._check_shape_(m2)

        if verbose:
            print(f"- {self.name}, shapes={m1.shape!r} @ {m2.shape!r}")

        output = m1 * m2
        self._check_shape_(output)
        return output

    def _apply_batch_dot(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        self._check_shape_(m1)
        self._check_shape_(m2)
        batch_axes = self.kwargs["batch_axes"]
        keep_axes = self.kwargs["keep_axes"]
        sum_axes = self.kwargs["sum_axes"]
        left = self.kwargs["left"]
        right = self.kwargs["right"]

        if verbose:
            print(
                "- %s, shapes=%r @ %r batch_axes=%r keep_axes=%r "
                "sum_axes=%r"
                % (self.name, m1.shape, m2.shape, batch_axes, keep_axes, sum_axes)
            )

        assert len(m1.shape) == len(m2.shape), (
            "batch_dot only work with two tensors with the same number "
            "of dimensions not %r @ %r." % (m1.shape, m2.shape)
        )

        dim0 = int(numpy.prod([m1.shape[i] for i in batch_axes]))
        dim0b = int(numpy.prod([m2.shape[i] for i in batch_axes]))
        dimb = int(
            -1 if keep_axes is None else numpy.prod([m1.shape[i] for i in keep_axes])
        )
        dim1 = int(numpy.prod([m1.shape[i] for i in sum_axes]))
        dim2 = int(numpy.prod([m2.shape[i] for i in sum_axes]))

        if verbose:
            print(f"- {self.name}, reshape={m1.shape!r} into {dim0, dimb, dim1!r}")
            print(f"- {self.name}, reshape={m2.shape!r} into {dim0b, dimb, dim2!r}")
        m1sh = m1.reshape((dim0, dimb, dim1))
        m2sh = m2.reshape((dim0b, dimb, dim2))

        batch_kind = self.get_dot_kind()
        if batch_kind in ("11", "N1", "N1"):
            m1sh = m1sh.reshape((-1, m1sh.shape[-1]))
            m2sh = m2sh.reshape((-1, m2sh.shape[-1]))
            if verbose:
                print(
                    "- %s, use gemm with shape %r, %r"
                    % (self.name, m1sh.shape, m2sh.shape)
                )
            dot = gemm_dot(m1sh, m2sh, False, True)
        else:
            dot = m1sh @ numpy.transpose(m2sh, (0, 2, 1))

        # new shape
        new_shape = (
            [max(m1.shape[i], m2.shape[i]) for i in batch_axes]
            + [m1.shape[i] for i in left if i not in batch_axes]
            + [m2.shape[i] for i in right if i not in batch_axes]
        )
        while len(new_shape) < len(m1.shape):
            new_shape.append(1)

        if verbose:
            taken = set(batch_axes) | set(sum_axes)
            ax = [i for i in range(len(m1.shape)) if i not in taken]
            print(
                "- %s, shapes=%r @ %r -> %r"
                % (self.name, m1sh.shape, m2sh.shape, dot.shape)
            )
            print(
                "- %s, batch_axes=%r ax=%r new_shape=%r left=%r right=%r"
                % (self.name, batch_axes, ax, new_shape, left, right)
            )

        output = dot.reshape(tuple(new_shape))
        self._check_shape_(output)
        return output

    def _apply_reduce_sum(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        axes = self.kwargs["axes"]
        if verbose:
            print(f"- {self.name}, shape={m.shape!r} axes={self.kwargs['axes']!r}")
        output = numpy.sum(m, axis=axes, keepdims=True)
        self._check_shape_(output)
        return output

    def _apply_reduce_sum_mm(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(2, True)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        if verbose:
            print(f"- {self.name}, shape={m.shape!r} axes={self.kwargs['axes']!r}")
        output = numpy.sum(m, self.kwargs["axes"])
        self._check_shape_(output)
        return output

    def _apply_squeeze(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        axes = self.kwargs["axes"]
        if verbose:
            print(f"- {self.name}, shape={m.shape!r} axes={self.kwargs['axes']!r}")
        output = m
        for a in axes[::-1]:
            output = numpy.squeeze(output, axis=a)
        return output

    def apply(
        self, data: Dict[int, Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Any:
        """
        Applies one operator on the data.

        :param data: dictionary storing the results
        :param verbose: prints out intermediate results
        :param kwargs: additional parameters, see
            methods `_apply*`
        :return: output

        Known additional paramaters:

        * 'matmul_impl': if None calls :func:`numpy.einsum` through
          :func:`numpy_extended_dot
          <onnx_extended.tools.einsum.einsum_impl_ext.numpy_extended_dot>`
          (default) or 'py' to call
          :func:`numpy_extended_dot_python
          <onnx_extended.tools.einsum.einsum_impl_ext.numpy_extended_dot_python>`
          instead.
        """
        if verbose:
            print()
            print(
                "apply %r  (%s)."
                % (self.name, ", ".join([str(id(s)) for s in self.inputs]))
            )

        method_name = f"_apply_{self.name}"
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(f"apply not implemented for {self.name!r}.")
        output = meth(data, verbose, **kwargs)

        data[id(self)] = output
        if verbose:
            print("+ %s, shape=%r -- %d" % (self.name, output.shape, id(self)))
        return output

    def _onnx_name(self) -> str:
        return f"einsum{id(self)}_{self.name}"

    def _check_onnx_opset_(self, opset: Optional[int], limit: int):
        if opset is not None and opset < limit:
            raise RuntimeError(
                f"Opset ({opset!r}) must be >= {limit!r} for operator {self.name!r}."
            )

    def _to_onnx_id(
        self,
        names: List[str],
        opset: Optional[int],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterable[NodeProto]:
        self._check_inputs_(1)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        yield helper.make_node("Identity", [name], [self._onnx_name()])

    def _to_onnx_expand_dims(
        self,
        names: List[str],
        opset: Optional[int],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterable[NodeProto]:
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs["axes"]
        name_axes = name + "_axes"
        yield numpy_helper.from_array(
            numpy.array([a[1] for a in axes], dtype=numpy.int64), name=name_axes
        )
        s_axes = "".join(map(str, [a[1] for a in axes]))
        yield helper.make_node(
            "Unsqueeze",
            [name, name_axes],
            [self._onnx_name()],
            name="Unsqueeze%s_%d" % (s_axes, id(self)),
        )

    def _to_onnx_squeeze(
        self,
        names: List[str],
        opset: Optional[int],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterable[NodeProto]:
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs["axes"]
        name_axes = name + "_axes"
        yield numpy_helper.from_array(
            numpy.array(axes, dtype=numpy.int64), name=name_axes
        )
        s_axes = "".join(map(str, axes))
        yield helper.make_node(
            "Squeeze",
            [name, name_axes],
            [self._onnx_name()],
            name="Squeeze%s_%d" % (s_axes, id(self)),
        )

    def _to_onnx_transpose(
        self,
        names: List[str],
        opset: Optional[int],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterable[NodeProto]:
        self._check_inputs_(1)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        perm = self.kwargs["perm"]
        s_perm = "".join(map(str, perm))
        yield helper.make_node(
            "Transpose",
            [name],
            [self._onnx_name()],
            perm=perm,
            name="Transpose%s_%d" % (s_perm, id(self)),
        )

    def _to_onnx_reduce_sum(
        self,
        names: List[str],
        opset: Optional[int],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterable[NodeProto]:
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs["axes"]
        name_axes = self._onnx_name() + "_axes"
        yield numpy_helper.from_array(
            numpy.array(axes, dtype=numpy.int64), name=name_axes
        )
        s_axes = "".join(map(str, axes))
        yield helper.make_node(
            "ReduceSum",
            [name, name_axes],
            [self._onnx_name()],
            keepdims=1,
            name="ReduceSum%s_%d" % (s_axes, id(self)),
        )

    def _to_onnx_mul(
        self, data: List[Any], verbose: bool = False, **kwargs: Dict[str, Any]
    ) -> Iterable[NodeProto]:
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        yield helper.make_node("Mul", [m1, m2], [self._onnx_name()])

    def _to_onnx_batch_dot(
        self,
        names: List[str],
        opset: Optional[int],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterable[NodeProto]:
        self._check_inputs_(2)
        self._check_onnx_opset_(opset, 13)
        inp1, inp2 = self.inputs[:2]
        name1 = self._get_data(names, inp1)
        name2 = self._get_data(names, inp2)

        batch_axes = self.kwargs["batch_axes"]
        keep_axes = self.kwargs["keep_axes"]
        sum_axes = self.kwargs["sum_axes"]
        left = self.kwargs["left"]
        right = self.kwargs["right"]
        root = self._onnx_name()

        def return_name_one():
            name_one = root + "_1"
            return name_one, numpy_helper.from_array(
                numpy.array([1], dtype=numpy.int64), name=name_one
            )

        name_one = None
        name_shape1 = root + "_shape1"
        name_shape2 = root + "_shape2"
        concat_left = []
        concat_right = []
        yield helper.make_node("Shape", [name1], [name_shape1])
        yield helper.make_node("Shape", [name2], [name_shape2])

        if len(batch_axes) > 0:
            name_batch_axes = root + "_batch_axes"
            yield numpy_helper.from_array(
                numpy.array(batch_axes, dtype=numpy.int64), name=name_batch_axes
            )

        if len(sum_axes) > 0:
            name_sum_axes = root + "_sum_axes"
            yield numpy_helper.from_array(
                numpy.array(sum_axes, dtype=numpy.int64), name=name_sum_axes
            )

        # dim0 = int(numpy.prod([m1.shape[i] for i in batch_axes]))
        # dim0b = int(numpy.prod([m2.shape[i] for i in batch_axes]))
        if len(batch_axes) > 1:
            name_dim0 = root + "_dim0"
            name_dim0b = root + "_dim0b"
            name_dim0g = name_dim0 + "g"
            name_dim0bg = name_dim0b + "g"
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)
            yield helper.make_node(
                "Gather", [name_shape1, name_batch_axes], [name_dim0g]
            )
            yield helper.make_node(
                "Gather", [name_shape2, name_batch_axes], [name_dim0bg]
            )
            yield helper.make_node("ReduceProd", [name_dim0g], [name_dim0], keepdims=1)
            yield helper.make_node(
                "ReduceProd", [name_dim0bg], [name_dim0b], keepdims=1
            )
        elif len(batch_axes) == 1:
            name_dim0g = root + "_dim0g"
            name_dim0bg = root + "_dim0bg"
            name_dim0 = name_dim0g
            name_dim0b = name_dim0bg
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)
            yield helper.make_node(
                "Gather", [name_shape1, name_batch_axes], [name_dim0g]
            )
            yield helper.make_node(
                "Gather", [name_shape2, name_batch_axes], [name_dim0bg]
            )
        else:
            if name_one is None:
                name_one, cst_init = return_name_one()
                yield cst_init
            name_dim0 = name_one
            name_dim0b = name_one
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)

        # dimb = int(-1 if keep_axes is None else numpy.prod(
        #     [m1.shape[i] for i in keep_axes]))
        if keep_axes in (-1, None) or len(keep_axes) == 0:
            name_dimb = root + "__1"
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                numpy.array([-1], dtype=numpy.int64), name=name_dimb
            )
        elif len(keep_axes) == 1:
            name_keep_axes = root + "_keep_axes"
            name_dimb = root + "_dimb"
            name_dimbg = name_dimb
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                numpy.array(keep_axes, dtype=numpy.int64), name=name_keep_axes
            )
            yield helper.make_node(
                "Gather", [name_shape1, name_keep_axes], [name_dimbg]
            )
        else:
            name_keep_axes = root + "_keep_axes"
            name_dimb = root + "_dimb"
            name_dimbg = name_dimb + "g"
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                numpy.array(keep_axes, dtype=numpy.int64), name=name_keep_axes
            )
            yield helper.make_node(
                "Gather", [name_shape1, name_keep_axes], [name_dimbg]
            )
            yield helper.make_node("ReduceProd", [name_dimbg], [name_dimb], keepdims=1)

        # dim1 = int(numpy.prod([m1.shape[i] for i in sum_axes]))
        # dim2 = int(numpy.prod([m2.shape[i] for i in sum_axes]))

        if len(sum_axes) == 0:
            if name_one is None:
                name_one, cst_init = return_name_one()
                yield cst_init
            name_dim1 = name_one
            name_dim2 = name_one
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
        elif len(sum_axes) == 1:
            name_dim1 = root + "_dim1"
            name_dim2 = root + "_dim2"
            name_dim1g = name_dim1
            name_dim2g = name_dim2
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
            yield helper.make_node("Gather", [name_shape1, name_sum_axes], [name_dim1g])
            yield helper.make_node("Gather", [name_shape2, name_sum_axes], [name_dim2g])
        else:
            name_dim1 = root + "_dim1"
            name_dim2 = root + "_dim2"
            name_dim1g = name_dim1 + "g"
            name_dim2g = name_dim2 + "g"
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
            yield helper.make_node("Gather", [name_shape1, name_sum_axes], [name_dim1g])
            yield helper.make_node("Gather", [name_shape2, name_sum_axes], [name_dim2g])
            yield helper.make_node("ReduceProd", [name_dim1g], [name_dim1], keepdims=1)
            yield helper.make_node("ReduceProd", [name_dim2g], [name_dim2], keepdims=1)

        batch_kind = self.get_dot_kind()
        if batch_kind in ("11", "N1", "N1"):
            # *shape1, *shape2
            name_minus_one = root + "__01"
            yield numpy_helper.from_array(
                numpy.array([-1], dtype=numpy.int64), name=name_minus_one
            )
            name_agg_shape1_2 = root + f"_resh1_{batch_kind}"
            name_agg_shape2_2 = root + f"_resh2_{batch_kind}"
            yield helper.make_node(
                "Concat", [name_minus_one, name_dim1], [name_agg_shape1_2], axis=0
            )
            yield helper.make_node(
                "Concat", [name_minus_one, name_dim2], [name_agg_shape2_2], axis=0
            )

            # m1sh = m1.reshape((-1, dim1))
            # m2sh = m2.reshape((-1, dim2))
            name_agg1_2 = root + "_aresh1"
            name_agg2_2 = root + "_aresh2"
            yield helper.make_node("Reshape", [name1, name_agg_shape1_2], [name_agg1_2])
            yield helper.make_node("Reshape", [name2, name_agg_shape2_2], [name_agg2_2])

            # dot = gemm(m1sh, m2sh, False, True)
            name_dot = root + "_gemm"
            yield helper.make_node(
                "Gemm",
                [name_agg1_2, name_agg2_2],
                [name_dot],
                alpha=1.0,
                beta=0.0,
                transA=0,
                transB=1,
            )
        else:
            # *shape1, *shape2
            name_agg_shape1 = root + "_resh1"
            name_agg_shape2 = root + "_resh2"
            yield helper.make_node("Concat", concat_left, [name_agg_shape1], axis=0)
            yield helper.make_node("Concat", concat_right, [name_agg_shape2], axis=0)

            # m1sh = m1.reshape((dim0, dimb, dim1))
            # m2sh = m2.reshape((dim0b, dimb, dim2))
            name_agg1 = root + "_aresh1"
            name_agg2 = root + "_aresh2"
            yield helper.make_node("Reshape", [name1, name_agg_shape1], [name_agg1])
            yield helper.make_node("Reshape", [name2, name_agg_shape2], [name_agg2])

            # dot = m1sh @ numpy.transpose(m2sh, (0, 2, 1))
            name_agg2_tr = root + "_aresh2_tr"
            yield helper.make_node(
                "Transpose",
                [name_agg2],
                [name_agg2_tr],
                perm=[0, 2, 1],
                name=f"Transpose021_{id(self)}",
            )

            name_dot = root + "_dot"
            yield helper.make_node("MatMul", [name_agg1, name_agg2_tr], [name_dot])

        # new_shape = ([max(m1.shape[i], m2.shape[i]) for i in batch_axes] +
        #      [m1.shape[i] for i in left if i not in batch_axes] +
        #      [m2.shape[i] for i in right if i not in batch_axes])
        concat_final = []
        if len(batch_axes) > 0:
            name_max_dim = root + "_max_dim"
            concat_final.append(name_max_dim)
            yield helper.make_node("Max", [name_dim0g, name_dim0bg], [name_max_dim])

        left_set = list(sorted(set(left) - (set(batch_axes) & set(left))))
        if len(left_set) > 0:
            name_left_dim = root + "_left_dim"
            name_left_set = root + "_left_set"
            yield numpy_helper.from_array(
                numpy.array(left_set, dtype=numpy.int64), name=name_left_set
            )
            yield helper.make_node(
                "Gather", [name_shape1, name_left_set], [name_left_dim]
            )
            concat_final.append(name_left_dim)

        right_set = list(sorted(set(right) - (set(batch_axes) & set(right))))
        if len(right_set) > 0:
            name_right_dim = root + "_right_dim"
            name_right_set = root + "_right_set"
            yield numpy_helper.from_array(
                numpy.array(right_set, dtype=numpy.int64), name=name_right_set
            )
            yield helper.make_node(
                "Gather", [name_shape2, name_right_set], [name_right_dim]
            )
            concat_final.append(name_right_dim)

        name_new_shape = root + "_new_shape"
        diff = self.full_dim - (len(batch_axes) + len(left_set) + len(right_set))
        if diff > 0:
            names_ones = root + "_ones"
            yield numpy_helper.from_array(
                numpy.array([1 for i in range(diff)], dtype=numpy.int64),
                name=names_ones,
            )
            concat_final.append(names_ones)

        yield helper.make_node("Concat", concat_final, [name_new_shape], axis=0)

        name_final = root + "_final"
        yield helper.make_node("Reshape", [name_dot, name_new_shape], [name_final])

    def to_onnx(
        self,
        names: List[str],
        opset: Optional[int],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterable[NodeProto]:
        """
        Converts this node into ONNX. Enumerates all ONNX node
        which participate to the conversion. The last one
        is the final output.

        :param names: dictionary where to find already converted name
        :param opset: opset
        :param verbose: prints out intermediate results
        :param kwargs: additional parameter for the conversion
        :return: output
        """
        if opset is None:
            opset = DEFAULT_OPSET
        if verbose:
            print()
            print(
                "to_onnx %r  (%s) opset=%r."
                % (self.name, ", ".join([str(id(s)) for s in self.inputs]), opset)
            )

        method_name = f"_to_onnx_{self.name}"
        meth = getattr(self, method_name, None)
        if meth is None:
            if self.name.endswith("_mm"):
                raise NotImplementedError(
                    "to_onnx not implemented for %r."
                    "You should call method simplify_mm_nodes "
                    "to remove it." % self.name
                )
            raise NotImplementedError(f"to_onnx not implemented for {self.name!r}.")
        for node in meth(names, verbose=verbose, opset=opset, **kwargs):
            if hasattr(node, "output"):
                names[id(self)] = node.output[0]
                if verbose:
                    print(
                        "+ OP %r -- (%s - %d)" % (node.output[0], self.name, id(self))
                    )
            elif verbose:
                # Initializer
                print("+ CT %r -- (%s - %d)" % (node.name, self.name, id(self)))
            yield node

    def get_dot_kind(self) -> str:
        """
        Every matrix multiplication can be either:

        * a simple multiplication (`M`) (undetected)
        * a 2D matrix multiplication (`11`)
        * a broadcasted matrix multiplication (`N1` or `1N`)
        * a batch matrix multiplication (`NN`)

        This method returns which kind it is.
        """
        batch_axes = self.kwargs["batch_axes"]
        # keep_axes = self.kwargs['keep_axes']
        # sum_axes = self.kwargs['sum_axes']
        # left = self.kwargs['left']
        # right = self.kwargs['right']
        info = self._info
        row_left = info["i_row"]
        row_right = info["i_row2"]

        batch_left = [row_left[k] for k in batch_axes]
        batch_right = [row_right[k] for k in batch_axes]
        n_left = len(batch_left) > 0 and max(batch_left) == 2
        n_right = len(batch_right) > 0 and max(batch_right) == 2
        return f"{'N' if n_left else '1'}{'N' if n_right else '1'}"


class GraphEinsumSubOp:
    """
    Class gathering all nodes produced to explicit einsum
    operators.

    :param letters: list of distinct letters
    :param mat: matrix, see :func:`analyse_einsum_equation
        <onnx_extended.tools.einsum.einsum_impl.analyse_einsum_equation>`
    :param lengths: lengths of every input
    :param duplicates: see :func:`analyse_einsum_equation
        <onnx_extended.tools.einsum.einsum_impl.analyse_einsum_equation>`
    """

    def __init__(
        self,
        letters: str,
        mat: numpy.ndarray,
        lengths: List[int],
        duplicates: List[Dict[str, int]],
    ):
        self._nodes: Dict[int, Union[int, EinsumSubOp]] = {}
        self._mark: Dict[int, Union[int, EinsumSubOp]] = {}
        self._ops: List[EinsumSubOp] = []
        self._inputs: Dict[int, Union[int, EinsumSubOp]] = {}
        self.last_op: Optional[Union[int, EinsumSubOp]] = None
        self.last_added_op: Optional[Union[int, EinsumSubOp]] = None
        self.metadata = dict(
            letters=letters,
            mat=mat,
            lengths=lengths,
            mat0=mat.copy(),
            duplicates=duplicates,
        )

    def append(self, op: Union[int, EinsumSubOp]) -> Optional[EinsumSubOp]:
        """
        Adds one input or result.

        :param op: integer (an input) or an instance of :class:`EinsumSubOp
            <onnx_extended.tools.einsum.einsum_impl_classes.EinsumSubOp>`.
        :return: op or None if op is an integer
        """
        if isinstance(op, int):
            assert op not in self._nodes, f"Key {op!r} already added."
            self._nodes[op] = op
            self.last_added_op = op
            self._inputs[op] = op
            return None
        if isinstance(op, EinsumSubOp):
            assert op not in self._nodes, f"Key {id(op)} already added, op={op!r}."
            self._nodes[id(op)] = op
            self._ops.append(op)
            self.last_added_op = op
            return op
        raise TypeError(f"Unexpected type {type(op)!r}.")

    def mark_last_node(self):
        """
        Marks the last node as the final output.
        """
        assert self.last_added_op is not None, "last_added_op is None."
        self.mark(-1, self.last_added_op)

    def mark(self, i: int, op: EinsumSubOp):
        """
        Marks one input or result as an intermediate result
        after a full einsum step.

        :param i: a position
        :param op: an instance of :class:`EinsumSubOp
            <onnx_extended.tools.einsum.einsum_impl_classes.EinsumSubOp>`.
        """
        assert isinstance(i, int), f"i must an integer not {type(i)!r}."
        if i != -1 and i not in self._inputs:
            raise RuntimeError("Input %d was not registered in %r." % (i, self._inputs))
        if isinstance(op, EinsumSubOp):
            assert id(op) in self._nodes, "Key %d not found, op=%r." % (id(op), op)
            self._mark[i] = op
            self._mark[id(op)] = i
            self.last_op = op
        else:
            raise TypeError(f"Unexpected type {type(i)!r}.")

    def __iter__(self) -> Iterator[EinsumSubOp]:
        "Iterates on nodes."
        yield from self._ops

    def to_dot(self, **kwargs: Dict[str, Any]) -> str:
        """
        Produces a graph in :epkg:`dot`.

        :param kwargs: additional graph option
        :return: string
        """
        options = {
            "orientation": "portrait",
            "ranksep": "0.25",
            "nodesep": "0.05",
            "width": "0.5",
            "height": "0.1",
            "size": "5",
            "node": "[shape=record]",
        }
        options.update(kwargs)

        def d2s(d):
            it = []
            for k, v in sorted(d.items()):
                it.append(f"{k}={v}")
            return " ".join(it)

        def d2sd(d):
            it = []
            for k, v in sorted(d.items()):
                if len(v) > 1:
                    it.append(f"{k}={','.join(map(str, v))}")
            return " ".join(it)

        rows = ["digraph{"]
        for k, v in options.items():
            if isinstance(v, str) and "[" in v:
                rows.append(f"{k} {v};")
            else:
                rows.append(f"{k}={v};")
        for k, v in self._nodes.items():
            if isinstance(v, int):
                let = [
                    (r, self.metadata["letters"][i])
                    for i, r in enumerate(self.metadata["mat0"][v])
                    if r != -1
                ]
                dup = self.metadata["duplicates"][v]
                if dup is None:
                    dup = ""
                else:
                    dup = f" - {d2sd(dup)}"
                let.sort()
                letters = "".join(_[1] for _ in let)
                lab = "input %d\\\\n%s\\\\n%s%s" % (
                    v,
                    letters,
                    str(self.metadata["mat0"][v]),
                    dup,
                )
                sk = v
                extended_lab = ""
            else:
                lab = f"{v.name}\\\\n{d2s(v.kwargs)}"
                sk = id(v)
                extended_lab = v.dot_label()
                if extended_lab:
                    extended_lab = "\\\\n" + extended_lab

            if sk in self._mark and isinstance(self._mark[sk], int):
                la = self._mark[sk]
                lab = lab.replace("\\\\n", " - I%d\\\\n" % la)
                s = f'{k} [label="{lab}{extended_lab}" style=filled fillcolor=red];'
            else:
                s = f'{k} [label="{lab}{extended_lab}"];'
            rows.append(s)
            if not hasattr(v, "inputs"):
                continue
            for i in v.inputs:
                vid = i if isinstance(i, int) else id(i)
                s = "%d -> %d;" % (vid, k)
                rows.append(s)
        rows.append("}")
        return "\n".join(rows)

    def apply_sequence(
        self,
        *inputs: List[EinsumSubOp],
        verbose: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Any:
        """
        Applies a sequence of operations on a list of inputs.

        :param inputs: inputs
        :param verbose: prints out intermediate results
        :param kwargs: additional parameters,
            see :meth:`apply
            <onnx_extended.tools.einsum.einsum_impl_classes.EinsumSubOp.apply>`.
        :return: output
        """
        if verbose:
            print("######### apply_sequence")
        data = dict(enumerate(inputs))
        last = None
        for op in self:
            last = op.apply(data, verbose=verbose, **kwargs)
        assert last is not None, "Sequence of operations is empty."
        return last

    def clean_unused_nodes(self, verbose: bool = False):
        """
        Cleans nodes with unused outputs.

        :param verbose: display intermediate information
        """

        def iteration(it):
            # Walks through all nodes.
            is_used = {}
            for node in self._ops:
                if not isinstance(node, EinsumSubOp):
                    continue
                if id(node) not in is_used:
                    is_used[id(node)] = []
                for inp in node.inputs:
                    if not isinstance(inp, EinsumSubOp):
                        continue
                    idn = id(inp)
                    if idn not in is_used:
                        is_used[idn] = []
                    is_used[idn].append(id(node))

            # Remove unused nodes.
            removed = []
            for k, v in is_used.items():
                if len(v) == 0:
                    removed.append(k)
            removed = set(removed)
            i_rem = []
            for i, op in enumerate(self._ops):
                if not isinstance(op, EinsumSubOp):
                    continue
                if id(op) in removed and id(op) not in self._mark:
                    i_rem.append((i, id(op)))
            for i, idn in reversed(i_rem):
                if verbose:
                    print(
                        "[GraphEinsumSubOp.clean_nodes] remove node "
                        "i=%d: %d - id=%d" % (it, i, idn)
                    )
                del self._ops[i]
                del self._nodes[idn]
            return len(i_rem) > 0

        it = 1
        while iteration(it):
            it += 1

        self.last_op = None
        self.last_added_op = None

    def simplify_mm_nodes(self, verbose: bool = False):
        """
        Node name suffixed by `mm` are an artifact to keep
        the graph consistent while building it. They can
        now be replaced by the equivalent node without suffix `mm`.

        :param verbose: display intermediate information
        """
        for op in self:
            if not isinstance(op, EinsumSubOp):
                continue
            if op.name.endswith("_mm"):
                if verbose:
                    print(
                        "[GraphEinsumSubOp.simplify_mm_nodes] node %r"
                        " - id=%d" % (op.name, id(op))
                    )
                assert (
                    len(op.inputs) == 2
                ), "Expecting 2 inputs for node %r not %r id=%r." % (
                    op.name,
                    len(op.inputs),
                    id(op),
                )
                op.name = op.name[:-3]
                op.inputs = op.inputs[:1]

    def _get_forward_nodes(self) -> Dict[int, EinsumSubOp]:
        """
        Returns the forward nodes.
        """
        forward: Dict[int, EinsumSubOp] = {}
        for op in self:
            if isinstance(op, int):
                continue
            for inp in op.inputs:
                key = inp if isinstance(inp, int) else id(inp)
                if key in forward:
                    forward[key].append(op)
                else:
                    forward[key] = [op]
        return forward

    def _pprint_forward(self) -> str:
        rows = []
        for op in self:
            line = "%r <- %s(%s)" % (
                id(op),
                op.name,
                ", ".join(map(str, [id(_) for _ in op.inputs])),
            )
            rows.append(line)
        return "\n".join(rows)

    def _replace_node_sequence(
        self, added: Optional[EinsumSubOp], deleted: List[EinsumSubOp]
    ):
        """
        Removes a sequence of nodes. The method does not check
        that the graph remains consistent.
        """
        forward = self._get_forward_nodes()
        key = id(deleted[-1])
        assert key in forward, (
            "Key {} missing in all forward nodes (other keys {}), "
            "all keys:\n{}".format(
                key, [id(_) for _ in deleted], self._pprint_forward()
            )
        )

        # deletion
        mark_input = None
        for d in deleted:
            del self._nodes[id(d)]
            if id(d) in self._mark:
                del self._mark[id(d)]
                dels = []
                for k, v in self._mark.items():
                    if id(v) == id(d):
                        mark_input = k
                        dels.append(k)
                assert len(dels) == 1, (
                    "Input %d has more than one marked operator "
                    "(%r)."
                    % (  # noqa: ISC001
                        id(d),
                        dels,
                    )
                )
                del self._mark[dels[0]]

        dels = set(id(o) for o in deleted)
        rem = []
        for i, op in enumerate(self._ops):
            if id(op) in dels:
                rem.append(i)
        assert len(rem) == len(
            deleted
        ), f"Mismatched length {rem!r}, {dels!r}, len={len(deleted)!r}."
        for i in reversed(rem):
            del self._ops[i]
        self.last_add_op = None

        # insertion
        if added is not None:
            self._ops.insert(rem[0], added)
            self._nodes[id(added)] = added
            for op in forward[key]:
                new_inputs = list(op.inputs)
                for i in range(len(op.inputs)):
                    if id(op.inputs[i]) == key:
                        new_inputs[i] = added
                op.inputs = tuple(new_inputs)
            if mark_input is not None:
                self.mark(mark_input, added)
        else:
            inps = deleted[0].inputs
            assert len(inps) == 1, "More than one input. Call another method."
            inp = inps[0]
            for op in forward[key]:
                new_inputs = list(op.inputs)
                for i in range(len(op.inputs)):
                    if id(op.inputs[i]) == key:
                        new_inputs[i] = inp
                op.inputs = tuple(new_inputs)
            if mark_input is not None:
                self.mark(mark_input, inp)

    def remove_duplicate_transpose(self, verbose: bool = False):
        """
        Removes consecutive transpose by merging them.

        :param verbose: display intermediate information
        """
        modif = 1
        while modif > 0:
            modif = 0
            candidates = []
            forward = self._get_forward_nodes()
            for op in self:
                if op.name == "transpose":
                    inp = op.inputs[0]
                    if (
                        isinstance(inp, EinsumSubOp)
                        and inp.name == "transpose"
                        and len(forward[id(inp)]) == 1
                    ):
                        candidates.append(op)

            if len(candidates) > 0:
                modif = 1
                # Not efficient to take the first one and to
                # start again but the graph should not be too big.
                cand = candidates[0]
                op2 = cand
                op1 = cand.inputs[0]
                perm1 = op1.kwargs["perm"]
                perm2 = op2.kwargs["perm"]
                assert len(perm1) == len(perm2), (
                    "Transposition should have the same length "
                    "%r, %r."
                    % (  # noqa: ISC001
                        perm1,
                        perm2,
                    )
                )
                perm = list(perm1)
                for i in range(len(perm)):
                    perm[i] = perm1[perm2[i]]
                if list(range(len(perm))) == perm:
                    # identity, everything needs to be removed
                    new_op = None
                else:
                    new_op = op2.__class__(
                        op2.full_dim, op2.name, op1.inputs[0], perm=tuple(perm)
                    )
                self._replace_node_sequence(new_op, [op1, op2])  # type: ignore[list-item]
                if verbose:
                    print(
                        "[GraphEinsumSubOp.remove_duplicate_transpose] remove nodes %r"
                        " - id=%d,%d + %d perm1=%r perm2=%r -> perm=%r"
                        % (
                            op2.name,
                            id(op1),
                            id(op2),
                            id(new_op) if new_op is not None else -1,
                            perm1,
                            perm2,
                            perm,
                        )
                    )

    def to_onnx(
        self,
        output: str,
        *inputs: List[str],
        dtype: Optional[Any] = None,
        verbose: bool = False,
        opset: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> ModelProto:
        """
        Converts the graph into ONNX.

        :param output: output name
        :param inputs: input names
        :param dtype: type used for all operators
        :param opset: desired opset, None for the last one
        :param verbose: display intermediate operators
        :param kwargs: additional parameter to use when building
            the ONNX graph, list of supported parameters:
            *name*, *ir_version*, *producer_name*,
            *producer_version*, *initializer*
        :return: ONNX graph

        Not all graphs can be converted into ONNX. Only graphs produced
        with `strategy='numpy'` can be converted otherwise the following
        error shows up:

        ::

            NotImplementedError: to_onnx not implemented for 'matmul'.
        """
        from ..onnx_nodes import onnx_remove_node_unused

        # inputs
        if opset is None:
            opset = DEFAULT_OPSET
        if verbose:
            print(
                "[GraphEinsumSubOp.to_onnx] %r -> %s opset=%r "
                "dtype=%r" % (inputs, output, opset, dtype)
            )
        onx_inputs = []
        proto = guess_proto_dtype(numpy.float32 if dtype is None else dtype)
        lengths = self.metadata["lengths"]
        names: Dict[int, str] = {}
        for inp, le in zip(inputs, lengths):
            if isinstance(inp, tuple):
                name, (typ, shape) = inp
                assert le == len(shape), (
                    "Irreconcialable shapes for input %r: "
                    "%r != len(%r)."
                    % (  # noqa: ISC001
                        name,
                        le,
                        typ.shape,
                    )
                )
                onx_inputs.append(helper.make_tensor_value_info(name, typ, shape))
                names[len(names)] = name
            else:
                onx_inputs.append(
                    helper.make_tensor_value_info(inp, proto, [None for i in range(le)])
                )
                names[len(names)] = inp

        # output
        onx_output = helper.make_tensor_value_info(
            output, proto, [None for i in range(lengths[-1])]
        )

        # nodes
        nodes = []
        inits: List[TensorProto] = []
        if "initializer" in kwargs:
            inits.extend(kwargs["initializer"])
        for op in self:
            for onx_node in op.to_onnx(names, verbose=verbose, opset=opset):
                if hasattr(onx_node, "output"):
                    nodes.append(onx_node)
                else:
                    inits.append(onx_node)

        # last node
        last_node = nodes[-1]
        nodes.append(helper.make_node("Identity", [last_node.output[0]], [output]))

        # Builds the graph
        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid("", opset)],
            ir_version=kwargs.get("ir_version", DEFAULT_IR_VERSION),
            producer_name=kwargs.get("producer_name", "onnx_extended"),
            producer_version=kwargs.get("producer_version", "0.0.dev"),
            graph=helper.make_graph(
                name=kwargs.get("name", "einsum"),
                inputs=onx_inputs,
                outputs=[onx_output],
                initializer=inits,
                nodes=nodes,
            ),
        )

        return onnx_remove_node_unused(model)
