from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy
from .einsum_impl_classes import EinsumSubOp, GraphEinsumSubOp


def analyse_einsum_equation(
    equation: str,
) -> Tuple[str, numpy.ndarray, List[int], List[Optional[Dict[str, List[int]]]]]:
    """
    Analyses an einsum equation.

    :param equation: :func:`numpy.einsum` equation
    :return: four results, list of letters,
        a matrix (see below), lengths of each components,
        duplicates

    The returned a matrix is defined as follows:

    .. math::

        m_{ij}=\\left\\{\\begin{array}{ll}-1 &
        \\text{if letter j is involved in input i} \\\\
        p & \\text{p is position of letter j in equation i}
        \\end{array}\\right.
    """
    spl = equation.strip(" ,").split("->")
    if len(spl) != 2 or not spl[1] or not spl[0]:
        raise NotImplementedError(
            "The function only implements the case when there are "
            "two sides in the equation: %r." % equation
        )
    inputs = [s.strip() for s in spl[0].split(",")]
    output = spl[1]
    all_letters = set(inputs[0])

    # Set of letters
    for inp in inputs[1:]:
        all_letters |= set(inp)
    letters = list(sorted(all_letters))
    for c in letters:
        if not (("a" <= c <= "z") or ("A" <= c <= "Z")):
            raise ValueError(
                "Equation %r must only contain lower or upper letters "
                "but %r is not." % (equation, c)
            )

    rev = {c: i for i, c in enumerate(letters)}
    for c in output:
        assert c in letters, (
            f"Output contains one unexpected letter {c!r} in equation "
            f"{equation!r}, letters={letters!r}."
        )
    mat = numpy.full((len(inputs) + 1, len(letters)), -1, dtype=numpy.int8)
    for i, inp in enumerate(inputs):
        for k, c in enumerate(inp):
            mat[i, rev[c]] = k
    for k, c in enumerate(output):
        mat[len(inputs), rev[c]] = k
    lengths = [len(inp) for inp in inputs]
    lengths.append(len(output))

    # Look for duplicates
    duplicates: List[Optional[Dict[str, List[int]]]] = []
    for inp in [*inputs, output]:
        if len(inp) == len(set(inp)):
            duplicates.append(None)
            continue
        # There is some duplicates.
        counts: Dict[str, List[int]] = {}
        for i, c in enumerate(inp):
            if c in counts:
                counts[c].append(i)
            else:
                counts[c] = [i]
        duplicates.append(counts)

    return "".join(letters), mat, lengths, duplicates


def decompose_einsum_equation(
    equation: str,
    *shapes: List[Tuple[int, ...]],
    strategy: str = "simple",
    clean: bool = False,
    verbose: bool = False,
) -> GraphEinsumSubOp:
    """
    Decomposes an equation used in :func:`numpy.einsum` knowing
    the input shapes. It returns a sequence of operations
    to do to compute the results.

    :param equation: a string
    :param shapes: sequence of input shapes
    :param strategy: there are different way to decompose the equation,
        this parameters defines the way to do it (see below)
    :param clean: clean the unnecessary node in the graph
    :param verbose: verbosity
    :return: instance of :class:`GraphEinsumSubOp
        <onnx_extended.tools.einsum.einsum_impl_classes.GraphEinsumSubOp>`

    About *strategy*:

    * `'simple'`: align all dimensions in the alphabetical order,
      some generic matrix multiplication remains implemented with
      :func:`numpy.einsum` but only with two matrices aligned on
      the same dimension (see :func:`numpy_extended_dot
      <onnx_extended.tools.einsum.einsum_impl_ext.numpy_extended_dot>`)
    * `'numpy'`: same as `simple` but the decomposition does not use
      :func:`numpy.einsum` anymore but only multiplication or
      matrix multiplication merged into a single operator called
      *batch_dot* (see :func:`numpy_extended_dot_matrix
      <onnx_extended.tools.einsum.einsum_impl_ext.numpy_extended_dot_matrix>`)

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*, *diagonal*. It analyses an equation and produces a graph
    where nodes are instance of class :class:`EinsumSubOp
    <onnx_extended.tools.einsum.einsum_impl_classes.EinsumSubOp>`.

    .. runpython::
        :showcode:

        from onnx_extended.tools.einsum import decompose_einsum_equation
        seq = decompose_einsum_equation("bac,cd,def->ebc")
        for op in seq:
            print(op)

    It can be better displayed as the following.

    .. gdot::
        :script: DOT-SECTION
        :process:

        from onnx_extended.tools.einsum import decompose_einsum_equation
        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        print("DOT-SECTION", seq.to_dot())
    """
    if shapes:
        for sh in shapes:
            if not isinstance(sh, tuple):
                raise TypeError(f"All shapes must be tuples for {sh!r} is not.")
    if strategy in ("simple", "numpy"):
        op_matmul = {"simple": "matmul", "numpy": "batch_dot"}
        graph = _decompose_einsum_equation_simple(
            equation, *shapes, verbose=verbose, op_matmul=op_matmul[strategy]
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}.")

    # Last step: clean unused nodes.
    if clean:
        last_node = graph.last_added_op
        assert isinstance(last_node, EinsumSubOp)
        graph.append(EinsumSubOp(last_node.full_dim, "id", last_node))
        graph.mark_last_node()
        graph.simplify_mm_nodes(verbose=verbose)
        graph.remove_duplicate_transpose(verbose=verbose)
        graph.clean_unused_nodes(verbose=verbose)
    else:
        graph.mark_last_node()
    return graph


def apply_einsum_sequence(
    seq: List[numpy.ndarray],
    *inputs: List[EinsumSubOp],
    verbose: bool = False,
    **kwargs: Dict[str, Any],
) -> numpy.ndarray:
    """
    Applies a sequence of operations on a list of inputs.
    The sequence of operations is produced by function
    :func:`decompose_einsum_equation`.

    :param seq: sequence of operations
    :param inputs: inputs
    :param verbose: verbosity
    :param kwargs: additional parameters,
        see `apply_sequence` in :class:`GraphEinsumSubOp
        <onnx_extended.tools.einsum.einsum_impl_classes.GraphEinsumSubOp>`
    :return: output

    .. runpython::
        :showcode:

        import numpy
        from onnx_extended.tools.einsum import (
            decompose_einsum_equation, apply_einsum_sequence)

        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = numpy.arange(4).reshape((2, 2)) + 100
        m3 = numpy.arange(8).reshape((2, 2, 2)) + 1000

        seq = decompose_einsum_equation("bac,cd,def->ebc")
        res = apply_einsum_sequence(seq, m1, m2, m3)
        print(res)
    """
    return seq.apply_sequence(*inputs, verbose=verbose, **kwargs)


def is_transpose_identity(perm: Tuple[int, ...]) -> bool:
    """
    Tells if the permutation *perm* does nothing (itentity).

    :param perm: permutation
    :return: boolean
    """
    return list(perm) == list(range(len(perm)))


def _basic_verification(
    lengths: List[int], shapes: List[Tuple[int, ...]], equation: str
):
    assert len(lengths) - 1 == len(
        shapes
    ), "Equation %r has %d inputs but %d shapes are given." % (  # noqa: ISC001
        equation,
        len(lengths),
        len(shapes),
    )
    for i, (le, sh) in enumerate(zip(lengths, shapes)):
        assert le == len(sh), (
            "Inputs %d has %d dimensions but shapes %r has %d in equation %r."
            % (  # noqa: ISC001
                i,
                le,
                sh,
                len(sh),
                equation,
            )
        )


def _apply_transpose_reshape(
    op: Union[int, EinsumSubOp], row: str
) -> Iterable[EinsumSubOp]:
    """
    Put all dimensions in the same order.

    :param op: integer (for one input) or an operator
    :param row: letter involved in this input (as a vector of binaries)
    :return: last created operator
    """
    axes = []
    p = 0
    perm = []
    for i, r in enumerate(row):
        if r == -1:
            axes.append((p, i))
        else:
            p += 1
            perm.append((r, i))
    op = EinsumSubOp(len(row), "expand_dims", op, axes=tuple(axes))
    yield op
    perm.sort()
    p = 0
    new_perm = numpy.arange(len(row))
    for i, r in enumerate(row):
        if r == -1:
            continue
        new_perm[perm[p][1]] = i
        p += 1
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row), "transpose", op, perm=tuple(new_perm))
        yield op


def _apply_squeeze_transpose(
    op: Union[int, EinsumSubOp], row_last: str, row_output: List[int]
) -> Iterable[EinsumSubOp]:
    """
    Puts output dimension in the expected order.
    """
    perm = []
    sq = []
    for i, d in enumerate(row_output):
        if d == -1:
            sq.append(i)
        else:
            perm.append((d, i))
    perm.sort()
    new_perm = numpy.arange(len(row_last))
    p = 0
    for i, d in enumerate(row_output):
        if d == -1:
            continue
        new_perm[i] = perm[p][1]
        p += 1
    perm = [p[1] for p in perm]
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row_last), "transpose", op, perm=tuple(new_perm))
        yield op
    if sq:
        op = EinsumSubOp(len(row_last), "squeeze", op, axes=tuple(sq))
        yield op


def _apply_einsum_matmul(
    fd, op1, op2, axes, left, right, ndim, op_matmul, row1, row2, verbose=False
) -> Iterable[EinsumSubOp]:
    """
    Decomposes the generic matrix multiplication into numpy operations
    depending on the operator to use for matrix multiplication
    *op_matmul* (see :func:`decompose_einsum_equation`).
    """
    allowed = {"matmul", "batch_dot", "dot"}
    assert (
        op_matmul in allowed
    ), f"Unknown operator op_matmul={op_matmul!r} not in {allowed!r}."
    if op_matmul == "matmul":
        if verbose:
            print(f"  -- MATMUL -> matmul axes={axes!r} left={left!r} right={right!r}")
        yield EinsumSubOp(
            fd, "matmul", op1, op2, axes=axes, left=left, right=right, ndim=ndim
        )

    elif len(axes) == 0 and not (set(left) & set(right)):
        if verbose:
            print(f"  -- MATMUL -> mul axes={axes!r} left={left!r} right={right!r}")
        yield EinsumSubOp(fd, "mul", op1, op2)

    elif not (set(axes) & set(left)) and not (set(axes) & set(right)):
        # No intersection between axes and right: matrix multiplication
        if verbose:
            print(
                "  -- MATMUL -> batch_dot axes=%r left=%r right=%r"
                "" % (axes, left, right)
            )

        all_axes = set(left) | set(right) | set(axes)
        common_axes = list(set(left) & set(right))
        for i in range(ndim):
            if i not in all_axes:
                common_axes.append(i)
        common_axes.sort()

        # ReduceSum*
        has_dim = set(i for i in range(len(row1)) if row1[i] >= 0)
        right_no_left = (set(right) & has_dim) - (set(right) & (set(left) | set(axes)))
        if right_no_left:
            if verbose:
                print(f"  -- MATMUL reduce1 has_dim={has_dim!r} axes={right_no_left!r}")
            op1 = EinsumSubOp(
                fd, "reduce_sum_mm", op1, op2, axes=tuple(sorted(right_no_left))
            )
            yield op1

        has_dim = set(i for i in range(len(row2)) if row2[i] >= 0)
        left_no_right = (set(left) & has_dim) - (set(left) & (set(right) | set(axes)))
        if left_no_right:
            if verbose:
                print(f"  -- MATMUL reduce2 has_dim={has_dim!r} axes={left_no_right!r}")
            op2 = EinsumSubOp(fd, "reduce_sum", op2, axes=tuple(sorted(left_no_right)))
            yield op2

        # Transpose
        i_axes = [
            (-1 if i in common_axes else (1 if i in axes else 0), i)
            for i in range(ndim)
        ]
        i_axes.sort()
        perm = [_[1] for _ in i_axes]
        perm_left = [i for i in range(len(perm)) if perm[i] in left]
        perm_right = [i for i in range(len(perm)) if perm[i] in right]
        if not is_transpose_identity(perm):
            op1 = EinsumSubOp(fd, "transpose_mm", op1, op2, perm=tuple(perm))
            yield op1
            op2 = EinsumSubOp(fd, "transpose", op2, perm=tuple(perm))
            yield op2

        # Reshape
        all_axes = list(range(ndim))
        new_axes = all_axes[-len(axes) :] if len(axes) > 0 else []
        new_common_axes = all_axes[: len(common_axes)]
        not_in_both = []
        for i in range(ndim):
            if i not in left and i not in right and i not in common_axes:
                not_in_both.append(i)

        op = EinsumSubOp(
            fd,
            "batch_dot",
            op1,
            op2,
            batch_axes=tuple(new_common_axes),
            keep_axes=None,
            sum_axes=tuple(new_axes),
            left=tuple(perm_left),
            right=tuple(perm_right),
            ndim=ndim,
        )
        yield op

        # Transpose again
        ordered_axes = (
            common_axes
            + [i for i in left if i not in right]
            + [i for i in right if i not in left]
            + not_in_both
        )
        rev_perm = [(a, i) for i, a in enumerate(ordered_axes)]
        rev_perm.sort()
        rev_perm = [p[1] for p in rev_perm]

        if not is_transpose_identity(rev_perm):
            op_unused = EinsumSubOp(fd, "transpose_mm", op1, op, perm=tuple(rev_perm))
            yield op_unused
            op = EinsumSubOp(fd, "transpose", op, perm=tuple(rev_perm))
            yield op
    else:
        raise NotImplementedError(
            "axes and right or left have axes in common, "
            "axes=%r left=%r right=%r ndim=%r." % (axes, left, right, ndim)
        )


def _decompose_einsum_equation_simple(
    equation: str,
    *shapes: List[Tuple[int, ...]],
    verbose: bool = False,
    op_matmul: str = "matmul",
) -> GraphEinsumSubOp:
    """
    Applies strategy `simple`, `numpy`
    defined in by function :func:`decompose_einsum_equation`.

    :param equation: equation
    :param shapes: input shapes
    :param verbose: verbosity
    :param op_matmul: which operator to use for matrix multiplication,
        a single operator *matmul*, or *batch_dot* with *transposes*,
        *reduce_sum*, or just *dot*
    """
    letters, mat, lengths, duplicates = analyse_einsum_equation(equation)
    assert (
        len(letters) == mat.shape[1]
    ), f"Unexpected number of letters {letters!r}, shape={mat.shape!r}."
    if not shapes:
        shapes = [(2,) * le for le in lengths[:-1]]
    _basic_verification(lengths, shapes, equation)

    # last_row, current_row (row = shape)
    rows = numpy.full((2, mat.shape[1]), -1)
    graph = GraphEinsumSubOp(letters, mat, lengths, duplicates)
    fd = mat.shape[1]
    if verbose:
        print(f"EQUATION={equation!r}")
        print(f"LETTERS={letters!r}", f"LENGTHS={lengths!r}")
        print(f"DUPLICATES={duplicates!r}")

    for i, sh in enumerate(shapes):
        if verbose:
            print()
            print("######### ROW %d shape=%r row=%r" % (i, sh, rows[1, :]))
        graph.append(i)

        # Input matrix aligned to the same dimensions.
        op = EinsumSubOp(fd, "id", i)
        op.compute_output_row(rows[1, :], mat[i, :], verbose=verbose)
        marked = graph.append(op)

        duplicate = duplicates[i]
        if duplicate is not None:
            # Diagonal
            diag = []
            for _, v in duplicate.items():
                if len(v) == 1:
                    continue
                diag.append((v[0], tuple(v)))
            op = EinsumSubOp(fd, "diagonal", op, diag=diag)
            op.compute_output_row(rows[1, :], mat[i, :], verbose=verbose)
            tr_row = rows[1, :]
            marked = graph.append(op)
        else:
            diag = None
            tr_row = mat[i]

        for iop in _apply_transpose_reshape(op, tr_row):
            op = iop
            iop.compute_output_row(rows[1, :], verbose=verbose)
            marked = graph.append(iop)

        # Reduction? (a dimension not used later)
        red = []
        for d in range(mat.shape[1]):
            if mat[i + 1 :, d].max() == -1 and rows[1, d] != -1 and rows[0, d] == -1:
                red.append(d)
        if red:
            if verbose:
                print("  -- REDUCE1 row=%d axes=%r" % (i, red))
                print(mat)
                print("  -")
                print(rows)
            op = EinsumSubOp(fd, "reduce_sum", graph.last_added_op, axes=tuple(red))
            op.compute_output_row(rows[1, :], verbose=verbose)
            marked = graph.append(op)

        if graph.last_op is not None:
            # Matrix multiplication?
            common_dims = []
            left = []
            right = []
            for d in range(mat.shape[1]):
                if rows[:, d].min() >= 0:
                    if mat[i + 1 :, d].max() >= 0:
                        left.append(d)
                        right.append(d)
                    else:
                        common_dims.append(d)
                else:
                    if rows[0, d] >= 0:
                        left.append(d)
                    if rows[1, d] >= 0:
                        right.append(d)
            if verbose:
                print(f"  -- MATMUL common_dims={common_dims!r}")
                print(rows)
            for iop in _apply_einsum_matmul(
                fd,
                graph.last_op,
                op,
                axes=tuple(common_dims),
                left=tuple(left),
                right=tuple(right),
                ndim=rows.shape[1],
                op_matmul=op_matmul,
                row1=rows[0, :],
                row2=rows[1, :],
                verbose=verbose,
            ):
                op = iop
                op.compute_output_row(rows[0, :], rows[1, :], ab=True, verbose=verbose)
                marked = graph.append(op)

        # End
        graph.mark(i, marked)
        rows[0, :] = rows[1, :]

    # Final output
    if verbose:
        print()
        print(f"######### FIN row={rows[1, :]!r}")

    if mat[len(shapes), :].max() >= 0:
        rows[1, :] = mat[len(shapes), :]
        red = []
        for d in range(mat.shape[1]):
            if rows[0, d] > 0 and rows[1, d] == -1:
                red.append(d)
            elif rows[0, d] == -1 and rows[1, d] >= 0:
                raise RuntimeError(
                    "Issue in equation %r, variable %d, last_result is %r, "
                    "output is %r." % (equation, d, rows[0, :], rows[1, :])
                )
        if red:
            if verbose:
                print(f"-- REDUCE2 axes={red!r}")
                print(mat)
            op = EinsumSubOp(fd, "reduce_sum", op, axes=tuple(red))
            graph.append(op)
            op.compute_output_row(rows[1, :], verbose=verbose)

        # Removes empty axes.
        for iop in _apply_squeeze_transpose(op, rows[1, :], mat[len(shapes), :]):
            op = iop
            iop.compute_output_row(rows[1, :], verbose=verbose)
            graph.append(iop)
    return graph
