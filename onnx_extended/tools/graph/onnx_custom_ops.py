import numpy as np
from onnx import TensorProto
from onnx.defs import OpSchema
from onnx.helper import make_attribute
from onnx.reference.custom_element_types import (
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)
from onnx.reference.op_run import OpRun

try:
    from onnx.reference.ops.op_cast import Cast_19 as Cast
    from onnx.reference.ops.op_quantize_linear import (
        QuantizeLinear_19 as QuantizeLinear,
    )
except ImportError:
    from onnx.reference.ops.op_cast import Cast
    from onnx.reference.ops.op_quantize_linear import QuantizeLinear
from onnx.reference.ops.op_dequantize_linear import DequantizeLinear


def _make_schema_gemm_float8(cls_name: str) -> OpSchema:
    """
    Returns the schema for operator `cls_name="GemmFloat8"`.
    """
    gemm_float8_types = [
        "tensor(float8e4m3fn)",
        "tensor(float8e5m2)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(float)",
    ]
    return OpSchema(
        cls_name,
        "com.microsoft",
        1,
        attributes=[
            OpSchema.Attribute("transA", make_attribute("transA", 0), "transA"),
            OpSchema.Attribute("transB", make_attribute("transB", 0), "transB"),
            OpSchema.Attribute("alpha", make_attribute("alpah", 1.0), "alpha"),
            OpSchema.Attribute("beta", make_attribute("beta", 0.0), "beta"),
            OpSchema.Attribute("smCount", make_attribute("smCount", 0), "smCount"),
            OpSchema.Attribute(
                "fastAccumulationMode",
                make_attribute("fastAccumulationModel", 1),
                "fastAccumulationMode",
            ),
            OpSchema.Attribute(
                "computeType",
                make_attribute("computeType", "CUBLAS_COMPUTE_32F_FAST_TF32"),
                "a value in CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_32F, "
                "CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16BF, "
                "CUBLAS_COMPUTE_32F_FAST_TF32",
            ),
            OpSchema.Attribute("rowMajor", make_attribute("rowMajor", 0), "rowMajor"),
            OpSchema.Attribute(
                "dtype", make_attribute("dtype", TensorProto.FLOAT), "dtype"
            ),
            OpSchema.Attribute(
                "activation", OpSchema.AttrType.STRING, "activation", required=None
            ),
        ],
        inputs=[
            OpSchema.FormalParameter("A", "TA"),
            OpSchema.FormalParameter("B", "TB"),
            OpSchema.FormalParameter("C", "TC"),
            OpSchema.FormalParameter("scaleA", "TS"),
            OpSchema.FormalParameter("scaleB", "TS"),
            OpSchema.FormalParameter("scaleY", "TS"),
        ],
        outputs=[OpSchema.FormalParameter("Y", "TR")],
        type_constraints=[
            ("TA", gemm_float8_types, "Constrain input type to input A."),
            ("TB", gemm_float8_types, "Constrain input type to input B."),
            (
                "TC",
                ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
                "Constrain input type to input C.",
            ),
            ("TS", ["tensor(float)"], "Constrain input type to input scales."),
            ("TR", gemm_float8_types, "Constrain output type."),
        ],
    )


def is_float8_dtype(dtype) -> bool:
    """
    Returns true if dtype is a float 8 type.
    """
    f8 = {
        (float8e4m3fn, "e4m3fn"),
        (float8e4m3fnuz, "e4m3fnuz"),
        (float8e5m2, "e5m2"),
        (float8e5m2fnuz, "e5m2fnuz"),
    }
    for dt, name in f8:
        if dtype == dt and dtype.descr[0][0] == name:
            return True
    return False


class GemmFloat8(OpRun):
    op_domain = "com.microsoft"
    op_schema = _make_schema_gemm_float8("GemmFloat8")

    def _run(
        self,
        A,
        B,
        *args,
        transA: int = None,
        transB: int = None,
        alpha: float = None,
        beta: float = None,
        smCount: int = None,
        fastAccumulationMode: int = None,
        computeType: str = None,
        dtype: int = None,
        rowMajor: int = None,
        activation: str = None,
    ):
        if len(A.shape) != 2:
            raise ValueError(f"A is not a matrix, its shape is {A.shape}.")
        if len(B.shape) != 2:
            raise ValueError(f"B is not a matrix, its shape is {B.shape}.")
        C = args[0] if len(args) > 0 else None
        scaleA = args[1] if len(args) > 1 else None
        scaleB = args[2] if len(args) > 2 else None
        scaleR = args[3] if len(args) > 3 else None

        if computeType == "CUBLAS_COMPUTE_16F":
            compute_dtype = np.float16
            compute_xtype = TensorProto.FLOAT16
        elif computeType in {
            "CUBLAS_COMPUTE_32F",
            "CUBLAS_COMPUTE_32F_FAST_16F",
            "CUBLAS_COMPUTE_32F_FAST_TF32",
        }:
            compute_dtype = np.float32
            compute_xtype = TensorProto.FLOAT
        else:
            raise ValueError(f"Unexpected value {computeType!r} for computeType.")

        # if rowMajor == 0:
        #     The parameter does not any impact on the result, only on the computation.

        alpha = compute_dtype(alpha)
        beta = compute_dtype(beta)

        fp8a = is_float8_dtype(A.dtype)
        fp8b = is_float8_dtype(B.dtype)
        if fp8a or fp8b:
            if not fp8a or not fp8b:
                raise RuntimeError(
                    f"Both types must be float 8, not one of them only, "
                    f"{A.dtype} @ {B.dtype}."
                )
            if rowMajor == 1:
                raise RuntimeError("Only rowMajor == 0 is supported on GPU.")
            if self.__class__.__name__ == "GemmFloat8" and (transA or not transB):
                raise RuntimeError(
                    f"If both types are float 8, then transA=0 and "
                    f"transB=1 but transA={transA} and transB={transB}."
                )

        ca = Cast.eval(A, to=compute_xtype)
        cb = Cast.eval(B, to=compute_xtype)
        cc = None if C is None else Cast.eval(C, to=compute_xtype)

        if transA:
            ca = ca.T
        if transB:
            cb = cb.T

        if scaleA is not None:
            if scaleA.dtype != np.float32:
                raise ValueError(f"scaleA must be a float not {scaleA.dtype}.")
            ca = DequantizeLinear.eval(ca, scaleA.astype(compute_dtype))
        if scaleB is not None:
            if scaleB.dtype != np.float32:
                raise ValueError(f"scaleB must be a float not {scaleB.dtype}.")
            cb = DequantizeLinear.eval(cb, scaleB.astype(compute_dtype))

        try:
            res = ca @ cb * alpha
        except ValueError as e:
            raise ValueError(
                f"Unable to multiply shapes {ca.shape!r} and {cb.shape!r}."
            ) from e
        if C is not None:
            res += cc * beta

        if scaleR is not None:
            res = QuantizeLinear(res, scaleR.astype(compute_dtype))

        if activation is not None:
            raise NotImplementedError(
                f"activation={activation!r} is not implemented yet."
            )

        final = Cast.eval(res, to=dtype)
        return (final,)


class GemmFloat8Quiet(GemmFloat8):
    op_domain = "com.microsoft"
    op_schema = _make_schema_gemm_float8("GemmFloat8Quiet")
