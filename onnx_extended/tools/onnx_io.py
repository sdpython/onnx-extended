import base64
import os
import textwrap
from typing import Optional, Set, Union
import onnx


def load_model(
    model: Union[str, onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto],
    external: bool = True,
    base_dir: Optional[str] = None,
) -> Union[onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto]:
    """
    Loads a model or returns the only argument if the type
    is already a ModelProto.

    :param model: proto file
    :param external: loads the external data as well
    :param base_dir: needed if external is True and
        the model has external weights
    :return: ModelProto
    """
    if isinstance(model, onnx.ModelProto):
        if base_dir is not None and external:
            if not os.path.exists(base_dir):
                raise FileNotFoundError(f"Unable to find folder {base_dir!r}.")
            onnx.load_external_data_for_model(model, base_dir)
        return model
    if isinstance(model, (onnx.GraphProto, onnx.FunctionProto)):
        return model
    if not os.path.exists(model):
        raise FileNotFoundError(f"Unable to find model {model!r}.")
    with open(model, "rb") as f:
        return onnx.load(f, load_external_data=external)


def load_external(
    model: onnx.ModelProto, base_dir: str, names: Optional[Set[str]] = None
):
    """
    Loads external data into memory.

    :param model: the model loaded with :func:`load_model`
    :param base_dir: directory when the data can be found
    :param names: subsets of names to load or None for all
    """
    from onnx.external_data_helper import (
        _get_all_tensors,
        uses_external_data,
        load_external_data_for_tensor,
    )

    for tensor in _get_all_tensors(model):
        if names is not None and tensor.name not in names:
            continue
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)
            # After loading raw_data from external_data, change the state of tensors
            tensor.data_location = onnx.TensorProto.DEFAULT
            # and remove external data
            del tensor.external_data[:]


def save_model(
    proto: onnx.ModelProto,
    filename: str,
    external: bool = False,
    convert_attribute: bool = True,
    size_threshold: int = 1024,
    all_tensors_to_one_file: bool = True,
):
    """
    Saves a model into an onnx file.

    :param proto: ModelProto
    :param filename: where to save it
    :param external: saves weights as external data
    :param convert_attribute: converts attributes as well
    :param size_threshold: every weight above that threshold is saved as external
    :param all_tensors_to_one_file: saves all tensors in one unique file
    """
    if not external:
        onnx.save_model(proto, filename)
        return

    dirname, shortname = os.path.split(filename)
    onnx.convert_model_to_external_data(
        proto,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=shortname + ".data",
        convert_attribute=convert_attribute,
        size_threshold=size_threshold,
    )

    proto = onnx.write_external_data_tensors(proto, dirname)
    with open(filename, "wb") as f:
        f.write(proto.SerializeToString())


def onnx2string(proto: onnx.ModelProto, as_code: bool = False) -> str:
    """
    Takes a model and returns a string pluggagle in
    a python script. It uses module `base64`.

    :param proto: model proto
    :param as_code: if true, returns the model as a piece of code
    :return: string
    """
    text = proto.SerializeToString()
    content = base64.b64encode(text).decode("ascii")
    if not as_code:
        return content
    lines = []
    while content:
        if len(content) > 64:
            lines.append(content[:64])
            content = content[64:]
        else:
            lines.append(content)
            break
    slines = "\n".join([f'    "{li}"' for li in lines])
    template = textwrap.dedent(
        """
    from onnx_extended.tools.onnx_io import string2onnx

    text = (
    {model}
    )
    model = string2onnx(text)
    """
    )
    return template.format(model=slines)


def string2onnx(text: str) -> onnx.ModelProto:
    """
    Restores a model saved with
    :func:`onnx2string <onnx_extended.tools.onnx_io.onnx2string>`.

    :param text: string produced by
        :func:`onnx2string <onnx_extended.tools.onnx_io.onnx2string>`
    :return: ModelProto
    """
    b = base64.b64decode(text.encode("ascii"))
    model = onnx.ModelProto()
    model.ParseFromString(b)
    return model
