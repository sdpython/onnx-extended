============================
External Data and Big Models
============================

:epkg:`protobuf` does not support files bigfer than 2 Gb
and that limit is usually exceeded for language models
such as :epkg:`Llama`. :epkg:`onnx` overcomes that limit
by saving the weights outside the model. The main file only
keeps the filename the weights the model are stored in.


Save a big model
================

Let's assume the model is in memory. It needs to be saved
with the weights outside the onnx file. Here is a short example
on how to do it. It relies on function
:func:`save_model <onnx_extended.tools.save_model>`.

.. runpython::
    :showcode:

    import os
    import pprint
    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onp
    from onnx_extended.tools import save_model


    def _get_model():
        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
        Z = oh.make_tensor_value_info("Z", onnx.TensorProto.INT64, [None, None])
        graph = oh.make_graph(
            [
                oh.make_node("Mul", ["X", "X"], ["X2"]),
                oh.make_node("Add", ["X2", "Y"], ["z1"]),
                oh.make_node("Mul", ["z1", "W"], ["z2"]),
                oh.make_node("Cast", ["z2"], ["Z"], to=onnx.TensorProto.INT64),
            ],
            "add",
            [X],
            [Z],
            [
                onp.from_array(np.arange(16).reshape((-1, 4)).astype(np.float32), name="Y"),
                onp.from_array(
                    (np.arange(16).reshape((-1, 4)) + 100).astype(np.float32), name="W"
                ),
            ],
        )
        onnx_model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=8
        )
        return onnx_model


    model = _get_model()
    
    # size_threshold: every constant tensor whose size is above this
    # threshold is taken out externally.
    save_model(model, "an_onnx_model.onnx", external=True, size_threshold=15)
    
    pprint.pprint([n for n in os.listdir() if "an_onnx" in n])
    print("--------------------------------------")
    print(model)

The data is stored externally just close to the models
and it has to be that way. At the ned of the example, the model
is printed on a the standard output. We can see it was modified
and now it does not contain the weights anymore but only their location.
It is possible to restore the weights and put them back in the onnx structure.
Function :func:`load_external <onnx_extended.tools.load_external>` does it.
It needs an extra parameter to indicate the location of the weights.

.. code-block:: python

    from onnx_extended.tools import load_external

    load_external(model, ".")

Load a big model
================

When loading the model back, two options are possible.
The first is load everything including the external data.
:func:`load_external <onnx_extended.tools.load_external>`
can either load the weights (`external=True`) or loads the
structure of the model and leaves the weights on the disk
(`external=False`).

.. runpython::
    :showcode:

    from onnx_extended.tools import load_model

    model = load_model("an_onnx_model.onnx", external=False)
    print(model)

Example with Llama
==================

The :epkg:`Llama` model is big. An onnx version can be retrieved from
this github repository `microsoft/Llama-2-Onnx
<https://github.com/microsoft/Llama-2-Onnx>`_.
As it takes time to play with the whole, it can be interested
to extract the first layers.

.. code-block:: python

    import os
    import onnx
    from onnx_extended.tools import load_model, save_model, load_external
    from onnx_extended.tools.onnx_nodes import select_model_inputs_outputs

    llama = (
        "Llama-2-Onnx/7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx"
    )

    # load model without loading the weights
    onx = load_model(llama, external=False)

    # extract a piece of it from the inputs to a some intermediate output
    outputs = ["/transformer/block_list.1/attention/Gather_output_0"]
    new_onx = select_model_inputs_outputs(onx, outputs)

    # load external data on the subpart: the weights are still on disk
    load_external(new_onx, os.path.dirname(llama))

    # save model without any external data
    name = "models/llama_16_block_list_1.onnx"
    save_model(new_onx, name, external=False)

The name of all intermediate results can be obtained with the
following command line. It runs shape inference and stores the
results in a dataframe.

::

    python -m onnx_extended display \
        --external=0 -s types_shapes.xlsx \
        -m ./Llama-2-Onnx/7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx
