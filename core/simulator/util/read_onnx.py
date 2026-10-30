from beartype import beartype
import onnxruntime as ort
import torch.nn as nn
import numpy as np
import traceback
import warnings
import torch
import onnx
import io

from . import onnx2pytorch

custom_quirks = {
    'Reshape': {
        'fix_batch_size': False
    },
    'Transpose': {
        'merge_batch_size_with_channel': True,
        'remove_gdvb_transpose': True,
    },
    'Softmax' :{
        'skip_last_layer': True
    },
    'Squeeze' :{
        'skip_last_layer': True
    },
    'Conv' :{
        'merge_batch_norm': True
    },
}

@beartype
def _load_onnx(path: str | io.BytesIO):
    if isinstance(path, str):
        onnx_model = onnx.load(path)
    else:
        onnx_model = onnx.load_model_from_string(path.getvalue())
    return onnx_model

@beartype
def inference_onnx(path: str | io.BytesIO, *inputs: np.ndarray):
    sess = ort.InferenceSession(_load_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


@beartype
def add_batch(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (1, shape[0])

    if shape[0] not in [-1, 1]:
        return (1, *shape)

    return shape


@beartype
def parse_onnx(path: str | io.BytesIO, input_shape: None | list = None, output_shape: None | list = None) -> tuple:
    onnx_model = _load_onnx(path)

    onnx_inputs = [node.name for node in onnx_model.graph.input]
    initializers = [node.name for node in onnx_model.graph.initializer]
    inputs = list(set(onnx_inputs) - set(initializers))
    inputs = [node for node in onnx_model.graph.input if node.name in inputs]

    if input_shape is None:
        onnx_input_dims = inputs[0].type.tensor_type.shape.dim
        orig_input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_input_dims)
        batched_input_shape = add_batch(orig_input_shape)
    else:
        orig_input_shape = batched_input_shape = tuple(input_shape)

    if output_shape is None:
        onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
        orig_output_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_output_dims) if len(onnx_output_dims) else (1,)
        batched_output_shape = add_batch(orig_output_shape)
    else:
        batched_output_shape = tuple(output_shape)

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=custom_quirks)
    pytorch_model.eval()

    pytorch_model.to(torch.get_default_dtype())

    if custom_quirks.get('Softmax', {}).get('skip_last_layer', False):
        custom_quirks['Softmax']['skip_last_layer'] = getattr(pytorch_model, 'is_last_removed', {}).get('Softmax', False)

    if custom_quirks.get('Squeeze', {}).get('skip_last_layer', False):
        custom_quirks['Squeeze']['skip_last_layer'] = getattr(pytorch_model, 'is_last_removed', {}).get('Squeeze', False)


    correct_conversion = True
    try:
        batch = 2
        dummy = torch.randn(batch, *batched_input_shape[1:], dtype=torch.get_default_dtype())
        output_onnx = torch.cat([torch.from_numpy(inference_onnx(path, dummy[i].view(orig_input_shape).float().numpy())[0]).view(batched_output_shape) for i in range(batch)])
        output_pytorch = pytorch_model(dummy).detach().numpy()
        correct_conversion = np.allclose(output_pytorch, output_onnx, 1e-5, 1e-5)
    except:
        raise NotImplementedError('Conversion failed')

    assert correct_conversion

    return pytorch_model, batched_input_shape, batched_output_shape
