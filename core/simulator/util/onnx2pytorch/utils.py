import numpy as np
import torch
import onnx
import io

try:
    import onnxruntime as ort
except ImportError:
    ort = None

PRINT_DEBUG = False

def value_wrapper(value):
    def callback(*args, **kwargs):
        return value

    return callback


def is_constant(value):
    return value.ndim == 0 or value.shape == torch.Size([1])


def is_symmetric(params):
    assert len(params) // 2 == len(params) / 2, "Non even number of parameters."
    idx = len(params) // 2
    for i in range(0, idx):
        if params[i] != params[idx + i]:
            return False
    return True


def extract_padding_params(params):
    pad_dim = len(params) // 2
    if pad_dim == 0:
        return []
    pads = np.array(params).reshape(-1, pad_dim).T.flatten()

    if (pads[:4] == 0).all():
        pads = pads[4:]
    pads = pads.tolist()
    pads.reverse()
    return pads


def extract_padding_params_for_conv_layer(params):
    if is_symmetric(params):
        return params[: len(params) // 2]
    else:
        pad_dim = len(params) // 2
        pad_layer = getattr(torch.nn, "ConstantPad{}d".format(pad_dim))
        pads = extract_padding_params(params)[::-1]
        return pad_layer(pads, value=0)


def get_selection(indices, dim):
    assert dim >= 0, "Negative dimension not supported."
    if isinstance(indices, list):
        indices = torch.tensor(indices)
    assert isinstance(indices, (torch.Tensor, np.ndarray))
    selection = [slice(None) for _ in range(dim + 1)]
    selection[dim] = indices
    return selection


def assign_values_to_dim(tensor, values, indices, dim, inplace=True):
    if dim < 0:
        dim = dim + len(tensor.shape)
    selection = get_selection(indices, dim)
    if not inplace:
        tensor = tensor.clone()
    tensor[selection] = values
    return tensor


def get_type(x):
    if x.type.startswith("tensor"):
        typ = x.type[7:-1]
    else:
        raise NotImplementedError("For type: {}".format(x.type))

    if typ == "float":
        typ = "float32"
    elif typ == "double":
        typ = "float64"
    return typ


def get_shape(x, unknown_dim_size=1):
    shape = x.shape
    shape = [i if isinstance(i, int) else unknown_dim_size for i in shape]
    return shape


def get_activation_value(onnx_model, inputs, activation_names):
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"
    assert all(isinstance(x, np.ndarray) for x in inputs)

    if not isinstance(activation_names, (list, tuple)):
        activation_names = [activation_names]

    while len(onnx_model.graph.output):
        onnx_model.graph.output.pop()

    for activation_name in activation_names:
        activation_value = onnx.helper.ValueInfoProto()
        activation_value.name = activation_name
        onnx_model.graph.output.append(activation_value)

    buffer = io.BytesIO()
    onnx.save(onnx_model, buffer)
    buffer.seek(0)
    onnx_model_new = onnx.load(buffer)
    sess = ort.InferenceSession(onnx_model_new.SerializeToString())

    input_names = [x.name for x in sess.get_inputs()]
    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = dict(zip(input_names, inputs))

    return sess.run(None, inputs)


def get_inputs_names(onnx_graph):
    param_names = set([x.name for x in onnx_graph.initializer])
    input_names = [x.name for x in onnx_graph.input]
    input_names = [x for x in input_names if x not in param_names]
    return input_names


def get_inputs_sample(onnx_model, to_torch=False):
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = sess.get_inputs()
    input_names = get_inputs_names(onnx_model.graph)
    input_tensors = [
        np.abs(np.random.rand(*get_shape(x)).astype(get_type(x))) for x in inputs
    ]
    if to_torch:
        input_tensors = [torch.from_numpy(x) for x in input_tensors]
    return dict(zip(input_names, input_tensors))


def get_outputs_names(onnx_graph):
    output_names = [x.name for x in onnx_graph.output]
    return output_names


def get_ops_names(onnx_graph):
    ops_used = set(node.op_type for node in onnx_graph.node)
    for node in onnx_graph.node:
        if node.op_type == "Loop":
            for attr in node.attribute:
                if attr.name == "body":
                    ops_used |= get_ops_names(attr.g)
        elif node.op_type == "If":
            for attr in node.attribute:
                if attr.name == "then_branch":
                    ops_used |= get_ops_names(attr.g)
                elif attr.name == "else_branch":
                    ops_used |= get_ops_names(attr.g)
    return ops_used
