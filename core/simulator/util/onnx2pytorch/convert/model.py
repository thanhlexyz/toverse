from collections import defaultdict
from functools import partial
from copy import deepcopy
from torch import nn
import torch
import onnx


from ..constants import (
    COMPOSITE_LAYERS,
    MULTIOUTPUT_LAYERS,
    STANDARD_LAYERS,
)
from .debug import debug_model_conversion
from .operations import (
    convert_operations,
    get_buffer_name,
    get_init_parameter,
    Loop,
)
from ..utils import (
    get_inputs_names,
    get_outputs_names,
)


def compute_activation_dependencies(onnx_graph, model, mapping):
    needed_by = defaultdict(set)
    for node in onnx_graph.node:
        out_op_id = node.output[0]
        for in_op_id in node.input:
            needed_by[in_op_id].add(out_op_id)
        if node.op_type == "Loop":
            l1 = getattr(model, mapping[out_op_id])
            loop_body_l1 = l1.body
            for node_l1 in loop_body_l1.node:
                for in_op_id in node_l1.input:
                    needed_by[in_op_id].add(out_op_id)
                if node_l1.op_type == "Loop":
                    l2 = getattr(model, l1.mapping[node_l1.output[0]])
                    loop_body_l2 = l2.body
                    for node_l2 in loop_body_l2.node:
                        for in_op_id in node_l2.input:
                            needed_by[in_op_id].add(out_op_id)
                        if node_l2.op_type == "Loop":
                            raise NotImplementedError(
                                "Activation garbage collection not implemented for >2 nested loops."
                            )
    needed_by.default_factory = None
    return needed_by


class ConvertModel(nn.Module):
    def __init__(
        self,
        onnx_model: onnx.ModelProto,
        batch_dim=0,
        experimental=False,
        debug=False,
        enable_pruning=False,
        quirks=None,
        enable_recording=False,
    ):
        super().__init__()
        self.onnx_model = onnx_model
        self.batch_dim = batch_dim
        self.experimental = experimental
        self.debug = debug
        self.enable_pruning = enable_pruning
        self.is_last_removed = {}
        self.enable_recording = enable_recording

        self.input_names = get_inputs_names(onnx_model.graph)
        self.output_names = get_outputs_names(onnx_model.graph)
        quirks = quirks if isinstance(quirks, dict) else {}
        opset_version = onnx_model.opset_import[0].version

        self.mapping = {}
        for op_id, op_name, op, op_type, is_last_removed in convert_operations(
            onnx_model.graph,
            opset_version,
            batch_dim,
            enable_pruning,
            quirks,
        ):
            setattr(self, op_name, op)
            if isinstance(op, Loop) and debug:
                raise NotImplementedError("debug-mode with Loop node not implemented.")
            self.mapping[op_id] = op_name
            if is_last_removed:
                self.is_last_removed[op_type] = is_last_removed

        for tensor in self.onnx_model.graph.initializer:
            buffer_name = get_buffer_name(tensor.name)
            self.register_buffer(
                buffer_name,
                torch.from_numpy(onnx.numpy_helper.to_array(tensor).copy()),
            )

        self.needed_by = compute_activation_dependencies(
            self.onnx_model.graph, self, self.mapping
        )


    def forward(self, *input_list, **input_dict):
        if len(input_list) > 0 and len(input_dict) > 0:
            raise ValueError(
                "forward-pass accepts either input_list (positional args) or "
                "input_dict (keyword args) but not both"
            )
        if len(input_list) > 0:
            inputs = input_list
        if len(input_dict) > 0:
            inputs = [input_dict[key] for key in self.input_names]

        if self.enable_recording:
            intermediate_outputs = []

        if not self.experimental and inputs[0].shape[self.batch_dim] > 1:
            raise NotImplementedError(
                "Input with larger batch size than 1 not supported yet."
            )
        activations = dict(zip(self.input_names, inputs))
        still_needed_by = deepcopy(self.needed_by)

        for node in self.onnx_model.graph.node:
            out_op_id = node.output[0]
            out_op_name = self.mapping[out_op_id]
            in_op_names = [
                self.mapping.get(in_op_id, in_op_id)
                for in_op_id in node.input
                if in_op_id in activations
            ]

            op = getattr(self, out_op_name)
            if op is None:
                continue

            if isinstance(op, STANDARD_LAYERS) or (
                isinstance(op, COMPOSITE_LAYERS)
                and any(isinstance(x, STANDARD_LAYERS) for x in op.modules())
            ):
                in_activations = [
                    activations[in_op_id]
                    for in_op_id in node.input
                    if in_op_id in activations
                ]
            else:
                in_activations = [
                    activations[in_op_id] if in_op_id in activations
                    else get_init_parameter([self], in_op_id, inputs[0])
                    for in_op_id in node.input
                ]

            in_activations = [in_act for in_act in in_activations if in_act is not None]

            if isinstance(op, Loop):
                outputs = op((self,), activations, *in_activations)
                for out_op_id, output in zip(node.output, outputs):
                    activations[out_op_id] = output
            elif isinstance(op, partial) and op.func == torch.cat:
                activations[out_op_id] = op(in_activations)
            elif isinstance(op, nn.Identity):
                activations[out_op_id] = op(in_activations[0])
            elif isinstance(op, MULTIOUTPUT_LAYERS) or (
                isinstance(op, COMPOSITE_LAYERS)
                and any(isinstance(x, MULTIOUTPUT_LAYERS) for x in op.modules())
            ):
                for out_op_id, output in zip(node.output, op(*in_activations)):
                    activations[out_op_id] = output
            else:
                activations[out_op_id] = op(*in_activations)
                if self.enable_recording:
                    intermediate_outputs.extend([_.clone() for _ in in_activations])

            for in_op_id in node.input:
                if in_op_id in still_needed_by:
                    still_needed_by[in_op_id].discard(out_op_id)
                    if len(still_needed_by[in_op_id]) == 0:
                        if in_op_id in activations:
                            del activations[in_op_id]

            if self.debug:
                debug_model_conversion(
                    self.onnx_model,
                    [activations[x] for x in self.input_names],
                    [activations[out_op_id] for out_op_id in node.output],
                    node,
                )

        outputs = [activations[x] for x in self.output_names]
        if len(outputs) == 1:
            outputs = outputs[0]

        if self.enable_recording:
            return outputs, intermediate_outputs

        return outputs
