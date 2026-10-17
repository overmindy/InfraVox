"""Rewrite fixed-shape static MatMuls as equivalent pointwise convolutions."""
import copy


def rewrite_pointwise_projections(model, policy, expected_bits, add_zero_bias=False):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    original_abi = [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)]
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*inferred.graph.input,*inferred.graph.value_info,*inferred.graph.output)}
    tensors = {v.name:v for v in model.graph.initializer}
    activations = {v["name"]:v for v in policy["activation_encodings"]}
    nodes, records = [], []
    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in expected_bits:
            nodes.append(node)
            continue
        name = node.input[1]
        weight = numpy_helper.to_array(tensors[name])
        ni, no = weight.shape
        before, after = shapes[node.input[0]], shapes[node.output[0]]
        if (before[-1] != ni or after[-1] != no or before[:-1] != after[:-1]
                or any(v < 1 for v in before)):
            raise ValueError("Pointwise conversion requires fixed, matching leading dimensions")
        batch = int(np.prod(before[:-1]))
        x, y = name+"_pointwise_input", name+"_pointwise_output"
        xs, ys = name+"_pointwise_input_shape", name+"_pointwise_output_shape"
        model.graph.initializer.extend([
            numpy_helper.from_array(np.array([batch,ni,1,1],np.int64),xs),
            numpy_helper.from_array(np.array(after,np.int64),ys)])
        tensors[name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(weight.T[:,:,None,None]),name))
        conv_inputs = [x, name]
        if add_zero_bias:
            bias = name+"_pointwise_zero_bias"
            model.graph.initializer.append(numpy_helper.from_array(np.zeros(no,np.float32),bias))
            conv_inputs.append(bias)
        nodes.extend([helper.make_node("Reshape",[node.input[0],xs],[x],name=node.name+"/to_pointwise"),
                      helper.make_node("Conv",conv_inputs,[y],name=node.name,kernel_shape=[1,1]),
                      helper.make_node("Reshape",[y,ys],[node.output[0]],name=node.name+"/from_pointwise")])
        for old,new in ((node.input[0],x),(node.output[0],y)):
            entry = copy.deepcopy(activations[old]);entry["name"]=new
            policy["activation_encodings"].append(entry)
        records.append({"node":node.name,"weight":name,"shape":[ni,no],"bitwidth":expected_bits[name]})
    if len(records) != len(expected_bits):
        raise ValueError("Projection rewrite count differs from the precision policy")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if [v.SerializeToString() for v in (*model.graph.input,*model.graph.output)] != original_abi:
        raise ValueError("Projection rewriting changed the public ABI")
    return records
