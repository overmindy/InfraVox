"""Model real DLC FP16 operation boundaries without requantizing weights.

The original CPU simulation omits QNN-inserted FP16 conversions. Add round trips
at matching ONNX operation inputs and outputs. Reuse the external weight file.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-directory", "output-directory", "sdk-root", "numpy-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root/"lib/python")]
    import numpy as np
    import onnx
    from onnx import helper
    import onnxruntime as ort
    from qti.aisw.dlc_utils import modeltools
    source, root = args.reference_directory, args.output_directory
    root.mkdir()
    state = json.loads((source/"quantization.json").read_text())
    metadata = json.loads((source/"phone_reference.json").read_text())
    reader = modeltools.IrDlcReader()
    reader.open(state["output_dlc"])
    graph = reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    operations = {v.name:v for v in graph.get_ops()}
    model = onnx.load(str(source/"cpu_simulation.onnx"), load_external_data=False)
    nodes, boundaries = [], []
    count = 0
    def casts(value, target, label):
        nonlocal count
        prefix = f"moshi_runtime_fp16_{count}"
        count += 1
        boundaries.append({"source":value, "output":target, "operation":label})
        return [helper.make_node("Cast", [value], [prefix+"_half"], name=prefix+"_to_half", to=onnx.TensorProto.FLOAT16),
                helper.make_node("Cast", [prefix+"_half"], [target], name=prefix+"_to_float", to=onnx.TensorProto.FLOAT)]
    for node in model.graph.node:
        operation = operations.get(node.name)
        if operation and len(operation.inputs()) == len(node.input):
            for index,tensor in enumerate(operation.inputs()):
                if tensor.data_type_string() == "Float_16":
                    value = node.input[index]
                    target = f"moshi_runtime_fp16_{count}_input"
                    nodes.extend(casts(value,target,node.name+f"/input_{index}"))
                    node.input[index] = target
        after = []
        if operation and len(operation.outputs()) == len(node.output):
            for index,tensor in enumerate(operation.outputs()):
                if tensor.data_type_string() == "Float_16":
                    target = node.output[index]
                    value = f"moshi_runtime_fp16_{count}_unrounded"
                    node.output[index] = value
                    after.extend(casts(value,target,node.name+f"/output_{index}"))
        nodes.append(node)
        nodes.extend(after)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    os.link(source/"cpu_simulation.data", root/"cpu_simulation.data")
    target = root/"cpu_simulation.onnx"
    onnx.save_model(model,str(target))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4,1
    candidate = ort.InferenceSession(str(target),options,providers=["CPUExecutionProvider"])
    names = [v.name for v in candidate.get_outputs()]
    caches = [v.name for v in candidate.get_inputs() if v.name.startswith("layer_")]
    states = {}
    for record in metadata["frames"]:
        frame, sample = record["frame"],record["sample"]
        path = Path(state["calibration_manifest"]).parent/sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Captured input changed")
        with np.load(path) as data:
            feed = {v.name:data[v.name] for v in candidate.get_inputs()}
        if frame == 0:
            states = {name:feed[name][:,:,:1000,:].copy() for name in caches}
        outputs = candidate.run(None,dict(feed,**states))
        for name in caches:
            states[name][:,:,frame:frame+1,:] = outputs[names.index("output_"+name)]
        output = root/record["quantized_file"]
        np.savez(output,**dict(zip(names,outputs)))
        record["quantized_sha256"] = sha256(output)
        shutil.copy2(source/record["original_file"],root/record["original_file"])
    shutil.copy2(source/"hidden_recurrent.raw",root/"hidden_recurrent.raw")
    metadata["fp16_boundaries_modeled"] = True
    metadata["scope"] += "; actual DLC FP16 operation input/output rounding is modeled"
    write(root/"phone_reference.json",metadata)
    write(root/"fp16_boundary_model.json", {"format":"moshi-dlc-fp16-boundaries-v1", "script_sha256":sha256(__file__),
          "candidate_dlc_sha256":state["output_sha256"],"boundaries":boundaries,
          "weights_requantized":False,"hardware_outputs_changed":False,
          "scope":"Matching operation FP16 I/O rounding; internal HTP kernels remain simulated"})
    print("FP16 reference ready",len(boundaries),flush=True)


if __name__ == "__main__":
    main()
