"""Reuse a guarded QDQ model's weights for isolated cache-capacity experiments.

ScatterElements is functional ONNX, not a promise of in-place QNN memory writes.
Only constant capacity/range tensors and the four cache-update Where nodes change.
The output directory must be new; external weight files are hard-linked.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def patch(model, capacity, update, cache_output="full"):
    result = copy.deepcopy(model)
    cache_inputs = [v for v in result.graph.input if v.name.startswith("layer_")]
    if len(cache_inputs) != 4:
        raise ValueError("Expected exactly four Temporal cache inputs")
    shapes = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in cache_inputs]
    old = shapes[0][2]
    if any(s != [1, 32, old, 128] for s in shapes) or not 1 <= capacity <= old:
        raise ValueError("Unexpected cache shape or capacity")
    producers = {o: n for n in result.graph.node for o in n.output}
    changes = []
    # Fail closed for unrelated constant tensors containing the old capacity.
    for node in result.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            value = numpy_helper.to_array(attr.t)
            if value.shape == (old,) and np.array_equal(value, np.arange(old)):
                new = np.arange(capacity, dtype=value.dtype)
            elif value.size <= 4 and np.any(value == old):
                new = np.where(value == old, capacity, value).astype(value.dtype)
            elif old in value.shape:
                raise ValueError(f"Unsupported capacity-dependent constant {node.name}")
            else:
                continue
            attr.t.CopyFrom(numpy_helper.from_array(new))
            changes.append(node.name)
    for value in list(result.graph.input) + list(result.graph.output):
        if value.name.startswith(("layer_", "output_layer_")):
            value.type.tensor_type.shape.dim[2].dim_value = capacity
    # Stale inferred shapes may still contain 3000; recompute rather than edit
    # arbitrary intermediate dimensions or weight matrices.
    del result.graph.value_info[:]
    replaced = []
    if update == "scatter":
        nodes = []
        for node in result.graph.node:
            if node.op_type != "Where" or not node.output[0].startswith("output_layer_"):
                nodes.append(node)
                continue
            mask = producers[node.input[0]]
            equal = producers[mask.input[0]]
            if mask.op_type != "Reshape" or equal.op_type != "Equal":
                raise ValueError("Unexpected cache write mask")
            mods = [producers.get(v) for v in equal.input]
            mods = [n for n in mods if n is not None and n.op_type == "Mod"]
            if len(mods) != 1:
                raise ValueError("Expected position modulo capacity")
            tag = "cache_scatter_" + str(len(replaced))
            result.graph.initializer.extend([
                numpy_helper.from_array(np.array([1, 1, 1, 1], np.int64), tag + "_scalar_shape"),
                numpy_helper.from_array(np.array([1, 32, 1, 128], np.int64), tag + "_index_shape"),
            ])
            nodes.extend([
                helper.make_node("Cast", [mods[0].output[0]], [tag + "_slot"], to=onnx.TensorProto.INT64),
                helper.make_node("Reshape", [tag + "_slot", tag + "_scalar_shape"], [tag + "_index_scalar"]),
                helper.make_node("Expand", [tag + "_index_scalar", tag + "_index_shape"], [tag + "_indices"]),
                helper.make_node("ScatterElements", [node.input[2], tag + "_indices", node.input[1]],
                                 list(node.output), name=tag, axis=2),
            ])
            replaced.append(node.name)
        if len(replaced) != 4:
            raise ValueError(f"Expected four cache updates, got {replaced}")
        del result.graph.node[:]
        result.graph.node.extend(nodes)
    if cache_output == "delta":
        # Keep the complete updated cache inside attention, but return only its
        # newly written slot. Gather AFTER output QDQ preserves stored values.
        for index, output in enumerate(result.graph.output):
            if not output.name.startswith("output_layer_"):
                continue
            output_name = output.name
            internal = output_name + "_internal_full"
            for node in result.graph.node:
                for values in (node.input, node.output):
                    for i, name in enumerate(values):
                        if name == output_name:
                            values[i] = internal
            tag = f"cache_delta_{index}"
            result.graph.initializer.append(numpy_helper.from_array(np.array(capacity, np.int64), tag + "_capacity"))
            result.graph.node.extend([
                helper.make_node("Cast", ["position"], [tag + "_position"], to=onnx.TensorProto.INT64),
                helper.make_node("Mod", [tag + "_position", tag + "_capacity"], [tag + "_slot"]),
                helper.make_node("Gather", [internal, tag + "_slot"], [output_name], axis=2),
            ])
            output.type.tensor_type.shape.dim[2].dim_value = 1
    return result, {"old_capacity": old, "capacity": capacity, "update": update,
                    "cache_output": cache_output,
                    "changed_constants": changes, "replaced_where": replaced,
                    "in_place_runtime_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--update", choices=("scatter", "where"), default="scatter")
    parser.add_argument("--cache-output", choices=("full", "delta"), default="full")
    parser.add_argument("--samples", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    source = onnx.load(str(args.model), load_external_data=False)
    changed, report = patch(source, args.capacity, args.update, args.cache_output)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for tensor in source.graph.initializer:
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        if "location" not in metadata:
            continue
        rel = Path(metadata["location"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe external tensor path")
        target = args.output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(args.model.parent / rel, target)
    destination = args.output_dir / "model.onnx"
    onnx.save(changed, str(destination))
    onnx.checker.check_model(str(destination))
    report["samples"] = []
    if args.samples:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        original = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
        modified = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
        for sample in args.samples:
            with np.load(sample) as data:
                feed = {v.name: data[v.name] for v in original.get_inputs()}
            position = int(feed["position"].item())
            if position >= args.capacity:
                raise ValueError("Early-prefix parity test cannot compare different sliding windows")
            smaller = {k: v[:, :, :args.capacity, :].copy() if k.startswith("layer_") else v
                       for k, v in feed.items()}
            expected = original.run(None, feed)
            actual = modified.run(None, smaller)
            metrics = {}
            for output, got, ref in zip(original.get_outputs(), actual, expected):
                if output.name.startswith("output_layer_"):
                    ref = (ref[:, :, position:position + 1, :] if args.cache_output == "delta"
                           else ref[:, :, :args.capacity, :])
                if not np.isfinite(got).all():
                    raise ValueError(f"Nonfinite {output.name}")
                np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-5)
                metrics[output.name] = float(np.max(np.abs(got - ref)))
            report["samples"].append({"sample": str(sample), "position": position, "max_abs": metrics})
    report["passed"] = True
    (args.output_dir / "patch_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
