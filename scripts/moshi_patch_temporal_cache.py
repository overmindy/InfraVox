"""Reuse a guarded QDQ model's weights for isolated cache-capacity experiments.

ScatterElements is functional ONNX, not a promise of in-place QNN memory writes.
Only constant capacity/range tensors and the four cache-update Where nodes change.
The output directory must be new; external weight files are hard-linked.
"""
import argparse
import copy
import json
import os
import errno
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def public_encoding(model, name):
    producers = {o: n for n in model.graph.node for o in n.output}
    seen = set()
    while name in producers and name not in seen:
        seen.add(name)
        node = producers[name]
        if node.op_type == "DequantizeLinear":
            return list(node.input[1:])
        if node.op_type not in ("Identity", "Cast"):
            break
        name = node.input[0]
    return None


def cache_input_for_output(model, name):
    """Prove numbered-cache mapping by tracing the old-cache edge of Where."""
    producers = {o: n for n in model.graph.node for o in n.output}
    caches = {v.name for v in model.graph.input if v.name.startswith("layer_")}
    seen, update_seen = set(), False
    while name not in seen:
        if name in caches and update_seen:
            return name
        seen.add(name)
        node = producers.get(name)
        if node is None:
            break
        if node.op_type == "Where" and not update_seen:
            name, update_seen = node.input[2], True
        elif node.op_type in ("Identity", "Cast", "QuantizeLinear", "DequantizeLinear"):
            name = node.input[0]
        else:
            break
    raise ValueError("Cannot prove numbered cache output mapping")


def patch(model, capacity, update, cache_output="full", scatter_encoding="source", external_base_dir=None):
    result = copy.deepcopy(model)
    cache_inputs = [v for v in result.graph.input if v.name.startswith("layer_")]
    if len(cache_inputs) != 4:
        raise ValueError("Expected exactly four Temporal cache inputs")
    shapes = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in cache_inputs]
    old = shapes[0][2]
    if any(s != [1, 32, old, 128] for s in shapes) or not 1 <= capacity <= old:
        raise ValueError("Unexpected cache shape or capacity")
    aliases = {}
    if [v.name for v in result.graph.output] == [f"output_{i}" for i in range(5)]:
        output_shapes = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in result.graph.output]
        if output_shapes != [[1, 1, 4096], *shapes]:
            raise ValueError("Numbered output shapes do not match the verified Temporal ABI")
        mapped = [cache_input_for_output(result, v.name) for v in result.graph.output[1:]]
        if set(mapped) != {v.name for v in cache_inputs} or len(set(mapped)) != 4:
            raise ValueError("Numbered outputs do not cover each cache exactly once")
        names = ["output_hidden", *("output_" + name for name in mapped)]
        for output, name in zip(result.graph.output, names):
            aliases[output.name] = name
            result.graph.node.append(helper.make_node("Identity", [output.name], [name], name="canonical_" + name))
            output.name = name
    producers = {o: n for n in result.graph.node for o in n.output}
    changes = []
    encoding_names = {v for n in result.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")
                      for v in n.input[1:]}
    # Fail closed for unrelated constant tensors containing the old capacity.
    for node in result.graph.node:
        if node.op_type != "Constant":
            continue
        if any(name in encoding_names for name in node.output):
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
    # Optimized cloud files hoist range/reshape constants into initializers.
    # Load only small integer metadata, never all external weight tensors.
    for tensor in result.graph.initializer:
        if tensor.name in encoding_names:
            continue
        elements = int(np.prod(tensor.dims or [1]))
        if elements > 4 and old not in tensor.dims:
            continue
        if tensor.data_type not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            if old in tensor.dims:
                raise ValueError(f"Unsupported non-integer capacity-dependent initializer {tensor.name}")
            continue
        if tensor.external_data:
            if external_base_dir is None:
                raise ValueError("External integer capacity metadata requires its source directory")
        value = numpy_helper.to_array(tensor, base_dir=str(external_base_dir or ""))
        if value.shape == (old,) and np.array_equal(value, np.arange(old)):
            new = np.arange(capacity, dtype=value.dtype)
        elif value.size <= 4 and np.any(value == old):
            new = np.where(value == old, capacity, value).astype(value.dtype)
        elif old in value.shape:
            raise ValueError(f"Unsupported capacity-dependent initializer {tensor.name}")
        else:
            continue
        name = tensor.name
        tensor.CopyFrom(numpy_helper.from_array(new, name))
        changes.append("initializer:" + name)
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
            scatter_data, scatter_update = node.input[2], node.input[1]
            if scatter_encoding == "aligned":
                quantizers = [n for n in result.graph.node if n.op_type == "QuantizeLinear"
                              and n.input[0] == node.output[0]]
                if not quantizers:
                    # Lite-MP FP16 100% graphs have floating cache paths. Do
                    # not invent quantizers/encodings for these valid graphs.
                    if any(producers.get(v) is not None and producers[v].op_type == "DequantizeLinear"
                           for v in (scatter_data, scatter_update)):
                        raise ValueError("Mixed quantized/float Scatter without an output encoding")
                    encoding = None
                elif len(quantizers) == 1:
                    encoding = list(quantizers[0].input[1:])
                    if len(encoding) != 2:
                        raise ValueError("Expected explicit cache output scale and zero point")
                else:
                    raise ValueError("Ambiguous cache output quantizers")
                # A scatter copies quantized payloads. Explicitly requantize
                # both cache and new values to its OUTPUT encoding beforehand.
                # This keeps all weight encodings untouched.
                for label, value in (("data", scatter_data), ("update", scatter_update)) if encoding else ():
                    name = tag + "_aligned_" + label
                    quantizer = quantizers[0]
                    q = helper.make_node("QuantizeLinear", [value, *encoding], [name + "_q"],
                                         name=name + "_quantize", **({"domain": quantizer.domain} if quantizer.domain else {}))
                    q.attribute.extend(copy.deepcopy(quantizer.attribute))
                    # Preserve contrib-domain uint16 support when present in
                    # older cloud ONNX files; don't silently change QDQ dialect.
                    downstream = [n for n in result.graph.node if n.op_type == "DequantizeLinear"
                                  and n.input[0] == quantizer.output[0]]
                    dq_domain = downstream[0].domain if downstream else quantizer.domain
                    dq = helper.make_node("DequantizeLinear", [name + "_q", *encoding], [name],
                                          name=name + "_dequantize", **({"domain": dq_domain} if dq_domain else {}))
                    if downstream:
                        dq.attribute.extend(copy.deepcopy(downstream[0].attribute))
                    nodes.extend([q, dq])
                if encoding:
                    scatter_data, scatter_update = tag + "_aligned_data", tag + "_aligned_update"
            result.graph.initializer.extend([
                numpy_helper.from_array(np.array([1, 1, 1, 1], np.int64), tag + "_scalar_shape"),
                numpy_helper.from_array(np.array([1, 32, 1, 128], np.int64), tag + "_index_shape"),
            ])
            nodes.extend([
                helper.make_node("Cast", [mods[0].output[0]], [tag + "_slot"], to=onnx.TensorProto.INT64),
                helper.make_node("Reshape", [tag + "_slot", tag + "_scalar_shape"], [tag + "_index_scalar"]),
                helper.make_node("Expand", [tag + "_index_scalar", tag + "_index_shape"], [tag + "_indices"]),
                helper.make_node("ScatterElements", [scatter_data, tag + "_indices", scatter_update],
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
                    "scatter_encoding": scatter_encoding,
                    "output_aliases": aliases,
                    "changed_constants": changes, "replaced_where": replaced,
                    "in_place_runtime_verified": False}


def semantic_bytes(model):
    """Ignore only protobuf presence of the default ONNX operator domain."""
    normalized = onnx.ModelProto()
    normalized.CopyFrom(model)
    for node in normalized.graph.node:
        if node.domain == "":
            node.ClearField("domain")
    return normalized.SerializeToString()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--update", choices=("scatter", "where"), default="scatter")
    parser.add_argument("--cache-output", choices=("full", "delta"), default="full")
    parser.add_argument("--scatter-encoding", choices=("source", "aligned"), default="aligned",
                        help="source is diagnostic; aligned requantizes Scatter I/O to a common encoding")
    parser.add_argument("--samples", nargs="*", type=Path, default=[])
    parser.add_argument("--validate-existing", action="store_true",
                        help="Validate an existing identical candidate without rewriting its model")
    parser.add_argument("--copy-external-data", action="store_true",
                        help="Allow one-time copy when source weights and output reside on different filesystems")
    parser.add_argument("--max-relative-rms", type=float,
                        help="Optional explicit RMS checkpoint; reports strict elementwise result separately")
    args = parser.parse_args()
    source = onnx.load(str(args.model), load_external_data=False)
    if args.max_relative_rms is not None and not 0 < args.max_relative_rms <= 0.01:
        raise ValueError("Diagnostic RMS checkpoint must be in (0, 0.01]")
    changed, report = patch(source, args.capacity, args.update, args.cache_output, args.scatter_encoding, args.model.parent)
    if args.validate_existing:
        existing = onnx.load(str(args.output_dir / "model.onnx"), load_external_data=False)
        if semantic_bytes(existing) != semantic_bytes(changed):
            raise ValueError("Existing candidate differs from requested patch")
    else:
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
            try:
                os.link(args.model.parent / rel, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV or not args.copy_external_data:
                    raise
                shutil.copy2(args.model.parent / rel, target)
    destination = args.output_dir / "model.onnx"
    if not args.validate_existing:
        onnx.save(changed, str(destination))
    onnx.checker.check_model(str(destination))
    report["samples"] = []
    report["max_relative_rms_checkpoint"] = args.max_relative_rms
    passed = True
    if args.samples:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        original = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
        modified = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
        for sample in args.samples:
            with np.load(sample) as data:
                dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                          "tensor(int64)": np.int64, "tensor(int32)": np.int32}
                feed = {v.name: data[v.name].astype(dtypes[v.type], copy=False) for v in original.get_inputs()}
            position = int(feed["position"].item())
            if position >= args.capacity:
                raise ValueError("Early-prefix parity test cannot compare different sliding windows")
            smaller = {k: v[:, :, :args.capacity, :].copy() if k.startswith("layer_") else v
                       for k, v in feed.items()}
            expected = original.run(None, feed)
            actual = modified.run(None, smaller)
            metrics = {}
            tolerances = {}
            details = {}
            for output, got, ref in zip(original.get_outputs(), actual, expected):
                canonical = report["output_aliases"].get(output.name, output.name)
                if canonical.startswith("output_layer_"):
                    ref = (ref[:, :, position:position + 1, :] if args.cache_output == "delta"
                           else ref[:, :, :args.capacity, :])
                if not np.isfinite(got).all():
                    raise ValueError(f"Nonfinite {output.name}")
                atol = 2e-5
                if args.scatter_encoding == "aligned" and canonical.startswith("output_layer_"):
                    encoding = public_encoding(source, output.name)
                    scales = [v for v in source.graph.initializer if encoding and v.name == encoding[0]]
                    if encoding and len(scales) != 1:
                        raise ValueError("Cannot establish output cache quantization LSB")
                    # Explicit Q/DQ introduces an extra float rounding boundary:
                    # allow at most one existing output quantization step, never
                    # a broad percentage tolerance that masks scale/offset bugs.
                    if scales:
                        atol = max(atol, 1.1 * float(numpy_helper.to_array(scales[0]).max()))
                error = got.astype(np.float64) - ref.astype(np.float64)
                rmse = float(np.sqrt(np.mean(error * error)))
                rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
                relative = rmse / max(rms, 1e-12)
                strict = bool(np.allclose(got, ref, rtol=2e-4, atol=atol))
                accepted = strict if args.max_relative_rms is None else relative <= args.max_relative_rms
                passed = passed and accepted
                metrics[canonical] = float(np.max(np.abs(got - ref)))
                tolerances[canonical] = atol
                details[canonical] = {"rmse": rmse, "relative_rms": relative,
                                      "strict_elementwise_passed": strict, "checkpoint_passed": accepted}
            report["samples"].append({"sample": str(sample), "position": position,
                                      "max_abs": metrics, "allowed_absolute_error": tolerances, "metrics": details})
    report["passed"] = passed
    (args.output_dir / "patch_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
