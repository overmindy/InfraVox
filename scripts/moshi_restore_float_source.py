"""Recover fixed-cache float graphs only from byte-verified original FP32 weights.

Remove paired fake-quantization nodes. Never dequantize integer checkpoint weights.
Keep public I/O and every other operation, including the existing cache updates.
"""
import argparse
import copy
import hashlib
from pathlib import Path

from moshi_quantize_temporal_offline import sha256, write


def tensor_digest(tensor, directory):
    """Hash tensor storage without loading the complete external model."""
    meta = {v.key: v.value for v in tensor.external_data}
    if "location" not in meta:
        return hashlib.sha256(tensor.raw_data).hexdigest()
    location = Path(meta["location"])
    if location.is_absolute() or ".." in location.parts:
        raise ValueError("Unsafe external tensor path")
    remaining = int(meta["length"])
    digest = hashlib.sha256()
    with (directory / location).open("rb") as stream:
        stream.seek(int(meta.get("offset", 0)))
        while remaining:
            block = stream.read(min(remaining, 8 << 20))
            if not block:
                raise ValueError("Truncated external tensor")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def restore(source, original, target):
    import onnx
    from onnx import helper
    model = onnx.load(str(source), load_external_data=False)
    reference = onnx.load(str(original), load_external_data=False)
    abi = [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)]
    tensors = {v.name: v for v in model.graph.initializer}
    original_tensors = {v.name: v for v in reference.graph.initializer}
    producers = {name: n for n in model.graph.node for name in n.output}
    aliases, removed = {}, []
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            quant = producers.get(node.input[0])
            if quant is None or quant.op_type != "QuantizeLinear" or list(node.input[1:]) != list(quant.input[1:]):
                raise ValueError("Unpaired DQ or integer checkpoint: refuse reconstruction")
            if [a.SerializeToString() for a in node.attribute] != [a.SerializeToString() for a in quant.attribute]:
                raise ValueError("Q/DQ attributes differ")
            aliases[node.output[0]] = quant.input[0]
            removed.append(node.name)
        elif node.op_type == "QuantizeLinear":
            if any(c.op_type != "DequantizeLinear" for c in model.graph.node if node.output[0] in c.input):
                raise ValueError("Quantized value has a consumer other than paired DQ")
            removed.append(node.name)

    def resolve(name):
        visited = set()
        while name in aliases:
            if name in visited:
                raise ValueError("Alias cycle")
            visited.add(name)
            name = aliases[name]
        return name

    nodes, verified, original_links = [], [], {}
    for old in model.graph.node:
        if old.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        node = copy.deepcopy(old)
        for i, name in enumerate(node.input):
            node.input[i] = resolve(name)
        if node.op_type == "MatMul" and node.input[1] in tensors:
            name = node.input[1]
            a, b = tensors[name], original_tensors[name]
            if a.data_type != onnx.TensorProto.FLOAT or b.data_type != a.data_type or list(a.dims) != list(b.dims):
                raise ValueError("Projection is not the original FP32 shape")
            previous_digest = tensor_digest(a, source.parent)
            digest = tensor_digest(b, original.parent)
            # QDQ exports can contain rounding-adjusted float weights. Recover
            # the approved original tensor, never quantize those adjusted values.
            a.CopyFrom(b)
            for entry in a.external_data:
                if entry.key == "location":
                    old_location = entry.value
                    entry.value = "original_export_"+Path(old_location).name
                    original_links[entry.value] = (original.parent/old_location).resolve()
            verified.append({"name": name, "shape": list(a.dims), "sha256": digest,
                             "previous_storage_sha256": previous_digest,
                             "recovered_from_original_export": True,
                             "previous_storage_was_identical": previous_digest == digest})
        nodes.append(node)
    if len(verified) != 8:
        raise ValueError("Expected eight verified FP32 projections")
    for value in model.graph.output:
        if value.name in aliases:
            nodes.append(helper.make_node("Identity", [resolve(value.name)], [value.name], name=value.name+"_restore_float"))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for n in nodes for name in n.input} | {v.name for v in model.graph.output}
    keep = [copy.deepcopy(v) for v in model.graph.initializer if v.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    # Remove stale quantized value metadata. ONNX shape inference reconstructs it.
    del model.graph.value_info[:]
    if [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)] != abi:
        raise ValueError("Restoration changed the public ABI")
    target.parent.mkdir(parents=True, exist_ok=False)
    external = {v.value for t in keep for v in t.external_data if v.key == "location"}
    for name in external:
        path = target.parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(original_links.get(name, (source.parent / name).resolve()))
    for record in verified:
        if tensor_digest(tensors[record["name"]],target.parent) != record["sha256"]:
            raise ValueError("Recovered tensor differs from the original export")
    onnx.save_model(model, str(target))
    onnx.checker.check_model(str(target))
    receipt = {"format": "moshi-verified-float-restoration-v1", "source": str(source),
               "source_sha256": sha256(source), "original_export": str(original),
               "original_export_sha256": sha256(original), "output": str(target), "output_sha256": sha256(target),
               "external_sha256": {n: sha256(target.parent/n) for n in external},
               "projection_storage": verified, "removed_qdq_count": len(removed),
               "public_abi_preserved": True, "other_operations_preserved": True,
               "numerical_equivalence_to_qdq_source": False, "checkpoint_dequantized": False}
    write(target.parent/"restoration.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    restore(args.source, args.original, args.output)
