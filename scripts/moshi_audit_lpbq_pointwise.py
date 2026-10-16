"""Audit LPBQ scale math, lossless layout changes, and retained state precision.

The official AIMET scale helpers are evaluated in isolation. This is not a
claim that the complete AIMET package or its calibration workflow was used.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Sequence, Tuple

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "official-source", "awq-source", "sdk-root", "numpy-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root / "lib/python")]
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from qti.aisw.dlc_utils import modeltools
    from moshi_check_offline_dlc import audit

    root = args.directory
    result = json.loads((root / "RESULT.json").read_text())
    parent = Path(result["parent_directory"])
    state = json.loads((root / "candidate/quantization.json").read_text())
    parent_state = json.loads((parent / "candidate/quantization.json").read_text())
    policy = json.loads((root / "quantization_overrides.json").read_text())
    entries = {v["name"]: v for v in policy["param_encodings"]}
    required = {"_split_blocks", "_get_per_group_scale_factor", "grouped_dynamic_quantize"}
    selected = [v for v in ast.parse(args.official_source.read_text()).body
                if isinstance(v, ast.FunctionDef) and v.name in required]
    if {v.name for v in selected} != required:
        raise ValueError("Official AIMET scale helper structure changed")
    namespace = {"np": np, "Sequence": Sequence, "Tuple": Tuple}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.official_source), "exec"), namespace)
    official_quantize = namespace["grouped_dynamic_quantize"]
    source = onnx.load(str(args.awq_source), load_external_data=False)
    initializers = {v.name: v for v in source.graph.initializer}
    scale_checks = []
    for name, bits in state["expected_projection_bits"].items():
        if bits != 4:
            continue
        e = entries[name]
        weight = numpy_helper.to_array(initializers[name], base_dir=str(args.awq_source.parent))
        ni, no = weight.shape
        blocks = weight.T.reshape(no, ni // e["block_size"], e["block_size"])
        raw_steps = np.maximum(-blocks.min(axis=2)/8, blocks.max(axis=2)/7)
        int_steps, channel_steps = official_quantize(raw_steps, [1, -1], 4)
        passed = (np.array_equal(int_steps.reshape(-1), np.array(e["per_block_int_scale"]))
                  and np.array_equal(channel_steps.reshape(-1), np.array(e["scale"], np.float32)))
        scale_checks.append({"weight": name, "block_count": int(int_steps.size),
                             "integer_scales_identical": bool(passed), "passed": bool(passed)})
        if not passed:
            raise ValueError("LPBQ scales differ from the official AIMET helper")
        del weight, blocks, raw_steps, int_steps, channel_steps

    def load_graph(path):
        reader = modeltools.IrDlcReader()
        reader.open(str(path))
        return reader, reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))

    old_reader, old = load_graph(parent_state["output_dlc"])
    new_reader, new = load_graph(state["output_dlc"])
    old_ops = {v.inputs()[1].name(): v for v in old.get_ops() if v.type == "FullyConnected"}
    new_ops = {v.inputs()[1].name(): v for v in new.get_ops() if v.type == "Conv2d"}
    if len(old_ops) != 8 or old_ops.keys() != new_ops.keys():
        raise ValueError("Projection identities changed")
    layout_checks = []
    for name in old_ops:
        a, b = old_ops[name].inputs()[1], new_ops[name].inputs()[1]
        blockwise = "BLOCKWISE_EXPANSION" in str(a.get_encoding().type)
        if blockwise != ("BLOCKWISE_EXPANSION" in str(b.get_encoding().type)):
            raise ValueError("LPBQ was replaced with an expanded weight representation")
        ae = a.get_encoding().lpbqEncInfo if blockwise else a.get_encoding().axisEncInfo
        be = b.get_encoding().lpbqEncInfo if blockwise else b.get_encoding().axisEncInfo
        if ae.axis != 0 or be.axis != 3:
            raise ValueError("Unexpected projection axis change")
        ac, bc = np.asarray(a.get_data()), np.asarray(b.get_data())[0, 0].T
        passed = (np.array_equal(ac, bc) and len(ae.encInfos) == len(be.encInfos)
                  and all((x.bw, x.scale, x.offset) == (y.bw, y.scale, y.offset)
                          for x, y in zip(ae.encInfos, be.encInfos)))
        if blockwise:
            passed &= (ae.blockSize == be.blockSize and ae.blockScaleBitwidth == be.blockScaleBitwidth
                       and ae.numBlocksPerAxis == be.numBlocksPerAxis
                       and list(ae.blocksScale8 or ae.blocksScale16) == list(be.blocksScale8 or be.blocksScale16))
        layout_checks.append({"weight": name, "parameter_count": int(ac.size), "lpbq": blockwise,
                              "codes_and_scales_identical": bool(passed), "passed": bool(passed)})
        if not passed:
            raise ValueError("Pointwise rewrite changed the quantized weights")

    def abi(graph):
        def describe(t):
            return {"name": t.name(), "shape": t.dims(), "dtype": t.data_type_string()}
        outputs = {}
        for op in graph.get_ops():
            for t in op.outputs():
                if t.is_app_read_tensor():
                    outputs[t.name()] = describe(t)
        return {"inputs": [describe(t) for t in graph.get_input_tensors_to_graph()],
                "outputs": list(outputs.values())}

    source_reader, original = load_graph(parent_state["source_dlc"])
    source_abi, candidate_abi = abi(original), abi(new)
    precision = audit(Path(state["output_dlc"]))
    norm_ops = [v for g in precision["graphs"] for v in g["normalization_ops"]]
    # QAIRT IR reports Pow as Eltwise_Binary. Match retained source node names.
    norm_powers = [v for v in norm_ops if v["name"].endswith("/Pow")]
    updates = [v for g in precision["graphs"] for v in g["state_updates"]]
    norms_passed = len(norm_powers) == 4 and all(
        all(t == "Float_32" for t in v["input_dtypes"] + v["output_dtypes"]) for v in norm_ops)
    cache_passed = len(updates) == 4 and all(
        v["inputs"][0]["dtype"] == v["inputs"][2]["dtype"] == v["outputs"][0]["dtype"] == "Float_16"
        and v["inputs"][0]["shape"] == [1, 32, 1000, 128] for v in updates)
    report = {"format": "moshi-lpbq-pointwise-audit-v1", "script_sha256": sha256(__file__),
              "official_scale_source_url": "https://raw.githubusercontent.com/qualcomm/aimet/develop/TrainingExtensions/onnx/src/python/aimet_onnx/lpbq_utils.py",
              "official_scale_source_sha256": sha256(args.official_source),
              "official_scope": "Three isolated NumPy scale helpers, not the full AIMET package",
              "candidate_dlc_sha256": sha256(state["output_dlc"]), "scale_checks": scale_checks,
              "layout_checks": layout_checks, "source_abi": source_abi, "candidate_abi": candidate_abi,
              "public_abi_identical": source_abi == candidate_abi,
              "normalization_powers": norm_powers, "normalization_ops": norm_ops,
              "normalization_precision_passed": norms_passed,
              "cache_precision_passed": cache_passed, "state_updates": updates,
              "passed": source_abi == candidate_abi and norms_passed and cache_passed,
              "device_verified": False, "independent_quality_test": False}
    write(root / "candidate/lpbq_correctness_audit.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "public_abi_identical", "normalization_precision_passed", "cache_precision_passed")}), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
