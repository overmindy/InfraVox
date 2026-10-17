"""Audit selective Mimi DLC weights, then compile isolated SM8850 contexts."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from moshi_quantize_mimi_offline import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "sdk-root", "numpy-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root/"lib/python")]
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from qti.aisw.dlc_utils import modeltools

    for comp in ("encoder", "decoder"):
        root = args.directory/comp
        state = json.loads((root/"RESULT.json").read_text())
        assert state["cpu_screen"]["passed"] and state["conversion"]["returncode"] == 0
        dlc = root/"model.dlc"
        assert sha(dlc) == state["dlc"]["sha256"]
        model = onnx.load(state.get("quantizer_source", state["source_onnx"]))
        weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        policy = json.loads((root/"quantization_overrides.json").read_text())
        expected = {t["name"]: t for t in policy["param_encodings"] if t["dtype"] == "INT"}
        reader = modeltools.IrDlcReader()
        reader.open(str(dlc))
        graph = reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
        records, tensors, static = [], {}, {}
        for op in graph.get_ops():
            for t in (*op.inputs(), *op.outputs()):
                tensors[t.name()] = t.data_type_string()
                if t.is_static():
                    static[t.name()] = (t.data_type_string(), np.asarray(t.get_data()).nbytes)
            if op.type not in ("FullyConnected", "Conv2d") or op.inputs()[1].name() not in expected:
                continue
            t = op.inputs()[1]
            blocked = "BLOCKWISE_EXPANSION" in str(t.get_encoding().type)
            e = t.get_encoding().lpbqEncInfo if blocked else t.get_encoding().axisEncInfo
            conv = op.type == "Conv2d"
            assert e.axis == (3 if conv else 0)
            bits = e.encInfos[0].bw - e.blockScaleBitwidth if blocked else e.encInfos[0].bw
            scales = np.array([x.scale for x in e.encInfos], np.float32)
            offsets = np.array([x.offset for x in e.encInfos], np.float32)
            raw = np.asarray(t.get_data()).astype(np.float32)
            raw = raw[0, 0].T if conv else raw
            if blocked:
                raw = np.where(raw > 2**(bits-1)-1, raw-2**bits, raw)
                expansion = np.asarray(e.blocksScale8 or e.blocksScale16, np.float32).reshape(len(scales), e.numBlocksPerAxis)
                grid = scales[:, None]*np.repeat(expansion, e.blockSize, axis=1)
            else:
                grid = scales[:, None]
            decoded = (raw+offsets[:, None])*grid
            original = weights[t.name()][:, :, 0, 0] if conv else weights[t.name()].T
            if blocked:
                authored = original
            else:
                authored_scales = np.array(expected[t.name()]["scale"], np.float32)[:, None]
                authored = np.clip(np.rint(original/authored_scales), -128, 127)*authored_scales
            error = float(np.linalg.norm(decoded-authored)/max(np.linalg.norm(authored), 1e-12))
            row = {"name": op.name, "weight": t.name(), "input_dtype": op.inputs()[0].data_type_string(),
                   "weight_dtype": t.data_type_string(), "output_dtype": op.outputs()[0].data_type_string(),
                   "authored_weight_relative_rms_difference": error, "shape": t.dims(), "bitwidth": bits}
            records.append(row)
        precision = len(records) == len(expected) == 32 and all(
            r["input_dtype"] == r["output_dtype"] == "uFxp_16" and r["weight_dtype"] == "sFxp_8"
            and r["bitwidth"] == expected[r["weight"]].get("compressed_bw", expected[r["weight"]]["bw"])
            and r["authored_weight_relative_rms_difference"] < .002 for r in records)
        abi = {n: tensors[n] for n in ("kv_cache", "conv_state", "kv_cache_out", "conv_state_out", "audio", "codes")}
        assert all(dtype == "Float_32" for n, dtype in abi.items() if n != "codes")
        assert abi["codes"] == "Int_32" and precision
        audit = {"passed": precision, "fc": records, "public_abi": abi,
                 "static_bytes_by_dtype": {dt: sum(size for kind, size in static.values() if kind == dt)
                                           for dt in {kind for kind, _ in static.values()}},
                 "static_dtype_counts": dict(Counter(kind for kind, _ in static.values()))}
        save(root/"PRECISION_AUDIT.json", audit)
        state["actual_precision_passed"] = True
        output = root/"htp"
        output.mkdir(exist_ok=False)
        lib = args.sdk_root/"lib/x86_64-linux-clang"
        save(output/"backend.json", {"devices": [{"dsp_arch": "v81", "soc_model": 87}]})
        save(output/"config.json", {"backend_extensions": {"shared_library_path": str(lib/"libQnnHtpNetRunExtensions.so"),
                                                           "config_file_path": str(output/"backend.json")}})
        cmd = [str(args.sdk_root/"bin/x86_64-linux-clang/qnn-context-binary-generator"),
               "--backend", str(lib/"libQnnHtp.so"), "--model", str(lib/"libQnnModelDlc.so"),
               "--dlc_path", str(dlc), "--config_file", str(output/"config.json"),
               "--binary_file", "mimi_"+comp, "--output_dir", str(output), "--log_level", "warn"]
        start = time.monotonic()
        with (output/"compile.log").open("w") as log:
            run = subprocess.run(cmd, env=dict(os.environ, LD_LIBRARY_PATH=str(lib)), stdout=log, stderr=subprocess.STDOUT)
        result = {"command": cmd, "returncode": run.returncode, "seconds": time.monotonic()-start}
        binary = output/("mimi_"+comp+".bin")
        if run.returncode == 0 and binary.exists():
            result.update(binary=str(binary), bytes=binary.stat().st_size, sha256=sha(binary))
        state["htp_compile"] = result
        save(root/"RESULT.json", state)
        print(comp, "actual precision passed; HTP compile", run.returncode, result.get("bytes"), flush=True)
        assert run.returncode == 0


if __name__ == "__main__":
    main()
