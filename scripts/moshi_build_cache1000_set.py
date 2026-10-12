"""Resumable remote CPU-validation/conversion of existing guarded Temporal sources.

No weight export/recalibration or cloud uploads. A conversion is NOT a device
accuracy pass. Continue remaining graphs after individual failures. Keep outputs
on data3, snapshot source metadata/weights and freeze helper scripts for recovery.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import onnx
from moshi_patch_temporal_cache import semantic_bytes


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Snapshot size mismatch: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def resolve_source(baseline, graph):
    state = json.loads((baseline / "graphs" / graph / "state.json").read_text())
    if state.get("status") != "verified":
        raise ValueError(f"Baseline graph is not verified: {graph}")
    dlc = Path(state["dlc"])
    candidates = [dlc.parent / "conversion.json"]
    if graph == "temporal_layers_0_1":
        candidates.insert(0, dlc.parent / "full_htp_convert.json")
    for receipt in candidates:
        if not receipt.is_file():
            continue
        data = json.loads(receipt.read_text())
        args = data.get("converter_args", [])
        if data.get("status") != "success" or "--input_network" not in args:
            continue
        if Path(args[args.index("--output_path") + 1]).resolve() != dlc.resolve():
            continue
        source = Path(args[args.index("--input_network") + 1])
        if not source.is_file():
            raise FileNotFoundError(source)
        return source, receipt
    raise ValueError(f"No successful exact-source conversion receipt for {graph}")


def run(command, log, timeout, env=None):
    with log.open("a") as handle:
        handle.write("\nCOMMAND " + repr(command) + "\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "output-root", "calibration-dir", "sdk-root", "numpy-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=int, default=20)
    parser.add_argument("--graph", action="append")
    parser.add_argument("--reuse-six-seven", type=Path)
    parser.add_argument("--reuse-build-root", type=Path,
                        help="Reuse matching source snapshots and converted candidates from a prior revision")
    parser.add_argument("--fallback-where", action="store_true",
                        help="Try capacity1000 Where only after Scatter numerical checkpoint failure")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1..4 workers")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    helpers = root / "runtime_scripts"
    helpers.mkdir(exist_ok=True)
    for filename in ("moshi_patch_temporal_cache.py", "moshi_guard_qairt_converter.py"):
        source = Path(__file__).with_name(filename)
        destination = helpers / filename
        if destination.exists() and digest(source) != digest(destination):
            raise ValueError("Helper code changed; use a new build root rather than mixing revisions")
        if not destination.exists():
            shutil.copy2(source, destination)
    plan = {"format": "moshi-cache1000-set-plan-v1", "baseline": str(args.baseline_root),
            "capacity": 1000, "cache_output": "delta", "update": "scatter",
            "patcher_sha256": digest(helpers / "moshi_patch_temporal_cache.py"),
            "fallback_where": args.fallback_where,
            "sdk": str(args.sdk_root), "numpy_dir": str(args.numpy_dir),
            "calibration": str(args.calibration_dir)}
    if (root / "plan.json").exists() and json.loads((root / "plan.json").read_text()) != plan:
        raise ValueError("Build plan changed; refusing stale resume")
    write(root / "plan.json", plan)
    write(root / "process.json", {"pid": os.getpid(), "started_at": time.time()})
    graphs = args.graph or [f"temporal_layers_{i}_{i+1}" for i in range(0, 32, 2)]
    if any(g not in [f"temporal_layers_{i}_{i+1}" for i in range(0, 32, 2)] for g in graphs):
        raise ValueError("Only Temporal graphs can be rebuilt by this script")

    def build(graph):
        directory = root / "graphs" / graph
        directory.mkdir(parents=True, exist_ok=True)
        status = {"graph": graph, "device_verified": False}
        def stage(name, **extra):
            status.update(status=name, updated_at=time.time(), **extra)
            write(directory / "state.json", status)
            print(f"{graph}: {name}", flush=True)
        try:
            if shutil.disk_usage(root).free < (args.minimum_free_gib + 4) * 1024**3:
                raise RuntimeError("Insufficient data3 headroom before snapshot/conversion")
            source, receipt = resolve_source(args.baseline_root, graph)
            model = onnx.load(str(source), load_external_data=False)
            inputs = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in model.graph.input}
            original = directory / "reference" / "model.onnx"
            prior = args.reuse_build_root / "graphs" / graph if args.reuse_build_root else None
            snapshot = source
            if prior and (prior / "signature.json").is_file():
                old_signature = json.loads((prior / "signature.json").read_text())
                if (old_signature["source"] == str(source)
                        and old_signature["source_size"] == source.stat().st_size
                        and old_signature["source_mtime_ns"] == source.stat().st_mtime_ns
                        and old_signature["receipt"] == digest(receipt)
                        and digest(prior / "reference/model.onnx") == digest(source)):
                    snapshot = prior / "reference/model.onnx"
            copy_once(snapshot, original)
            external = {}
            for tensor in model.graph.initializer:
                metadata = {v.key: v.value for v in tensor.external_data}
                if "location" not in metadata:
                    continue
                relative = Path(metadata["location"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe external-data path")
                info = (source.parent / relative).stat()
                external[str(relative)] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
                snapshot_weight = snapshot.parent / relative
                if snapshot != source and old_signature["external_files"].get(str(relative)) != external[str(relative)]:
                    raise ValueError("Prior external-data snapshot signature differs from source")
                copy_once(snapshot_weight, original.parent / relative)
            signature = {"source": str(source), "source_size": source.stat().st_size,
                         "source_mtime_ns": source.stat().st_mtime_ns, "receipt": digest(receipt),
                         "inputs": inputs, "external_files": external, "plan": plan}
            if (directory / "signature.json").exists() and json.loads((directory / "signature.json").read_text()) != signature:
                raise ValueError("Source signature changed; refuse stale model/conversion reuse")
            write(directory / "signature.json", signature)
            log = directory / "build.log"
            samples = []
            for frame in (0, 1):
                matches = list((args.calibration_dir / "samples" / graph).glob(f"clean-000_*_frame_{frame:04d}.npz"))
                if len(matches) != 1:
                    raise ValueError("Missing or ambiguous clean-000 calibration sample")
                samples.append(matches[0])
            stage("patch_and_cpu_validate", source=str(source))
            reused = None
            if graph == "temporal_layers_6_7" and args.reuse_six_seven:
                reused = args.reuse_six_seven
                copy_once(reused / "model.dlc", directory / "model.dlc")
                # Exact semantic model comparison in --validate-existing below
                # guards against accidentally taking a different prototype.
            command = [sys.executable, str(helpers / "moshi_patch_temporal_cache.py"),
                       "--model", str(original), "--output-dir", str(directory),
                       "--capacity", "1000", "--cache-output", "delta", "--scatter-encoding", "aligned"]
            # The graph directory already contains recovery metadata. Generate
            # the candidate in its own child, then link it into this directory.
            candidate = directory / "candidate"
            command[command.index("--output-dir") + 1] = str(candidate)
            if (candidate / "model.onnx").is_file():
                command += ["--validate-existing"]
            command += ["--samples", *(str(p) for p in samples)]
            cpu_signature = directory / "cpu_signature.json"
            cpu_report = candidate / "patch_report.json"
            cpu_complete = (cpu_signature.is_file() and cpu_report.is_file()
                and json.loads(cpu_signature.read_text()) == signature
                and json.loads(cpu_report.read_text()).get("passed")
                and len(json.loads(cpu_report.read_text()).get("samples", [])) == 2)
            if not cpu_complete:
                try:
                    run(command, log, 1800)
                except subprocess.CalledProcessError as exc:
                    if not (args.fallback_where and exc.returncode == 2 and cpu_report.is_file()
                            and json.loads(cpu_report.read_text()).get("passed") is False):
                        raise
                    # Retain failed Scatter candidate/report. Never hide a
                    # shape, converter, or runtime exception behind fallback.
                    candidate = directory / "candidate_where"
                    fallback = list(command)
                    fallback[fallback.index("--output-dir") + 1] = str(candidate)
                    if "--validate-existing" in fallback:
                        fallback.remove("--validate-existing")
                    if (candidate / "model.onnx").is_file():
                        fallback += ["--validate-existing"]
                    fallback += ["--update", "where"]
                    stage("where_fallback_cpu_validate", scatter_report=str(cpu_report))
                    run(fallback, log, 1800)
                    status["selected_update"] = "where"
                    reused = None
                write(cpu_signature, signature)
            candidate_model = onnx.load(str(candidate / "model.onnx"), load_external_data=False)
            if reused and semantic_bytes(candidate_model) != semantic_bytes(onnx.load(str(reused / "model.onnx"), load_external_data=False)):
                raise ValueError("Reused prototype is not identical to requested patched model")
            if not reused and prior and (prior / "conversion.json").is_file():
                old_conversion = json.loads((prior / "conversion.json").read_text())
                if (old_conversion.get("status") == "success" and (prior / "model.dlc").is_file()
                        and semantic_bytes(candidate_model) == semantic_bytes(onnx.load(
                            str(prior / "candidate/model.onnx"), load_external_data=False))):
                    copy_once(prior / "model.dlc", directory / "model.dlc")
                    reused = prior
            if shutil.disk_usage(root).free < args.minimum_free_gib * 1024**3:
                raise RuntimeError("Insufficient data3 headroom; leave prior results intact")
            conversion = directory / "conversion.json"
            dlc = directory / "model.dlc"
            complete = conversion.exists() and json.loads(conversion.read_text()).get("status") == "success" and dlc.is_file()
            if reused:
                write(conversion, {"status": "success", "reused_from": str(reused), "candidate_model_identical": True})
                complete = True
            if not complete:
                stage("converting", cpu_validated=True)
                # Never overwrite a failed/running receipt or partial DLC.
                # Successful attempt receipts can be reused on later resumes.
                attempt = None
                for old in sorted((directory / "attempts").glob("*/conversion.json")):
                    info = json.loads(old.read_text())
                    if info.get("status") == "success" and (old.parent / "model.dlc").is_file():
                        attempt = old.parent
                if attempt is None:
                    attempt = directory / "attempts" / str(time.time_ns())
                    attempt.mkdir(parents=True)
                attempt_receipt = attempt / "conversion.json"
                attempt_dlc = attempt / "model.dlc"
                preserve = [v.name for v in list(candidate_model.graph.input) + list(candidate_model.graph.output)
                            if v.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)]
                command = [sys.executable, str(helpers / "moshi_guard_qairt_converter.py"),
                           "--sdk-root", str(args.sdk_root), "--numpy-dir", str(args.numpy_dir),
                           "--receipt", str(attempt_receipt), "--", "--input_network", str(candidate / "model.onnx"),
                           "--output_path", str(attempt_dlc), "--preserve_onnx_output_order", "--onnx_skip_simplification",
                           "--source_model_input_datatype", "position", "int32", "--preserve_io_datatype", *preserve]
                env = dict(os.environ, LD_LIBRARY_PATH=str(args.sdk_root / "lib/x86_64-linux-clang"))
                if not attempt_receipt.exists():
                    run(command, log, 2400, env)
                if json.loads(attempt_receipt.read_text()).get("status") != "success":
                    raise ValueError("Conversion did not finish successfully")
                copy_once(attempt_dlc, dlc)
                write(conversion, {"status": "success", "attempt": str(attempt), "candidate": str(candidate / "model.onnx")})
            if not dlc.is_file() or not dlc.stat().st_size:
                raise ValueError("Missing converted DLC")
            stage("converted_not_device_verified", dlc=str(dlc), candidate=str(candidate / "model.onnx"),
                  bytes=dlc.stat().st_size, cpu_validated=True)
        except Exception as exc:
            stage("failed", error=str(exc))
        return status

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(build, graph) for graph in graphs]):
            results.append(future.result())
            write(root / "build_result.json", {"graphs": results, "device_verified": False,
                "all_converted": len(results) == len(graphs) and all(v["status"] == "converted_not_device_verified" for v in results)})
    print("Build finished; read build_result.json. Converted is not a device pass.", flush=True)


if __name__ == "__main__":
    main()
