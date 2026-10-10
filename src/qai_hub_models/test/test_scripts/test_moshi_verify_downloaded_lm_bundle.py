"""Transport checks are separate from model execution/parity checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[4] / "scripts/moshi_verify_downloaded_lm_bundle.py"
spec = importlib.util.spec_from_file_location("bundle_verify", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BundleVerificationTest(unittest.TestCase):
    def fixture(self, root):
        names = ["frontend", "head"]
        names += [f"temporal_layers_{i}_{i+1}" for i in range(0, 32, 2)]
        names += [f"depformer_codebook_{i}" for i in range(8)]
        (root / "dlc").mkdir()
        inventory = {"all_graphs_verified": True, "graphs": {}}
        manifest = {"graphs": {}}
        sums = []
        for name in names:
            content = name.encode()
            filename = f"dlc/{name}.dlc"
            (root / filename).write_bytes(content)
            checksum = hashlib.sha256(content).hexdigest()
            inventory["graphs"][name] = {"file": filename, "sha256": checksum, "compiled_model_id": name}
            manifest["graphs"][name] = {"compiled_model_id": name}
            sums.append(f"{checksum}  {filename}\n")
        (root / "dlc_inventory.json").write_text(json.dumps(inventory))
        (root / "quantization_manifest.json").write_text(json.dumps(manifest))
        (root / "SHA256SUMS").write_text("".join(sums))

    def test_complete_delivery_checks_all_26(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            result = module.verify(root)
            self.assertTrue(result["download_verified"])
            self.assertEqual(len(result["graphs"]), 26)

    def test_corrupt_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            (root / "dlc/frontend.dlc").write_bytes(b"incomplete")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                module.verify(root)


if __name__ == "__main__":
    unittest.main()
