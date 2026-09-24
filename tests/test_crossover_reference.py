import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "crossover_fsr4", Path(__file__).resolve().parents[1] / "tools/crossover/crossover_fsr4.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class CrossoverReferenceTest(unittest.TestCase):
    def test_old_gpu_uses_fresh_fp16_reference_even_with_int8_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testcase = root / "testcases/testcase_quality_720to1080"
            testcase.mkdir(parents=True)
            (testcase / "meta.json").write_text("{}")
            (root / "out").mkdir()
            (root / "out/f00_out.bin").write_bytes(b"stale M5 INT8 frame")
            info = {"device": "Apple M1 Pro", "backend": "f16", "reason": "no GPU tensor acceleration"}

            def run(command, **kwargs):
                if command[0] == str(root / "tools/bin/fsr4mac"):
                    self.assertEqual(command[-1], "--fp16")
                    destination = Path(command[2])
                    destination.mkdir(parents=True)
                    (destination / "f00_out.bin").write_bytes(b"fresh FP16 frame")

            with patch.object(installer, "ROOT", root), \
                 patch.object(installer, "DIST", root / "dist/crossover"), \
                 patch.object(installer, "run", side_effect=run), \
                 patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)) as query:
                reference = installer.build_reference({"FSR4_MODEL": "int8"})
                self.assertEqual(reference.read_bytes(), b"fresh FP16 frame")
                self.assertEqual(query.call_args.args[0][-1], "i8mpp")
                self.assertEqual((root / "out/f00_out.bin").read_bytes(), b"stale M5 INT8 frame")


if __name__ == "__main__":
    unittest.main()
