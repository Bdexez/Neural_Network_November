from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from chimera import ChimeraConfig


class DistributedResumeTests(unittest.TestCase):
    def test_two_rank_resume_restores_each_random_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "model.json"
            replace(ChimeraConfig.preset("tiny"), dropout=0.2).save(config)
            data = root / "data.jsonl"
            records = [dict(text="Classe " + str(i) + "!" * i,
                            series=[[float(i), float(t), float(i + t)] for t in range(8 + i % 3)],
                            class_label=i % 6) for i in range(9)]
            data.write_text("\n".join(json.dumps(row) for row in records))
            common = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                      "-m", "chimera", "--threads", "1", "train", "--config", str(config),
                      "--steps", "5", "--batch-size", "2", "--accumulation", "2", "--data", str(data),
                      "--val-data", str(data), "--eval-every", "2", "--save-every", "2",
                      "--modality-dropout", "0.4", "--series-missing-rate", "0.2", "--seed", "44"]
            env = dict(os.environ, OMP_NUM_THREADS="1")
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)

            def run(*options):
                result = subprocess.run([*common, *map(str, options)], env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                return result.stdout

            full, split, from_best = root / "full", root / "split", root / "from-best"
            run("--output", full)
            run("--output", split, "--stop-after", "3")
            snapshot = torch.load(split / "latest.pt", weights_only=True)
            self.assertEqual(len(snapshot["rank_rng"]), 2)
            self.assertFalse(torch.equal(snapshot["rank_rng"][0]["torch"], snapshot["rank_rng"][1]["torch"]))
            legacy = root / "legacy.pt"
            del snapshot["rank_rng"]
            torch.save(snapshot, legacy)
            self.assertIn("les états aléatoires par rang ne sont pas disponibles",
                          run("--output", root / "legacy", "--resume", legacy))
            run("--output", from_best, "--resume", split / "best.pt")
            run("--output", split, "--resume", split / "latest.pt")
            expected = torch.load(full / "latest.pt", weights_only=True)
            for directory in (split, from_best):
                actual = torch.load(directory / "latest.pt", weights_only=True)
                self.assertEqual(expected["trainer"]["validation_state"], actual["trainer"]["validation_state"])
                self.assertEqual(expected["step"], actual["step"])
                for name, tensor in expected["model"].items():
                    torch.testing.assert_close(tensor, actual["model"][name], rtol=0, atol=0, msg=name)
                for key, state in expected["optimizer"]["state"].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, actual["optimizer"]["state"][key][name], rtol=0, atol=0)
                for rank in range(2):
                    torch.testing.assert_close(expected["rank_rng"][rank]["torch"], actual["rank_rng"][rank]["torch"],
                                               rtol=0, atol=0)
                    self.assertEqual(expected["rank_rng"][rank]["python"], actual["rank_rng"][rank]["python"])


if __name__ == "__main__":
    unittest.main()
