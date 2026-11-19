import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def routing_worker():
    from chimera import Chimera
    from chimera.cli import main

    capture = Path(os.environ["CHIMERA_ROUTING_CAPTURE"])
    rank = os.environ["RANK"]
    original = Chimera.forward

    def forward(model, *args, **kwargs):
        output = original(model, *args, **kwargs)
        if model.training:
            row = dict(usage=output.expert_usage.detach().cpu().tolist(), tokens=output.routed_tokens.item())
            with (capture / f"rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return output

    Chimera.forward = forward
    main(["--threads", "1", "train", "--steps", "1", "--batch-size", "2", "--accumulation", "2",
          "--modality-policy", "balanced", "--output", str(capture / "run")])


class RoutingTrainingTests(unittest.TestCase):
    def test_training_reports_token_weighted_usage_across_all_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, OMP_NUM_THREADS="1", CHIMERA_ROUTING_CAPTURE=directory)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                       str(Path(__file__).resolve()), "--worker"]
            result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout[-6000:])
            samples = [json.loads(line) for rank in range(2)
                       for line in (root / f"rank{rank}.jsonl").read_text().splitlines()]
            self.assertEqual(len(samples), 4)
            self.assertGreater(len({row["tokens"] for row in samples}), 1)
            weighted = sum(torch.tensor(row["usage"], dtype=torch.float64) * row["tokens"] for row in samples)
            expected = weighted / weighted.sum(-1, keepdim=True)
            report = json.loads((root / "run/metrics.jsonl").read_text())
            self.assertEqual(report["expert_routed_tokens"], sum(row["tokens"] for row in samples))
            torch.testing.assert_close(torch.tensor(report["expert_usage"], dtype=torch.float64), expected)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        routing_worker()
    else:
        unittest.main()
