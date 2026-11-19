from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.cli import main
from chimera.observed_objective import observed_objective, target_counts
from chimera.training import DistributedTrainingModel, load_checkpoint


def test_config():
    return replace(ChimeraConfig.preset("tiny"), prediction_steps=4, balance_weight=0,
                   router_z_weight=0, alignment_weight=0)


def test_batch(config):
    batch = synthetic_batch(config, batch_size=4, seed=912)
    batch["labels"][0, 2:] = -100
    batch["labels"][1, 5:] = -100
    batch["class_labels"][:3] = -100
    batch["forecast_targets"][1:] = float("nan")
    batch["forecast_targets"][0, 4:] = float("nan")
    return batch


def distributed_worker(rank, init_file, output_dir, mixed=False):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        config = test_config()
        torch.manual_seed(316)
        model = Chimera(config).eval()
        wrapped = DistributedDataParallel(DistributedTrainingModel(model), find_unused_parameters=True)
        batch = test_batch(config)
        chunks = [{name: value[rank * 2 + index:rank * 2 + index + 1] for name, value in batch.items()}
                  for index in range(2)]
        if mixed is True:
            for chunk in chunks:
                chunk.pop("class_labels" if rank == 0 else "forecast_targets")
            if rank == 1:
                for key in ("input_ids", "attention_mask", "labels"):
                    chunks[1].pop(key)
        elif mixed == "unlabeled" and rank == 1:
            for chunk in chunks:
                for key in ("labels", "class_labels", "forecast_targets"):
                    chunk.pop(key)
        counts = [target_counts(chunk, config.prediction_steps) for chunk in chunks]
        total = sum(counts)
        dist.all_reduce(total)
        for index, (chunk, local) in enumerate(zip(chunks, counts)):
            with wrapped.no_sync() if index == 0 else nullcontext():
                loss, _, _ = wrapped(local_counts=local, total_counts=total, accumulation=2, world=2, **chunk)
                loss.backward()
        torch.save({name: p.grad for name, p in model.named_parameters()}, Path(output_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class ObservedObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_accumulated_gradients_match_full_batch_with_missing_targets(self):
        config = test_config()
        torch.manual_seed(316)
        full = Chimera(config).eval()
        accumulated = Chimera(config).eval()
        accumulated.load_state_dict(full.state_dict())
        batch = test_batch(config)
        chunks = [{name: value[section] for name, value in batch.items()}
                  for section in (slice(0, 1), slice(1, 4))]
        counts = [target_counts(chunk, config.prediction_steps) for chunk in chunks]
        total = sum(counts)
        torch.testing.assert_close(total, target_counts(batch, config.prediction_steps))
        full(**batch).loss.backward()
        for chunk, local in zip(chunks, counts):
            output = accumulated(**chunk)
            observed_objective(output, local, total, config, world=1, accumulation=2).backward()
        for (name, expected), (_, actual) in zip(full.named_parameters(), accumulated.named_parameters()):
            if expected.grad is None:
                self.assertIsNone(actual.grad, name)
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=1e-6, msg=name)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo indisponible")
    def test_distributed_gradients_match_full_batch_with_missing_targets(self):
        config = test_config()
        torch.manual_seed(316)
        model = Chimera(config).eval()
        model(**test_batch(config)).loss.backward()
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(distributed_worker, args=(str(Path(directory) / "group"), directory),
                                        nprocs=2, join=True)
            for rank in range(2):
                gradients = torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True)
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or gradients[name] is None:
                        existing = parameter.grad if parameter.grad is not None else gradients[name]
                        if existing is not None:
                            self.assertEqual(existing.count_nonzero().item(), 0, name)
                    else:
                        torch.testing.assert_close(gradients[name], parameter.grad, rtol=3e-4, atol=1e-6, msg=name)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo indisponible")
    def test_distributed_gradients_with_varying_modalities(self):
        config = test_config()
        torch.manual_seed(316)
        model = Chimera(config).eval()
        batch = test_batch(config)
        chunks = [{name: value[index:index + 1] for name, value in batch.items()} for index in range(4)]
        for chunk in chunks[:2]:
            chunk.pop("class_labels")
        for chunk in chunks[2:]:
            chunk.pop("forecast_targets")
        for key in ("input_ids", "attention_mask", "labels"):
            chunks[-1].pop(key)
        counts = [target_counts(chunk, config.prediction_steps) for chunk in chunks]
        total = sum(counts)
        for chunk, count in zip(chunks, counts):
            observed_objective(model(**chunk), count, total, config, world=1, accumulation=4).backward()
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(distributed_worker, args=(str(Path(directory) / "group"), directory, True),
                                        nprocs=2, join=True)
            for rank in range(2):
                gradients = torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True)
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or gradients[name] is None:
                        existing = parameter.grad if parameter.grad is not None else gradients[name]
                        if existing is not None:
                            self.assertEqual(existing.count_nonzero().item(), 0, name)
                    else:
                        torch.testing.assert_close(gradients[name], parameter.grad, rtol=3e-4, atol=1e-6, msg=name)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo indisponible")
    def test_distributed_rank_can_have_no_supervised_targets(self):
        config = test_config()
        torch.manual_seed(316)
        model = Chimera(config).eval()
        batch = test_batch(config)
        chunks = [{name: value[index:index + 1] for name, value in batch.items()} for index in range(4)]
        for chunk in chunks[2:]:
            for key in ("labels", "class_labels", "forecast_targets"):
                chunk.pop(key)
        counts = [target_counts(chunk, config.prediction_steps) for chunk in chunks]
        total = sum(counts)
        for chunk, count in zip(chunks, counts):
            observed_objective(model(**chunk), count, total, config, world=1, accumulation=4).backward()
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(distributed_worker,
                                        args=(str(Path(directory) / "group"), directory, "unlabeled"),
                                        nprocs=2, join=True)
            for rank in range(2):
                gradients = torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True)
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or gradients[name] is None:
                        existing = parameter.grad if parameter.grad is not None else gradients[name]
                        if existing is not None:
                            self.assertEqual(existing.count_nonzero().item(), 0, name)
                    else:
                        torch.testing.assert_close(gradients[name], parameter.grad, rtol=3e-4, atol=1e-6, msg=name)

    def test_cli_accumulation_matches_full_batch_with_partial_supervision(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = replace(ChimeraConfig.preset("tiny"), prediction_steps=4, balance_weight=0,
                             router_z_weight=0, alignment_weight=0)
            config_path = root / "model.json"
            config.save(config_path)
            batch = synthetic_batch(config, batch_size=4, seed=1009)
            records = []
            for index in range(4):
                record = dict(text="s" * (index + 1), series=batch["series"][index].tolist())
                if index != 1:
                    record["class_label"] = int(batch["class_labels"][index])
                if index == 0:
                    record["future"] = batch["forecast_targets"][index].tolist()
                records.append(record)
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps(record) for record in records))
            common = ["--threads", "2", "train", "--config", str(config_path), "--data", str(data),
                      "--steps", "1", "--warmup", "0", "--no-shuffle", "--modality-dropout", "0",
                      "--loss-normalization", "observed", "--grad-clip", "100"]
            full, accumulated = root / "full", root / "accumulated"
            main(common + ["--batch-size", "4", "--output", str(full)])
            main(common + ["--batch-size", "2", "--accumulation", "2", "--output", str(accumulated)])
            expected, _ = load_checkpoint(full / "latest.pt")
            actual, payload = load_checkpoint(accumulated / "latest.pt")
            self.assertEqual(payload["trainer"]["loss_normalization"], "observed")
            for name, tensor in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=2e-4, atol=2e-5, msg=name)
            first = json.loads((full / "metrics.jsonl").read_text())
            second = json.loads((accumulated / "metrics.jsonl").read_text())
            self.assertAlmostEqual(first["loss"], second["loss"], places=5)

    def test_unlabeled_microbatch_is_allowed_when_step_has_targets(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = replace(ChimeraConfig.preset("tiny"), balance_weight=0, router_z_weight=0, alignment_weight=0)
            config_path = root / "model.json"
            config.save(config_path)
            batch = synthetic_batch(config, batch_size=4, seed=1010)
            records = [dict(series=batch["series"][index].tolist()) for index in range(4)]
            records[2]["class_label"] = 1
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps(row) for row in records))
            common = ["--threads", "2", "train", "--config", str(config_path), "--data", str(data),
                      "--steps", "1", "--warmup", "0", "--no-shuffle", "--modality-dropout", "0",
                      "--loss-normalization", "observed", "--grad-clip", "100"]
            full, accumulated = root / "full", root / "accumulated"
            main(common + ["--batch-size", "4", "--output", str(full)])
            main(common + ["--batch-size", "2", "--accumulation", "2", "--output", str(accumulated)])
            expected, _ = load_checkpoint(full / "latest.pt")
            actual, _ = load_checkpoint(accumulated / "latest.pt")
            for name, tensor in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=2e-4, atol=2e-5, msg=name)
            data.write_text("\n".join(json.dumps(dict(series=row["series"])) for row in records))
            with self.assertRaisesRegex(ValueError, "étape ne contient aucune cible"):
                main(common + ["--batch-size", "2", "--accumulation", "2", "--output", str(root / "empty")])

    def test_observed_mode_is_restored_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = root / "model.json"
            replace(ChimeraConfig.preset("tiny"), dropout=0.1).save(config)
            common = ["--threads", "2", "train", "--config", str(config), "--steps", "4",
                      "--batch-size", "2", "--accumulation", "2", "--eval-every", "2"]
            full, split = root / "full", root / "split"
            main(common + ["--output", str(full), "--loss-normalization", "observed"])
            main(common + ["--output", str(split), "--loss-normalization", "observed", "--stop-after", "1"])
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            expected, _ = load_checkpoint(full / "latest.pt")
            actual, payload = load_checkpoint(split / "latest.pt")
            self.assertEqual(payload["trainer"]["loss_normalization"], "observed")
            for name, tensor in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0, msg=name)
            with self.assertRaisesRegex(ValueError, "loss_normalization"):
                main(common + ["--output", str(split), "--resume", str(split / "latest.pt"),
                               "--loss-normalization", "batch"])

    def test_legacy_checkpoint_defaults_to_batch_normalization(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            common = ["--threads", "2", "train", "--steps", "2", "--batch-size", "2", "--eval-every", "2"]
            initial = root / "initial"
            main(common + ["--output", str(initial), "--stop-after", "1"])
            source = initial / "latest.pt"
            payload = torch.load(source, weights_only=True)
            del payload["trainer"]["loss_normalization"]
            legacy = root / "legacy.pt"
            torch.save(payload, legacy)
            main(common + ["--output", str(root / "current"), "--resume", str(source)])
            main(common + ["--output", str(root / "legacy"), "--resume", str(legacy)])
            expected, _ = load_checkpoint(root / "current/latest.pt")
            actual, payload = load_checkpoint(root / "legacy/latest.pt")
            self.assertEqual(payload["trainer"]["loss_normalization"], "batch")
            for name, tensor in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
