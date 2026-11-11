"""Mesurer la mémoire et le travail du cache, du lecteur JSONL et du chargement."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import JointDataset, synthetic_batch
from chimera.training import load_checkpoint


def profile_cache():
    torch.manual_seed(42)
    config = ChimeraConfig.preset("tiny")
    model = Chimera(config).eval()
    batch = synthetic_batch(config, batch_size=1, seed=7)
    ids = torch.randint(4, config.vocab_size, (1, 160))
    projections = [0] * config.n_layers
    handles = []
    for index, block in enumerate(model.blocks):
        def count(module, args, output, index=index):
            projections[index] += 1
        handles.append(block.cross.kv.register_forward_hook(count))
    start = time.perf_counter()
    with torch.inference_mode():
        result = model(ids[:, :96], images=batch["images"], series=batch["series"], use_cache=True)
        for position in range(96, ids.size(1)):
            result = model(ids[:, position:position + 1], cache=result.cache, use_cache=True)
    elapsed = time.perf_counter() - start
    for handle in handles:
        handle.remove()
    layers = result.cache.layers
    return {
        "preset": "tiny", "prefill_tokens": 96, "decoded_tokens": 64,
        "seconds": elapsed,
        "cached_tokens_per_layer": [layer.key.size(-2) for layer in layers],
        "self_attention_kv_bytes": sum(t.numel() * t.element_size() for layer in layers
                                       for t in (layer.key, layer.value)),
        "context_projections_per_layer": projections,
        "last_logits_checksum": result.logits.double().sum().item(),
    }


def profile_dataset(path):
    tracemalloc.start()
    start = time.perf_counter()
    dataset = JointDataset(path, ChimeraConfig.preset("tiny"))
    dataset[0]
    dataset[len(dataset) - 1]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"examples": len(dataset), "file_bytes": Path(path).stat().st_size,
            "peak_python_bytes": peak, "seconds": time.perf_counter() - start}


def profile_checkpoint(path):
    start = time.perf_counter()
    model, payload = load_checkpoint(path)
    return {
        "seconds": time.perf_counter() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "parameters": sum(p.numel() for p in model.parameters()),
        "step": payload["step"],
        "scope": "Checkpoint loading only, before inference; mmap pages are read on demand.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.threads < 1 or not any((args.dataset, args.checkpoint, args.cache)):
        parser.error("Choisir --cache, --dataset ou --checkpoint, avec threads > 0")
    torch.set_num_threads(args.threads)
    report = {"torch_version": torch.__version__, "threads": args.threads}
    if args.cache:
        report["cache"] = profile_cache()
    if args.dataset:
        report["dataset"] = profile_dataset(args.dataset)
    if args.checkpoint:
        report["checkpoint"] = profile_checkpoint(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
