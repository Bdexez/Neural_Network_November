"""Comparer précision et temps CPU de deux modes RMSNorm sur les mêmes poids."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.benchmark import Timer

from chimera.calibration import file_sha256
from chimera.cli import prediction_stream, same_file
from chimera.data import synthetic_batch
from chimera.modules import RMSNorm
from chimera.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1901)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--min-run-time', type=float, default=.5)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.threads < 1 or args.repeats < 1 or not 0 <= args.seed < 2**63
            or not math.isfinite(args.min_run_time) or args.min_run_time <= 0):
        parser.error('Tailles, temps et seed invalides')
    if same_file(args.output, args.checkpoint):
        parser.error('La sortie doit être distincte du checkpoint')
    torch.set_num_threads(args.threads)
    digest = file_sha256(args.checkpoint)
    models = {}
    for name, flag in (('historique', False), ('stable', True)):
        models[name], payload = load_checkpoint(args.checkpoint, overrides={'stable_rms_norm': flag})
        models[name].eval()
    config = models['stable'].config
    if config.max_seq_len < 4:
        parser.error('Le scénario requiert max_seq_len >= 4')
    batch = synthetic_batch(config, batch_size=args.batch_size, seed=args.seed)
    batch = {k:v for k,v in batch.items() if k in ('input_ids','attention_mask','series','series_mask')}
    # La même paire de modèles est mesurée en alternant l'ordre pour limiter les effets de dérive.
    times = {name: [] for name in models}
    with torch.inference_mode():
        outputs = {name: model(**batch) for name, model in models.items()}
        for rep in range(args.repeats):
            order = ('historique', 'stable') if rep % 2 == 0 else ('stable', 'historique')
            for name in order:
                timer = Timer('model(**batch)', globals={'model':models[name], 'batch':batch}, num_threads=args.threads)
                times[name].append(timer.blocked_autorange(min_run_time=args.min_run_time).median)
    old, new = outputs['historique'], outputs['stable']
    differences = {name: (getattr(old,name).double() - getattr(new,name).double()).abs().max().item()
                   for name in ('logits', 'classification_logits', 'forecast')}
    classes_changed = (old.classification_logits.argmax(-1) != new.classification_logits.argmax(-1)).sum().item()
    extreme = torch.tensor([[1e20,2e20]], requires_grad=True)
    norm_old, norm_new = RMSNorm(2), RMSNorm(2, stable=True)
    old_extreme, new_extreme = norm_old(extreme), norm_new(extreme)
    gradient = torch.autograd.grad(new_extreme.sum(),extreme)[0]
    medians = {name: statistics.median(values) for name,values in times.items()}
    if file_sha256(args.checkpoint) != digest:
        raise ValueError('Le checkpoint a changé pendant la comparaison')
    report = dict(settings=vars(args), checkpoint_sha256=digest, checkpoint_step=payload['step'],
                  torch_version=str(torch.__version__), machine=platform.machine(), cpu_count=os.cpu_count(),
                  timing_seconds=dict(samples=times, medians=medians,
                                      stable_over_historical=medians['stable']/medians['historique']),
                  max_absolute_difference=differences, class_predictions_changed=classes_changed,
                  extreme_example=dict(input=extreme.detach().tolist(), historical=old_extreme.detach().tolist(),
                                       stable=new_extreme.detach().tolist(), stable_gradient=gradient.tolist()),
                  scope='CPU, un checkpoint et un lot synthétique ; la vitesse dépend du matériel et de la charge.')
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=args.output,checkpoint_sha256=digest)))


if __name__ == '__main__':
    main()
