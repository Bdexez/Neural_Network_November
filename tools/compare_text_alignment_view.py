"""Comparer deux vues textuelles pour l'alignement sur des entraînements synthétiques appariés."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from chimera.calibration import file_sha256
from chimera.cli import prediction_stream
from chimera.data import synthetic_batch
from chimera.training import load_checkpoint


def measure(model, *, seed, batches, batch_size):
    """Recherche texte isolé vers image et série ; diagnostic de dépendance du texte."""
    model.eval()
    n = text_series = text_image = image_series = correct = observations = 0
    absolute = leakage = 0.
    with torch.inference_mode():
        for index in range(batches):
            batch = synthetic_batch(model.config, batch_size=batch_size, seed=seed+index)
            labels = batch['class_labels']
            visual, _ = model.vision(batch['images'])
            temporal, valid, *_ = model.temporal(batch['series'], batch['series_mask'])
            pooled = (temporal * valid.unsqueeze(-1)).sum(1) / valid.sum(1,keepdim=True).clamp_min(1)
            text_hidden = model._isolated_text_representation(batch['input_ids'],batch['attention_mask'].bool())
            text = F.normalize(model.alignment[2](text_hidden).float(),dim=-1)
            image = F.normalize(model.alignment[0](visual.mean(1)).float(),dim=-1)
            series = F.normalize(model.alignment[1](pooled).float(),dim=-1)
            for source, target, name in ((text,series,'text_series'),(text,image,'text_image'),
                                         (image,series,'image_series')):
                selected = (source @ target.T).argmax(-1)
                hits = (labels[selected] == labels).sum().item()
                if name == 'text_series': text_series += hits
                elif name == 'text_image': text_image += hits
                else: image_series += hits
            captured = []
            handle = model.alignment[2].register_forward_hook(lambda module, inputs, output: captured.append(output.clone()))
            try:
                inputs = {key:batch[key] for key in ('input_ids','attention_mask','images','series','series_mask')}
                model(**inputs)
                shifted = dict(inputs,images=inputs['images'].roll(1,0),series=inputs['series'].roll(1,0),
                               series_mask=inputs['series_mask'].roll(1,0))
                model(**shifted)
            finally:
                handle.remove()
            leakage += (captured[0]-captured[1]).norm(dim=-1).sum().item()
            supervised = {key:batch[key] for key in ('images','series','series_mask','class_labels','forecast_targets')}
            output = model(**supervised)
            correct += (output.classification_logits.argmax(-1) == labels).sum().item()
            predicted = output.forecast[..., model.config.quantiles.index(.5)]
            absolute += (predicted - batch['forecast_targets']).abs().sum().item()
            observations += predicted.numel()
            n += labels.numel()
    return dict(examples=n, text_to_series_class=text_series/n, text_to_image_class=text_image/n,
                image_to_series_class=image_series/n, accuracy=correct/n, forecast_mae=absolute/observations,
                text_view_sensitivity_l2=leakage/n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True, help='Nouveau dossier, créé par la commande')
    parser.add_argument('--seeds', type=int, nargs='+', default=[111000000,113000000,115000000])
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--test-seed', type=int, default=120000000)
    parser.add_argument('--test-batches', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--train-threads', type=int, default=1)
    parser.add_argument('--eval-threads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--alignment-weight', type=float, default=.2)
    args = parser.parse_args()
    if (min(args.steps, args.test_batches, args.train_threads, args.eval_threads) < 1 or args.test_batch_size < 2
            or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.alignment_weight)
            or args.alignment_weight < 0):
        parser.error('Tailles, taux et poids invalides')
    ranges = [(seed, 2*args.steps) for seed in args.seeds]
    ranges += [(seed+1000000, 1) for seed in args.seeds]
    ranges.append((args.test_seed, args.test_batches))
    if (any(start < 0 or start + length >= 2**63 for start,length in ranges)
            or any(max(a,b) < min(a+n,b+m) for i,(a,n) in enumerate(ranges) for b,m in ranges[i+1:])):
        parser.error('Les plages de seeds de train, validation et test doivent être disjointes')
    checkpoint = Path(args.checkpoint).resolve()
    parent_sha = file_sha256(checkpoint)
    output = Path(args.output_dir).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('output-dir doit être un nouveau dossier')
    env = dict(os.environ, OMP_NUM_THREADS=str(args.train_threads))
    for name in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(name,None)
    torch.set_num_threads(args.eval_threads)
    runs = []
    for seed in args.seeds:
        for scope in ('joint','isolated'):
            if file_sha256(checkpoint) != parent_sha:
                raise ValueError('Le checkpoint parent a changé pendant la comparaison')
            overrides = output / f'{scope}-{seed}.json'
            overrides.write_text(json.dumps(dict(alignment_scope='global', alignment_text_view=scope, alignment_mode='class',
                                                  alignment_weight=args.alignment_weight))+'\n')
            directory = output / f'{scope}-{seed}'
            command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                       '-m','chimera','--threads',str(args.train_threads),'train','--init-from',str(checkpoint),
                       '--model-overrides',str(overrides),'--steps',str(args.steps),'--batch-size','1',
                       '--modality-dropout','0','--seed',str(seed),'--warmup',str(min(10,args.steps)),
                       '--lr',str(args.lr),'--eval-every',str(args.steps),'--save-every',str(args.steps),
                       '--output',str(directory)]
            print(json.dumps(dict(event='training', seed=seed, scope=scope)),flush=True)
            log = output / f'{scope}-{seed}.log'
            with log.open('w',encoding='utf-8') as stream:
                process = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"Échec de l'entraînement ({scope}, {seed}), voir {log}")
            if file_sha256(checkpoint) != parent_sha:
                raise ValueError('Le checkpoint parent a changé pendant la comparaison')
            trained_path = directory/'latest.pt'
            model, payload = load_checkpoint(trained_path)
            metrics = measure(model,seed=args.test_seed,batches=args.test_batches,batch_size=args.test_batch_size)
            rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            alignment_steps = sum('alignment' in row for row in rows)
            runs.append(dict(seed=seed, scope=scope, checkpoint_sha256=file_sha256(trained_path),
                             step=payload['step'], alignment_steps=alignment_steps, metrics=metrics))
            print(json.dumps(dict(event='evaluated',seed=seed,scope=scope,**metrics)),flush=True)
    metrics = ('text_to_series_class','text_to_image_class','image_to_series_class',
               'accuracy','forecast_mae','text_view_sensitivity_l2')
    aggregate = {}
    for metric in metrics:
        local = [row['metrics'][metric] for row in runs if row['scope']=='joint']
        global_ = [row['metrics'][metric] for row in runs if row['scope']=='isolated']
        differences = [b-a for a,b in zip(local,global_)]
        aggregate[metric] = dict(joint_mean=statistics.mean(local), isolated_mean=statistics.mean(global_),
                                 paired_difference_mean=statistics.mean(differences),
                                 paired_difference_std=(statistics.stdev(differences) if len(differences)>1 else None))
    report = dict(settings=vars(args), parent_sha256=parent_sha, torch_version=str(torch.__version__),
                  runs=runs, aggregate=aggregate,
                  scope='Deux processus CPU et données synthétiques ; le texte contient la classe. Aucune généralisation réelle n\'est établie.')
    with prediction_stream(output/'comparison.json') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='complete',report=str(output/'comparison.json'))),flush=True)


if __name__ == '__main__':
    main()
