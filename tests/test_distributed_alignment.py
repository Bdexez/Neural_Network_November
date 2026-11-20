import copy
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
from torch import distributed as dist

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.distributed_alignment import distributed_alignment_loss
from chimera.training import DistributedTrainingModel
from chimera.losses import symmetric_contrastive_loss


LEFT = torch.tensor([[.8,.1,.3], [.2,.9,.4], [.1,.3,.9]], dtype=torch.float32)
RIGHT = torch.tensor([[.7,.2,.4], [.3,.8,.1], [.2,.2,.8]], dtype=torch.float32)
LABELS = torch.tensor([0, 0, 1], dtype=torch.long)


def alignment_worker(rank, init_file, directory, scenario):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        if scenario in ('paired','class','class_unknown'):
            rows = slice(0, 2) if rank == 0 else slice(2, 3)
            left = LEFT[rows].clone().requires_grad_()
            right = RIGHT[rows].clone().requires_grad_()
            vectors = {0:left,1:right}
            labels = LABELS[rows].clone()
            if scenario == 'class_unknown' and rank == 1:
                labels[:] = -100
        elif scenario == 'mixed':
            if rank == 0:
                left, right = LEFT[:2].clone().requires_grad_(), RIGHT[:2].clone().requires_grad_()
                vectors, labels = {0:left,1:right}, LABELS[:2]
            else:
                left = LEFT[2:].clone().requires_grad_()
                right = None
                vectors, labels = {2:left}, None
        elif scenario == 'single_pair':
            left = LEFT[rank:rank+1].clone().requires_grad_()
            right = RIGHT[rank:rank+1].clone().requires_grad_() if rank == 0 else None
            vectors, labels = ({0:left,1:right} if rank == 0 else {1:left}), None
        else:
            left = LEFT[rank:rank+1].clone().requires_grad_(rank == 0)
            right = RIGHT[rank:rank+1].clone().requires_grad_(rank == 0)
            vectors, labels = {0:left,1:right}, None
        mode = 'class' if scenario in ('class','class_unknown') else 'paired'
        if scenario == 'mismatched_mode':
            mode = 'class' if rank == 1 else 'paired'
        if scenario == 'mismatched_dimension' and rank == 1:
            vectors = {key:value[:,:2] for key,value in vectors.items()}
        if scenario == 'mismatched_dtype' and rank == 1:
            vectors = {key:value.double() for key,value in vectors.items()}
        if scenario == 'mismatched_temperature':
            temperature = .4 if rank == 1 else .3
        else:
            temperature = .3
        try:
            loss = distributed_alignment_loss(vectors, labels, temperature, mode, group=dist.group.WORLD)
        except ValueError as exc:
            torch.save(dict(error=str(exc)), Path(directory) / f'{rank}.pt')
            return
        loss.backward()
        result = dict(loss=loss.detach(), left=left.grad, right=right.grad if right is not None else None)
        torch.save(result, Path(directory) / f'{rank}.pt')
    finally:
        dist.destroy_process_group()


def model_config(mode='paired', scope='global'):
    return replace(ChimeraConfig.preset('tiny'), alignment_mode=mode, alignment_scope=scope,
                   alignment_weight=1., balance_weight=0., router_z_weight=0., mtp_weight=0.,
                   text_weight=0., classification_weight=0., forecast_weight=0., dropout=0.)


def model_batch(config, n=3):
    batch = synthetic_batch(config, batch_size=n, seed=710)
    batch = {key: value for key, value in batch.items() if key not in ('labels', 'forecast_targets')}
    batch['class_labels'] = torch.tensor(([0,0,1,2])[:n])
    return batch


def model_worker(rank, init_file, directory, scenario, mode):
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        config = model_config(mode)
        torch.manual_seed(712)
        model = Chimera(config).train()
        wrapped = DistributedDataParallel(DistributedTrainingModel(model), find_unused_parameters=True)
        batch = model_batch(config, n=4 if scenario == 'accumulate' else 3)
        if scenario == 'accumulate':
            sections = (slice(rank,rank+1),slice(rank+2,rank+3))
        elif scenario == 'single_pair':
            sections = (slice(rank,rank+1),)
        else:
            sections = (slice(0,2),) if rank == 0 else (slice(2,3),)
        seen = []
        for i, section in enumerate(sections):
            part = {key:value[section] for key,value in batch.items()}
            if scenario in ('mixed','single_pair'):
                for key in ('input_ids','attention_mask'):
                    part.pop(key)
                if rank == 1:
                    part.pop('images')
            with wrapped.no_sync() if scenario == 'accumulate' and i == 0 else nullcontext():
                loss, terms, _, _ = wrapped(**part)
                seen.append(terms['alignment'].item())
                (loss / len(sections)).backward()
        names = ('embedding.weight','alignment.0.weight','alignment.1.weight','alignment.2.weight',
                 'vision.stem.weight','temporal.proj.weight')
        params = dict(model.named_parameters())
        grads = {name:params[name].grad for name in names}
        # Une validation sur le seul rang 0 doit rester locale et ne pas bloquer.
        copied = None
        if rank == 0:
            model.eval()
            with torch.no_grad():
                model(**{key:value[:1] for key,value in batch.items()})
            try:
                cloned = copy.deepcopy(model)
                copied = cloned._alignment_group is None
            except Exception:
                copied = False
        dist.barrier()
        torch.save(dict(alignment=seen, gradients=grads, copied=copied), Path(directory)/f'model{rank}.pt')
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
class DistributedAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_workers(self, scenario):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(alignment_worker,
                                        args=(str(Path(directory)/'group'), directory, scenario),
                                        nprocs=2, join=True)
            return [torch.load(Path(directory)/f'{rank}.pt', weights_only=True) for rank in range(2)]

    def test_global_loss_and_gradients_match_full_batch(self):
        for mode in ('paired','class'):
            with self.subTest(mode=mode):
                left, right = LEFT.clone().requires_grad_(), RIGHT.clone().requires_grad_()
                expected = symmetric_contrastive_loss(left @ right.T / .3, LABELS if mode == 'class' else None)
                expected.backward()
                result = self.run_workers(mode)
                for rank, section in ((0,slice(0,2)), (1,slice(2,3))):
                    torch.testing.assert_close(result[rank]['loss'], expected.detach(), rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(result[rank]['left'] / 2, left.grad[section], rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(result[rank]['right'] / 2, right.grad[section], rtol=1e-6, atol=1e-7)

    def test_ranks_reject_mismatched_configuration_before_shape_collectives(self):
        for scenario in ('mismatched_mode','mismatched_dimension','mismatched_dtype','mismatched_temperature'):
            with self.subTest(scenario=scenario):
                for result in self.run_workers(scenario):
                    self.assertIn('différents selon les rangs',result['error'])

    def test_class_mode_keeps_unknown_labels_separate_across_ranks(self):
        left, right = LEFT.clone().requires_grad_(), RIGHT.clone().requires_grad_()
        labels = torch.tensor([0,0,-100])
        expected = symmetric_contrastive_loss(left @ right.T / .3, labels)
        expected.backward()
        results = self.run_workers('class_unknown')
        torch.testing.assert_close(results[1]['loss'], expected.detach())
        torch.testing.assert_close(results[0]['left']/2,left.grad[:2])
        torch.testing.assert_close(results[1]['right']/2,right.grad[2:])

    def test_single_process_global_mode_preserves_local_behavior(self):
        for mode in ('paired','class'):
            config = model_config(mode,scope='local')
            torch.manual_seed(714)
            local = Chimera(config).train()
            global_model = Chimera(replace(config, alignment_scope='global')).train()
            global_model.load_state_dict(local.state_dict())
            batch = model_batch(config)
            left, right = local(**batch), global_model(**batch)
            torch.testing.assert_close(left.loss,right.loss,rtol=0,atol=0)
            self.assertEqual(global_model._alignment_group,None)
        old = config.to_dict()
        del old['alignment_scope']
        self.assertEqual(ChimeraConfig.from_dict(old).alignment_scope,'local')
        with self.assertRaises(ValueError):
            replace(config,alignment_scope='invalid')

    def test_incomplete_modalities_and_single_pair_participate_in_backward(self):
        left, right = LEFT[:2].clone().requires_grad_(), RIGHT[:2].clone().requires_grad_()
        expected = symmetric_contrastive_loss(left @ right.T / .3)
        expected.backward()
        result = self.run_workers('mixed')
        torch.testing.assert_close(result[0]['loss'],expected.detach())
        torch.testing.assert_close(result[1]['loss'],expected.detach())
        torch.testing.assert_close(result[0]['left']/2,left.grad)
        torch.testing.assert_close(result[0]['right']/2,right.grad)
        torch.testing.assert_close(result[1]['left'],torch.zeros_like(result[1]['left']))
        result = self.run_workers('single_pair')
        for entry in result:
            self.assertEqual(entry['loss'].item(),0.)
            self.assertEqual(entry['left'].abs().sum().item(),0.)

    def test_frozen_rank_still_participates_in_collective_backward(self):
        result = self.run_workers('frozen')
        self.assertTrue(torch.isfinite(result[0]['loss']))
        self.assertIsNotNone(result[0]['left'])
        self.assertIsNone(result[1]['left'])


    def model_results(self, scenario, mode='paired'):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(model_worker,
                args=(str(Path(directory)/'group'),directory,scenario,mode), nprocs=2, join=True)
            return [torch.load(Path(directory)/f'model{rank}.pt',weights_only=True) for rank in range(2)]

    def test_ddp_global_gradients_match_single_process_full_batch(self):
        for mode in ('paired','class'):
            with self.subTest(mode=mode):
                config = model_config(mode,scope='local')
                torch.manual_seed(712)
                reference = Chimera(config).train()
                expected = reference(**model_batch(config))
                expected.loss.backward()
                results = self.model_results('complete',mode)
                self.assertTrue(results[0]['copied'])
                for result in results:
                    self.assertAlmostEqual(result['alignment'][0],expected.losses['alignment'].item(),places=5)
                    for name, actual in result['gradients'].items():
                        wanted = dict(reference.named_parameters())[name].grad
                        self.assertIsNotNone(actual,name)
                        self.assertIsNotNone(wanted,name)
                        torch.testing.assert_close(actual,wanted,rtol=1e-3,atol=2e-6,msg=name)

    def test_ddp_mixed_modalities_and_no_global_pair(self):
        config = model_config(scope='local')
        torch.manual_seed(712)
        reference = Chimera(config).train()
        batch = {key:value[:2] for key,value in model_batch(config).items()
                 if key not in ('input_ids','attention_mask')}
        expected = reference(**batch)
        expected.loss.backward()
        results = self.model_results('mixed')
        for result in results:
            self.assertAlmostEqual(result['alignment'][0],expected.losses['alignment'].item(),places=5)
            for name in ('alignment.0.weight','alignment.1.weight','vision.stem.weight','temporal.proj.weight'):
                torch.testing.assert_close(result['gradients'][name],dict(reference.named_parameters())[name].grad,
                                           rtol=1e-3,atol=2e-6,msg=name)
        empty = self.model_results('single_pair')
        for result in empty:
            self.assertEqual(result['alignment'],[0.])
            for grad in result['gradients'].values():
                if grad is not None:
                    self.assertEqual(grad.count_nonzero().item(),0)

    def test_ddp_accumulation_compares_each_global_microbatch(self):
        config = model_config(scope='local')
        torch.manual_seed(712)
        reference = Chimera(config).train()
        batch = model_batch(config,n=4)
        for section in (slice(0,2),slice(2,4)):
            (reference(**{key:value[section] for key,value in batch.items()}).loss / 2).backward()
        results = self.model_results('accumulate')
        for result in results:
            for name, actual in result['gradients'].items():
                wanted = dict(reference.named_parameters())[name].grad
                torch.testing.assert_close(actual,wanted,rtol=1e-3,atol=2e-6,msg=name)


    def test_distributed_training_resume_with_global_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            replace(ChimeraConfig.preset('tiny'), alignment_scope='global', dropout=.1,
                    gradient_checkpointing=True).save(config)
            common = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                      '-m', 'chimera', '--threads', '1', 'train', '--config', str(config),
                      '--steps','3','--batch-size','1','--accumulation','2','--eval-every','2',
                      '--modality-dropout','0','--seed','510','--loss-normalization','observed']
            env = dict(os.environ, OMP_NUM_THREADS='1')
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
                env.pop(key,None)
            def run(*options):
                result = subprocess.run([*common,*map(str,options)],env=env,text=True,
                                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
                self.assertEqual(result.returncode,0,result.stdout[-6000:])
            full, resumed = root/'full',root/'resumed'
            run('--output',full)
            run('--output',resumed,'--stop-after','1')
            run('--output',resumed,'--resume',resumed/'latest.pt')
            expected = torch.load(full/'latest.pt',weights_only=True)
            actual = torch.load(resumed/'latest.pt',weights_only=True)
            self.assertEqual(actual['config']['alignment_scope'],'global')
            for name, value in expected['model'].items():
                torch.testing.assert_close(actual['model'][name],value,rtol=0,atol=0,msg=name)
            rows = [json.loads(row) for row in (full/'metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['alignment'] > 0 for row in rows))


if __name__ == '__main__':
    unittest.main()
