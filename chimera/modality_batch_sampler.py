"""Lots homogènes de modalités pour des jeux JSONL hétérogènes."""
from array import array
import torch
from torch.utils.data import Sampler


class ModalityBatchSampler(Sampler):
    """Former les lots par signature, puis les répartir à parts égales entre rangs."""

    def __init__(self, dataset, batch_size, *, num_replicas=1, rank=0, seed=0, shuffle=True,
                 shared_across_ranks=False, shared_interval=1):
        if (type(batch_size) is not int or batch_size < 1 or type(num_replicas) is not int or num_replicas < 1
                or type(rank) is not int or not 0 <= rank < num_replicas
                or type(shared_across_ranks) is not bool
                or type(shared_interval) is not int or shared_interval < 1
                or not shared_across_ranks and shared_interval != 1):
            raise ValueError('Taille de lot ou rang distribué invalide')
        if dataset.signatures is None or len(dataset.signatures) != len(dataset):
            raise ValueError('Le jeu doit être ouvert avec allow_mixed=True')
        self.batch_size, self.num_replicas, self.rank = batch_size, num_replicas, rank
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        self.shared_across_ranks = shared_across_ranks
        self.shared_interval = shared_interval
        self.groups = [array('Q') for _ in range(8)]
        for index, signature in enumerate(dataset.signatures):
            if not 1 <= signature <= 7:
                raise ValueError('Signature de modalités invalide')
            self.groups[signature].append(index)
        self.global_batches = sum((len(group) + batch_size - 1) // batch_size for group in self.groups)
        if not self.global_batches:
            raise ValueError('Le jeu ne contient aucun exemple')
        if shared_across_ranks:
            cycle_size = 1 + (shared_interval - 1) * num_replicas
            cycles, remaining = divmod(self.global_batches, cycle_size)
            self.batches_per_rank = cycles * shared_interval
            if remaining:
                self.batches_per_rank += 1 + (remaining - 1 + num_replicas - 1) // num_replicas
        else:
            self.batches_per_rank = (self.global_batches + num_replicas - 1) // num_replicas

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('epoch doit être un entier positif ou nul')
        self.epoch = epoch

    def __len__(self):
        return self.batches_per_rank

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for group in self.groups:
            if not group:
                continue
            order = torch.randperm(len(group), generator=generator).tolist() if self.shuffle else range(len(group))
            for start in range(0, len(group), self.batch_size):
                batches.append([group[position] for position in order[start:start + self.batch_size]])
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]
        if self.shared_across_ranks:
            cursor, step = 0, 0
            while cursor < self.global_batches:
                if step % self.shared_interval == 0:
                    yield batches[cursor]
                    cursor += 1
                else:
                    yield batches[(cursor + self.rank) % self.global_batches]
                    cursor += self.num_replicas
                step += 1
            return
        missing = self.batches_per_rank * self.num_replicas - len(batches)
        if missing:
            batches.extend(batches[index % self.global_batches] for index in range(missing))
        yield from batches[self.rank::self.num_replicas]


class OrderedModalityBatchSampler(Sampler):
    """Regrouper uniquement les lignes consécutives, sans réordonner les résultats."""

    def __init__(self, dataset, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Taille de lot invalide')
        if dataset.signatures is None or len(dataset.signatures) != len(dataset):
            raise ValueError('Le jeu doit être ouvert avec allow_mixed=True')
        self.signatures, self.batch_size = dataset.signatures, batch_size
        self.count = sum(1 for _ in self._batches())

    def _batches(self):
        start = 0
        for index in range(1, len(self.signatures) + 1):
            if (index == len(self.signatures) or index - start == self.batch_size
                    or self.signatures[index] != self.signatures[start]):
                yield range(start, index)
                start = index

    def __iter__(self):
        return self._batches()

    def __len__(self):
        return self.count
