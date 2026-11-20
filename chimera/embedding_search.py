"""Recherche cosinus exacte sur les fichiers produits par ``embed``."""
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .calibration import file_sha256
from .data import file_identity


class EmbeddingFile:
    """Lire par blocs et vérifier chaque parcours, sans charger le fichier entier."""

    fields = ('format_version', 'checkpoint_sha256', 'dataset_sha256', 'dataset_images_sha256',
              'alignment_text_view', 'amp', 'device', 'batch_size', 'torch_version')

    def __init__(self, path, modality, *, skip_missing=False, track_presence=False):
        self.path = Path(path)
        self.modality = modality
        self.skip_missing = skip_missing
        self.track_presence = track_presence
        self.presence = None
        self.total_rows = None
        self.identity = file_identity(self.path.stat())
        self.sha256 = file_sha256(self.path)
        if file_identity(self.path.stat()) != self.identity:
            raise ValueError(f'Fichier de vecteurs modifié : {self.path}')
        self.metadata = None
        self.dim = None

    def batches(self, size):
        digest = hashlib.sha256()
        vectors, indexes, count, selected = [], [], 0, 0
        presence = bytearray() if self.track_presence else None
        with self.path.open('rb') as stream:
            if file_identity(os.fstat(stream.fileno())) != self.identity:
                raise ValueError(f'Fichier de vecteurs modifié : {self.path}')
            for line in stream:
                digest.update(line)
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('objet JSON attendu')
                    metadata = {field: row[field] for field in self.fields}
                    if type(row['index']) is not int or row['index'] != count:
                        raise ValueError('indices non consécutifs')
                    if type(metadata['format_version']) is not int or metadata['format_version'] != 1:
                        raise ValueError('version inconnue')
                    for field in ('checkpoint_sha256', 'dataset_sha256'):
                        value = metadata[field]
                        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                            raise ValueError(f'{field} invalide')
                    image_sha = metadata['dataset_images_sha256']
                    if image_sha is not None and (not isinstance(image_sha, str) or len(image_sha) != 64
                                                  or any(c not in '0123456789abcdef' for c in image_sha)):
                        raise ValueError('dataset_images_sha256 invalide')
                    if metadata['alignment_text_view'] not in ('joint', 'isolated') or metadata['amp'] not in ('none', 'bf16', 'fp16'):
                        raise ValueError('configuration inconnue')
                    if (type(metadata['batch_size']) is not int or metadata['batch_size'] < 1
                            or not isinstance(metadata['device'], str) or not isinstance(metadata['torch_version'], str)):
                        raise ValueError('métadonnées invalides')
                    if self.metadata is None:
                        self.metadata = metadata
                    elif self.metadata != metadata:
                        raise ValueError('métadonnées incohérentes')
                    modalities = row['vectors']
                    if (not isinstance(modalities, dict) or not modalities
                            or not any(name in modalities for name in ('text', 'image', 'series'))):
                        raise ValueError('modalités de vecteurs invalides')
                    present = self.modality in modalities
                    if not present and not self.skip_missing:
                        raise ValueError(f'modalité {self.modality} absente')
                    if present:
                        values = modalities[self.modality]
                        if (not isinstance(values, list) or not values or
                                any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
                            raise ValueError('vecteur non numérique ou non fini')
                        if self.dim is None:
                            self.dim = len(values)
                        elif self.dim != len(values):
                            raise ValueError('dimensions incohérentes')
                        vector = torch.tensor(values, dtype=torch.float64)
                        scale = vector.abs().max()
                        if not torch.isfinite(scale) or scale == 0:
                            raise ValueError('vecteur nul ou hors limites')
                        vector = vector / scale
                        vector = vector / vector.norm()
                except (ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(f'{self.path}:{count + 1}: vecteur invalide : {exc}') from exc
                if presence is not None:
                    presence.append(int(present))
                if present:
                    vectors.append(vector)
                    indexes.append(count)
                    selected += 1
                count += 1
                if len(vectors) == size:
                    yield torch.stack(vectors), torch.tensor(indexes, dtype=torch.long)
                    vectors, indexes = [], []
            if (file_identity(os.fstat(stream.fileno())) != self.identity
                    or file_identity(self.path.stat()) != self.identity or digest.hexdigest() != self.sha256):
                raise ValueError(f'Fichier de vecteurs modifié : {self.path}')
        if count == 0:
            raise ValueError(f'Fichier de vecteurs vide : {self.path}')
        if selected == 0:
            raise ValueError(f'Aucun vecteur {self.modality} dans {self.path}')
        self.total_rows, self.presence = count, presence
        if vectors:
            yield torch.stack(vectors), torch.tensor(indexes, dtype=torch.long)


def search_embeddings(query_path, candidate_path, query_modality, candidate_modality, *, top_k=5,
                      query_batch_size=16, candidate_batch_size=1024, exclude_self=False, skip_missing=False):
    """Produire les voisins exacts, triés par score décroissant puis indice croissant."""
    if min(top_k, query_batch_size, candidate_batch_size) < 1:
        raise ValueError('top_k et les tailles de lot doivent être positifs')
    queries = EmbeddingFile(query_path, query_modality, skip_missing=skip_missing)
    candidates = EmbeddingFile(candidate_path, candidate_modality, skip_missing=skip_missing)
    for query_vectors, query_indices in queries.batches(query_batch_size):
        best_scores = torch.empty((len(query_indices), 0), dtype=torch.float64)
        best_indices = torch.empty((len(query_indices), 0), dtype=torch.long)
        for candidate_vectors, candidate_indices in candidates.batches(candidate_batch_size):
            for field in ('checkpoint_sha256', 'amp'):
                if queries.metadata[field] != candidates.metadata[field]:
                    raise ValueError(f'Fichiers de vecteurs incompatibles : {field}')
            if queries.dim != candidates.dim:
                raise ValueError('Dimensions des vecteurs incompatibles')
            scores = (query_vectors @ candidate_vectors.T).clamp(-1, 1)
            same_data = all(queries.metadata[key] == candidates.metadata[key]
                            for key in ('dataset_sha256', 'dataset_images_sha256'))
            if exclude_self and same_data:
                scores.masked_fill_(query_indices[:, None] == candidate_indices[None, :], -torch.inf)
            indices = candidate_indices.expand(len(query_indices), -1)
            scores = torch.cat((best_scores, scores), dim=1)
            indices = torch.cat((best_indices, indices), dim=1)
            # Pré-tri par indice, puis tri stable par score pour départager les égalités.
            order = indices.argsort(dim=1, stable=True)
            scores, indices = scores.gather(1, order), indices.gather(1, order)
            order = scores.argsort(dim=1, descending=True, stable=True)[:, :top_k]
            best_scores, best_indices = scores.gather(1, order), indices.gather(1, order)
        for i, index in enumerate(query_indices.tolist()):
            matches = [{'index': int(j), 'cosine': float(score)}
                       for j, score in zip(best_indices[i], best_scores[i]) if torch.isfinite(score)]
            yield {'index': index, 'query_modality': query_modality, 'candidate_modality': candidate_modality,
                   'query_embeddings_sha256': queries.sha256, 'candidate_embeddings_sha256': candidates.sha256,
                   'matches': matches}
