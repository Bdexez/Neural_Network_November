"""Mesures de recherche sur des pertinences explicites et des vecteurs figés."""
import hashlib
import json
import math
import os
from pathlib import Path

from .calibration import file_sha256
from .data import file_identity
from .embedding_search import EmbeddingFile, search_embeddings


def evaluate_retrieval(query_path, candidate_path, relevance_path, query_modality, candidate_modality, *,
                       top_k=5, query_batch_size=16, candidate_batch_size=1024, exclude_self=False,
                       skip_missing=False):
    """Évaluer en macro-moyenne les requêtes comportant au moins un candidat pertinent."""
    if min(top_k, query_batch_size, candidate_batch_size) < 1:
        raise ValueError('top_k et les tailles de lot doivent être positifs')
    queries = EmbeddingFile(query_path, query_modality, skip_missing=skip_missing, track_presence=True)
    candidates = EmbeddingFile(candidate_path, candidate_modality, skip_missing=skip_missing, track_presence=True)
    query_count = sum(len(indices) for _, indices in queries.batches(query_batch_size))
    candidate_count = sum(len(indices) for _, indices in candidates.batches(candidate_batch_size))
    for field in ('checkpoint_sha256', 'amp'):
        if queries.metadata[field] != candidates.metadata[field]:
            raise ValueError(f'Fichiers de vecteurs incompatibles : {field}')
    if queries.dim != candidates.dim:
        raise ValueError('Dimensions des vecteurs incompatibles')
    same_data = all(queries.metadata[key] == candidates.metadata[key]
                    for key in ('dataset_sha256', 'dataset_images_sha256'))
    overlap = exclude_self and same_data and any(q and c for q, c in zip(queries.presence, candidates.presence))
    if top_k > candidate_count - int(overlap):
        raise ValueError('top_k dépasse le nombre de candidats admissibles')

    path = Path(relevance_path)
    identity = file_identity(path.stat())
    relevance_sha = file_sha256(path)
    if file_identity(path.stat()) != identity:
        raise ValueError(f'Fichier de pertinence modifié : {path}')
    digest = hashlib.sha256()
    totals = dict(hit=0.0, recall=0.0, precision=0.0, reciprocal_rank=0.0, ndcg=0.0)
    judged = 0
    ranking = iter(search_embeddings(query_path, candidate_path, query_modality, candidate_modality,
                                     top_k=top_k, query_batch_size=query_batch_size,
                                     candidate_batch_size=candidate_batch_size, exclude_self=exclude_self,
                                     skip_missing=skip_missing))
    with path.open('rb') as stream:
        if file_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError(f'Fichier de pertinence modifié : {path}')
        header_line = stream.readline()
        digest.update(header_line)
        try:
            header = json.loads(header_line)
            if not isinstance(header, dict) or type(header.get('format_version')) is not int or header['format_version'] != 1:
                raise ValueError('version de pertinence inconnue')
            for prefix, source in (('query', queries), ('candidate', candidates)):
                if (header[f'{prefix}_dataset_sha256'] != source.metadata['dataset_sha256'] or
                        header[f'{prefix}_dataset_images_sha256'] != source.metadata['dataset_images_sha256']):
                    raise ValueError(f'empreinte du jeu {prefix} incompatible')
        except (KeyError, ValueError, TypeError, UnicodeDecodeError) as exc:
            raise ValueError(f'{path}:1: en-tête de pertinence invalide : {exc}') from exc
        for index in range(queries.total_rows):
            row = next(ranking, None) if queries.presence[index] else None
            if queries.presence[index] and (row is None or row['index'] != index):
                raise ValueError('Les requêtes de recherche ne correspondent pas aux vecteurs')
            if row is not None and (row['query_embeddings_sha256'] != queries.sha256
                                    or row['candidate_embeddings_sha256'] != candidates.sha256):
                raise ValueError('Les fichiers de vecteurs ont changé pendant l’évaluation')
            line = stream.readline()
            digest.update(line)
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or type(item.get('index')) is not int or item['index'] != index:
                    raise ValueError('indice manquant ou non consécutif')
                relevant = item['relevant']
                if (not isinstance(relevant, list) or
                        any(type(value) is not int or value < 0 or value >= candidates.total_rows for value in relevant) or
                        len(set(relevant)) != len(relevant)):
                    raise ValueError('indices pertinents invalides ou répétés')
                if row is not None and any(not candidates.presence[value] for value in relevant):
                    raise ValueError('un candidat pertinent ne possède pas la modalité choisie')
                if row is not None and exclude_self and same_data and index in relevant:
                    raise ValueError('une paire pertinente est exclue par exclude-self')
            except (KeyError, ValueError, TypeError, UnicodeDecodeError) as exc:
                raise ValueError(f'{path}:{index + 2}: pertinence invalide : {exc}') from exc
            if row is None or not relevant:
                continue
            judged += 1
            positive = set(relevant)
            positions = [rank for rank, match in enumerate(row['matches'], start=1) if match['index'] in positive]
            count = len(positions)
            totals['hit'] += bool(count)
            totals['recall'] += count / len(positive)
            totals['precision'] += count / top_k
            if positions:
                totals['reciprocal_rank'] += 1 / positions[0]
            dcg = sum(1 / math.log2(rank + 1) for rank in positions)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(positive), top_k) + 1))
            totals['ndcg'] += dcg / ideal
        if next(ranking, None) is not None:
            raise ValueError('Les requêtes de recherche ne correspondent pas aux vecteurs')
        if stream.read(1):
            raise ValueError(f'{path}: lignes de pertinence supplémentaires')
        if (file_identity(os.fstat(stream.fileno())) != identity or file_identity(path.stat()) != identity
                or digest.hexdigest() != relevance_sha):
            raise ValueError(f'Fichier de pertinence modifié : {path}')
    if judged == 0:
        raise ValueError('Aucune requête avec candidat pertinent')
    return dict(format_version=1, queries=query_count, queries_total=queries.total_rows,
                queries_without_modality=queries.total_rows - query_count,
                judged_queries=judged, unjudged_queries=query_count - judged,
                candidates=candidate_count, candidates_total=candidates.total_rows,
                candidates_without_modality=candidates.total_rows - candidate_count,
                top_k=top_k, exclude_self=exclude_self, skip_missing=skip_missing,
                query_modality=query_modality, candidate_modality=candidate_modality,
                query_embeddings_sha256=queries.sha256, candidate_embeddings_sha256=candidates.sha256,
                query_dataset_sha256=queries.metadata['dataset_sha256'],
                query_dataset_images_sha256=queries.metadata['dataset_images_sha256'],
                candidate_dataset_sha256=candidates.metadata['dataset_sha256'],
                candidate_dataset_images_sha256=candidates.metadata['dataset_images_sha256'],
                relevance_sha256=relevance_sha, checkpoint_sha256=queries.metadata['checkpoint_sha256'],
                amp=queries.metadata['amp'], metrics={name: value / judged for name, value in totals.items()})
