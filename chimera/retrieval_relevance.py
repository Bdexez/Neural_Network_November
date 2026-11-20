"""Construire des jugements de recherche à partir de classes explicites."""
from array import array
import json
from pathlib import Path

from .calibration import file_sha256
from .data import JointDataset, file_identity
from .embedding_search import EmbeddingFile


_MODALITY_BITS = {'text': 1, 'image': 2, 'series': 4}


def _label(dataset, index):
    record = dataset.read_record(index)
    value = record.get('class_label', -100)
    if type(value) is not int or (value < 0 and value != -100):
        raise ValueError(f'{dataset.path}:{dataset.line_numbers[index]}: class_label invalide')
    return None if value == -100 else value


def _validate_source(dataset, embeddings, modality, *, batch_size):
    if dataset.signatures is None or len(dataset.signatures) != len(dataset):
        raise ValueError('Le jeu doit être ouvert avec allow_mixed=True')
    for _ in embeddings.batches(batch_size):
        pass
    if ((dataset.fingerprint, dataset.image_fingerprint) !=
            (embeddings.metadata['dataset_sha256'], embeddings.metadata['dataset_images_sha256'])
            or len(dataset) != embeddings.total_rows):
        raise ValueError(f'Le jeu {dataset.path} ne correspond pas aux vecteurs {embeddings.path}')
    bit = _MODALITY_BITS[modality]
    if any(bool(signature & bit) != bool(present)
           for signature, present in zip(dataset.signatures, embeddings.presence)):
        raise ValueError(f'Modalités du jeu et des vecteurs incohérentes : {embeddings.path}')


def _verify_unchanged(dataset, embeddings):
    current = JointDataset(dataset.path, None, allow_mixed=True)
    if (current.fingerprint, current.image_fingerprint) != (dataset.fingerprint, dataset.image_fingerprint):
        raise ValueError(f'Le jeu ou ses images ont changé : {dataset.path}')
    path = Path(embeddings.path)
    if (file_identity(path.stat()) != embeddings.identity or file_sha256(path) != embeddings.sha256
            or file_identity(path.stat()) != embeddings.identity):
        raise ValueError(f'Fichier de vecteurs modifié : {path}')


def write_class_relevance(query_dataset, candidate_dataset, query_path, candidate_path, query_modality,
                          candidate_modality, stream, *, exclude_self=False, batch_size=1024):
    """Écrire une liste exhaustive des candidats de même classe, dans l'ordre source."""
    if query_modality not in _MODALITY_BITS or candidate_modality not in _MODALITY_BITS:
        raise ValueError('Modalité inconnue')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size doit être un entier positif')
    queries = EmbeddingFile(query_path, query_modality, skip_missing=True, track_presence=True)
    candidates = EmbeddingFile(candidate_path, candidate_modality, skip_missing=True, track_presence=True)
    _validate_source(query_dataset, queries, query_modality, batch_size=batch_size)
    _validate_source(candidate_dataset, candidates, candidate_modality, batch_size=batch_size)
    for field in ('checkpoint_sha256', 'amp'):
        if queries.metadata[field] != candidates.metadata[field]:
            raise ValueError(f'Fichiers de vecteurs incompatibles : {field}')
    if queries.dim != candidates.dim:
        raise ValueError('Dimensions des vecteurs incompatibles')
    same_data = ((query_dataset.fingerprint, query_dataset.image_fingerprint) ==
                 (candidate_dataset.fingerprint, candidate_dataset.image_fingerprint))
    by_class = {}
    for index in range(len(candidate_dataset)):
        label = _label(candidate_dataset, index)
        if candidates.presence[index]:
            if label is None:
                raise ValueError(f'{candidate_dataset.path}:{candidate_dataset.line_numbers[index]}: '
                                 'un candidat disponible doit avoir une classe connue')
            by_class.setdefault(label, array('Q')).append(index)
    header = dict(format_version=1, query_dataset_sha256=query_dataset.fingerprint,
                  query_dataset_images_sha256=query_dataset.image_fingerprint,
                  candidate_dataset_sha256=candidate_dataset.fingerprint,
                  candidate_dataset_images_sha256=candidate_dataset.image_fingerprint,
                  class_label_relevance=dict(query_modality=query_modality, candidate_modality=candidate_modality,
                                             exclude_self=exclude_self))
    stream.write(json.dumps(header, ensure_ascii=False, allow_nan=False) + '\n')
    judged = available = 0
    for index in range(len(query_dataset)):
        label = _label(query_dataset, index)
        present = bool(queries.presence[index])
        available += present
        matches = by_class.get(label, ()) if present and label is not None else ()
        relevant = [candidate for candidate in matches if not (exclude_self and same_data and candidate == index)]
        judged += bool(relevant)
        stream.write(json.dumps(dict(index=index, relevant=relevant), allow_nan=False) + '\n')
    if not judged:
        raise ValueError('Aucune requête ne possède de candidat pertinent')
    _verify_unchanged(query_dataset, queries)
    _verify_unchanged(candidate_dataset, candidates)
    return dict(queries=len(query_dataset), available_queries=available, judged_queries=judged,
                candidate_examples=len(candidate_dataset), labeled_candidates=sum(map(len, by_class.values())))
