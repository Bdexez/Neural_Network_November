import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

from .augmentation import SeriesAugmentation
from .calibration import ForecastCalibration, file_sha256, fit_calibration
from .config import ChimeraConfig
from .data import JointDataset, collate_records, synthetic_batch
from .forecast_adapter import fit_forecast_adapter
from .embedding_search import search_embeddings
from .model import Chimera
from .modality_batch_sampler import OrderedModalityBatchSampler
from .retrieval_evaluation import evaluate_retrieval
from .retrieval_relevance import write_class_relevance
from .series_classifier import SeriesClassifier, fit_series_classifier
from .series_gate import SeriesGate, fit_series_gate
from .series_gate_guard import SeriesGateGuard, fit_series_gate_guard
from .series_probability import SeriesProbabilityCalibration, fit_series_probability, log_probabilities
from .series_set import SeriesPredictionSet, fit_series_set
from .split_audit import audit_splits
from .temporal_splits import prepare_temporal_splits
from .tokenizer import ByteTokenizer
from .training import evaluate, load_checkpoint, save_checkpoint, train


def same_file(first, second):
    """Comparer aussi les chemins existants qui partagent un lien physique."""
    if Path(first).resolve() == Path(second).resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False


def audit_series_calibration(data, dataset_sha256, references, *, output, subject="calibration"):
    """Comparer les historiques JSON exacts aux jeux d'ajustement fournis."""
    if not references:
        return
    for name, (path, digest) in references.items():
        if digest is None:
            raise ValueError(f"Empreinte du jeu {name} absente de l'artefact")
    paths = {subject: data, **{name: path for name, (path, _) in references.items()}}
    report = audit_splits(paths, fields=("series",), output=output, max_examples=1)
    if report["splits"][subject]["sha256"] != dataset_sha256:
        raise ValueError(f"Le jeu {subject} a changé pendant l'audit")
    for name, (_, digest) in references.items():
        if report["splits"][name]["sha256"] != digest:
            raise ValueError(f"Le jeu {name} ne correspond pas à l'empreinte de son artefact")
    for pair in report["pairs"]:
        if pair["shared_unique_records"]:
            raise ValueError(f"Historiques partagés entre {pair['left']} et {pair['right']} : "
                             f"{pair['shared_unique_records']} valeur(s) distincte(s)")


@contextmanager
def prediction_stream(path):
    if path is None:
        yield sys.stdout
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inspect_config(config):
    # Pas d'allocation des poids, même pour une architecture de centaines de milliards.
    with torch.device("meta"):
        model = Chimera(config)
    total = sum(p.numel() for p in model.parameters())
    expert_size = 3 * config.d_model * config.hidden_dim
    unselected = config.n_layers * (config.n_experts - config.top_k) * expert_size
    return {
        "parameters": total,
        "parameters_with_selected_decoder_experts_upper_bound": total - unselected,
        "decoder_layers": config.n_layers,
        "experts_per_layer": config.n_experts,
        "selected_experts_per_token": config.top_k,
        "shared_experts_per_layer": config.shared_experts,
        "weights_gib_fp32": round(total * 4 / 2**30, 3),
        "weights_gib_bf16": round(total * 2 / 2**30, 3),
        "adamw_gib_fp32_excluding_activations": round(total * 16 / 2**30, 3),
        "modules": {name: sum(p.numel() for p in module.parameters())
                    for name, module in model.named_children() if name != "lm_head"},
    }


def get_config(args):
    return ChimeraConfig.load(args.config) if args.config else ChimeraConfig.preset(args.preset)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHIMERA — réseau multimodal à experts")
    parser.add_argument("--threads", type=int, default=4, help="Threads CPU (défaut : 4)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Compter les paramètres sans allouer les poids")
    inspect.add_argument("--all", action="store_true", help="Inspecter tous les presets")
    audit = commands.add_parser("audit-splits", help="Détecter des exemples ou fenêtres partagés entre jeux JSONL")
    audit.add_argument("--split", action="append", required=True, metavar="NOM=FICHIER",
                       help="Jeu nommé ; indiquer au moins deux fois")
    audit.add_argument("--fields", nargs="+", help="Champs exacts à comparer ; tous obligatoires")
    audit.add_argument("--interval-fields", nargs=3, metavar=("GROUPE", "DEBUT", "FIN"),
                       help="Métadonnées des intervalles numériques [début, fin[")
    audit.add_argument("--max-examples", type=int, default=5, help="Nombre maximal de paires de lignes par diagnostic")
    audit.add_argument("--output", help="Rapport JSON, sinon sortie standard")
    audit.add_argument("--fail-on-overlap", action="store_true", help="Quitter avec le code 2 si un recouvrement est détecté")
    prepare = commands.add_parser("prepare-series", help="Créer quatre jeux chronologiques sans fenêtres communes")
    prepare.add_argument("--input", required=True, help="JSONL de séries complètes avec series_id et values")
    prepare.add_argument("--output-dir", required=True, help="Nouveau dossier pour train, validation, calibration, test")
    prepare.add_argument("--preset", choices=("tiny", "small", "workstation", "base", "large", "ultra"), default="tiny")
    prepare.add_argument("--config", help="Configuration JSON définissant variables et horizon")
    prepare.add_argument("--history", type=int, required=True)
    prepare.add_argument("--train-end", type=int, required=True, help="Fin exclusive de la période d'entraînement")
    prepare.add_argument("--validation-end", type=int, required=True, help="Fin exclusive de la période de validation")
    prepare.add_argument("--calibration-end", type=int, required=True, help="Fin exclusive de la période de calibration")
    prepare.add_argument("--stride", type=int, default=1, help="Décalage entre fenêtres d'un même jeu")
    prepare.add_argument("--gap", type=int, default=0, help="Positions exclues après chaque coupure")
    demo = commands.add_parser("demo", help="Exécuter les trois modalités et les trois sorties")
    training = commands.add_parser("train", help="Entraînement conjoint réel")
    for command in (inspect, demo, training):
        command.add_argument("--preset", choices=("tiny", "small", "workstation", "base", "large", "ultra"), default="tiny")
        command.add_argument("--config", help="Configuration JSON personnalisée")
    demo.add_argument("--checkpoint")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--output", help="Enregistrer les sorties dans ce fichier JSON")
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--stop-after", type=int, help="Arrêt propre avant la fin du planning, pour reprise")
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--accumulation", type=int, default=1)
    training.add_argument("--loss-normalization", choices=("batch", "observed"),
                          help="Pondération des pertes supervisées : par micro-lot (défaut) ou par cible observée")
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--warmup", type=int, default=10)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--modality-dropout", type=float, default=0.15,
                          help="Probabilité d'omettre chaque modalité pendant l'entraînement")
    training.add_argument("--modality-policy", choices=("dropout", "balanced"),
                          help="dropout (défaut) ou cycle équilibré des combinaisons supervisées ; restauré à la reprise")
    training.add_argument("--series-crop-probability", type=float,
                          help="Probabilité de raccourcir chaque historique (défaut : 0 ; restaurée à la reprise)")
    training.add_argument("--series-min-history", type=int,
                          help="Longueur minimale après troncature (défaut : 8, bornée par la longueur disponible)")
    training.add_argument("--series-missing-rate", type=float,
                          help="Proportion de valeurs masquées dans les exemples sélectionnés (défaut : 0)")
    training.add_argument("--series-mask-probability", type=float,
                          help="Probabilité de sélectionner un exemple pour masquer ses valeurs (défaut : 0.5)")
    training.add_argument("--series-affine-probability", type=float,
                          help="Probabilité de transformer ensemble historique et futur (défaut : 0 ; restaurée à la reprise)")
    training.add_argument("--series-affine-min-scale", type=float,
                          help="Gain positif minimal (défaut : 0.1 ; tirage logarithmique)")
    training.add_argument("--series-affine-max-scale", type=float,
                          help="Gain maximal (défaut : 10)")
    training.add_argument("--series-affine-max-shift", type=float,
                          help="Amplitude maximale du décalage de niveau (défaut : 20)")
    training.add_argument("--device", default="cpu")
    training.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    training.add_argument("--shared-modality-batch-interval", type=int,
                          help="Partager un lot tous les N micro-lots de l'époque (défaut : 1)")
    training.add_argument("--shared-modality-batches", action=argparse.BooleanOptionalAction, default=None,
                          help="Présenter les mêmes lots aux rangs distribués ; restauré à la reprise")
    training.add_argument("--alignment-cross-rank-views", action=argparse.BooleanOptionalAction, default=None,
                          help="Aligner les vues complémentaires de la même ligne entre rangs")
    training.add_argument("--alignment-cross-class-views", action=argparse.BooleanOptionalAction, default=None,
                          help="Aligner les vues de même classe entre rangs (mode class distribué)")
    training.add_argument("--alignment-duplicate-positives", action=argparse.BooleanOptionalAction, default=None,
                          help="Traiter les copies d'une ligne comme positives pour l'alignement")
    training.add_argument("--mixed-modalities", action=argparse.BooleanOptionalAction, default=None,
                          help="Regrouper en lots les exemples de modalités différentes ; restauré à la reprise")
    training.add_argument("--data", help="Jeu d'entraînement JSONL ; sinon données synthétiques")
    training.add_argument("--val-data", help="Jeu de validation JSONL indépendant")
    initialization = training.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="Checkpoint complet dont reprendre l'entraînement")
    initialization.add_argument("--init-from", help="Poids initiaux pour un nouvel entraînement avec un optimiseur neuf")
    training.add_argument("--model-overrides", help="Objet JSON partiel de configuration à appliquer avec --init-from")
    training.add_argument("--output", default="runs/chimera")
    training.add_argument("--save-every", type=int, default=25)
    training.add_argument("--log-every", type=int, default=10)
    training.add_argument("--eval-every", type=int, default=25, help="Fréquence des validations et sélection du meilleur modèle")
    training.add_argument("--monitor", choices=("loss", "observed_loss", "text", "classification", "forecast", "forecast_mae", "forecast_rmse", "accuracy"),
                          help="Mesure de validation pour best.pt et l'arrêt anticipé ; défaut : loss")
    training.add_argument("--patience", type=int, default=0, help="Validations sans progrès avant arrêt ; 0 désactive l'arrêt")
    training.add_argument("--min-delta", type=float, default=0.0, help="Amélioration minimale de la mesure de validation")
    training.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    training.add_argument("--workers", type=int, default=0, help="Processus de chargement des données")
    generate = commands.add_parser("generate", help="Générer du texte avec le cache KV")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", default="Bonjour")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--device", default="cpu")
    predict = commands.add_parser("predict", help="Classer et prévoir depuis ses propres données JSONL")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", help="Fichier JSONL de résultats ; sinon stdout")
    predict.add_argument("--max-new-tokens", type=int, default=0, help="Générer aussi du texte si un prompt est présent")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    predict.add_argument("--calibration", help="Calibration JSON ajustée sur ce checkpoint et ces modalités")
    predict.add_argument("--series-classifier", help="Tête de classification temporelle ajustée séparément")
    predict.add_argument("--series-probability", help="Calibration par température pour cette tête")
    predict.add_argument("--series-set", help="Ensemble conforme de classes")
    predict.add_argument("--series-gate", help="Porte supervisée entre modèle et tête temporelle")
    predict.add_argument("--series-gate-guard", help="Protection calibrée de la porte")
    embed = commands.add_parser("embed", help="Extraire les représentations contrastives des modalités")
    embed.add_argument("--checkpoint", required=True)
    embed.add_argument("--data", required=True)
    embed.add_argument("--output", help="Fichier JSONL de vecteurs ; sinon stdout")
    embed.add_argument("--batch-size", type=int, default=4)
    embed.add_argument("--device", default="cpu")
    embed.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    search = commands.add_parser("search-embeddings", help="Rechercher les voisins cosinus exacts de vecteurs")
    search.add_argument("--queries", required=True)
    search.add_argument("--candidates", required=True)
    search.add_argument("--query-modality", choices=("text", "image", "series"), required=True)
    search.add_argument("--candidate-modality", choices=("text", "image", "series"), required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--query-batch-size", type=int, default=16)
    search.add_argument("--candidate-batch-size", type=int, default=1024)
    search.add_argument("--exclude-self", action="store_true")
    search.add_argument("--skip-missing", action="store_true", help="Ignorer les lignes sans la modalité choisie")
    search.add_argument("--output", help="Fichier JSONL des voisins ; sinon stdout")
    relevance = commands.add_parser("prepare-class-relevance", help="Construire les pertinences par égalité de classe")
    relevance.add_argument("--query-data", required=True)
    relevance.add_argument("--candidate-data", required=True)
    relevance.add_argument("--queries", required=True, help="Vecteurs des requêtes")
    relevance.add_argument("--candidates", required=True, help="Vecteurs des candidats")
    relevance.add_argument("--query-modality", choices=("text", "image", "series"), required=True)
    relevance.add_argument("--candidate-modality", choices=("text", "image", "series"), required=True)
    relevance.add_argument("--exclude-self", action="store_true")
    relevance.add_argument("--batch-size", type=int, default=1024, help="Taille de lecture des vecteurs")
    relevance.add_argument("--output", help="Fichier JSONL de pertinence ; sinon stdout")
    retrieval = commands.add_parser("evaluate-retrieval", help="Mesurer la recherche sur des pertinences explicites")
    retrieval.add_argument("--queries", required=True)
    retrieval.add_argument("--candidates", required=True)
    retrieval.add_argument("--relevance", required=True)
    retrieval.add_argument("--query-modality", choices=("text", "image", "series"), required=True)
    retrieval.add_argument("--candidate-modality", choices=("text", "image", "series"), required=True)
    retrieval.add_argument("--top-k", type=int, default=5)
    retrieval.add_argument("--query-batch-size", type=int, default=16)
    retrieval.add_argument("--candidate-batch-size", type=int, default=1024)
    retrieval.add_argument("--exclude-self", action="store_true")
    retrieval.add_argument("--skip-missing", action="store_true", help="Ignorer les lignes sans la modalité choisie")
    retrieval.add_argument("--output", help="Rapport JSON ; sinon stdout")
    export = commands.add_parser("export", help="Exporter les poids sans l'optimiseur pour l'inférence")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--stable-rms-norm", action="store_true",
                        help="Activer la normalisation RMS stable dans un nouveau checkpoint d'inférence")
    evaluation = commands.add_parser("evaluate", help="Évaluer un checkpoint sur un fichier JSONL indépendant")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    evaluation.add_argument("--output")
    evaluation.add_argument("--calibration", help="Mesurer aussi les intervalles calibrés")
    evaluation.add_argument("--series-classifier", help="Mesurer aussi la tête de classification temporelle")
    evaluation.add_argument("--series-probability", help="Mesurer aussi ses probabilités calibrées")
    evaluation.add_argument("--series-set", help="Mesurer les ensembles conformes de classes")
    evaluation.add_argument("--series-gate", help="Mesurer la porte supervisée")
    evaluation.add_argument("--series-gate-guard", help="Mesurer la protection calibrée de la porte")
    evaluation.add_argument("--iid-intervals", action="store_true",
                            help="Intervalles de Wilson : suppose des exemples de test indépendants et identiquement distribués")
    evaluation.add_argument("--classifier-data", help="Jeu d'ajustement de la tête, à auditer")
    evaluation.add_argument("--probability-data", help="Jeu d'ajustement de la température, à auditer")
    evaluation.add_argument("--set-data", help="Jeu de calibration de l'ensemble conforme, à auditer")
    evaluation.add_argument("--gate-data", help="Jeu d'ajustement de la porte, à auditer")
    evaluation.add_argument("--gate-validation-data", help="Jeu de validation de la porte, à auditer")
    evaluation.add_argument("--guard-data", help="Jeu de calibration de la protection, à auditer")
    set_fit = commands.add_parser("calibrate-series-set", help="Calibrer les ensembles conformes de classes")
    set_fit.add_argument("--checkpoint", required=True)
    set_fit.add_argument("--series-classifier", required=True)
    set_fit.add_argument("--series-probability")
    set_fit.add_argument("--classifier-data", help="Jeu d'ajustement de la tête, à auditer")
    set_fit.add_argument("--probability-data", help="Jeu d'ajustement de la température, à auditer")
    set_fit.add_argument("--data", required=True)
    set_fit.add_argument("--output", required=True)
    set_fit.add_argument("--coverage", type=float, default=.9)
    set_fit.add_argument("--method", choices=("lac", "aps"), default="aps")
    set_fit.add_argument("--scope", choices=("marginal", "class"), default="marginal")
    set_fit.add_argument("--batch-size", type=int, default=32)
    set_fit.add_argument("--device", default="cpu")
    set_fit.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    prob_fit = commands.add_parser("calibrate-series-probability", help="Calibrer les scores d'une tête temporelle")
    prob_fit.add_argument("--checkpoint", required=True)
    prob_fit.add_argument("--series-classifier", required=True)
    prob_fit.add_argument("--classifier-data", help="Jeu d'ajustement de la tête, à auditer")
    prob_fit.add_argument("--data", required=True, help="JSONL de calibration distinct avec class_label")
    prob_fit.add_argument("--output", required=True)
    prob_fit.add_argument("--batch-size", type=int, default=32)
    prob_fit.add_argument("--device", default="cpu")
    prob_fit.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    guard_fit = commands.add_parser("calibrate-series-gate-guard", help="Calibrer les erreurs supplémentaires de la porte")
    guard_fit.add_argument("--checkpoint", required=True)
    guard_fit.add_argument("--series-classifier", required=True)
    guard_fit.add_argument("--series-gate", required=True)
    guard_fit.add_argument("--data", required=True, help="JSONL étiqueté réservé à la protection")
    guard_fit.add_argument("--output", required=True)
    guard_fit.add_argument("--max-harm", type=float, default=.01)
    guard_fit.add_argument("--classifier-data", help="Jeu d'ajustement de la tête, à auditer")
    guard_fit.add_argument("--gate-data", help="Jeu d'ajustement de la porte, à auditer")
    guard_fit.add_argument("--gate-validation-data", help="Jeu de validation de la porte, à auditer")
    guard_fit.add_argument("--batch-size", type=int, default=32)
    guard_fit.add_argument("--device", default="cpu")
    guard_fit.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    gate_fit = commands.add_parser("refine-series-gate", help="Apprendre à choisir entre deux classifieurs temporels")
    gate_fit.add_argument("--checkpoint", required=True)
    gate_fit.add_argument("--series-classifier", required=True)
    gate_fit.add_argument("--data", required=True, help="JSONL étiqueté pour l'ajustement de la porte")
    gate_fit.add_argument("--validation-data", required=True, help="JSONL étiqueté distinct pour sélectionner la porte")
    gate_fit.add_argument("--classifier-data", help="Jeu d'ajustement de la tête, à auditer")
    gate_fit.add_argument("--audit-series", action="store_true", help="Refuser les historiques communs aux jeux fournis")
    gate_fit.add_argument("--output", required=True)
    gate_fit.add_argument("--epochs", type=int, default=50)
    gate_fit.add_argument("--batch-size", type=int, default=32)
    gate_fit.add_argument("--fit-batch-size", type=int, default=128)
    gate_fit.add_argument("--learning-rate", type=float, default=.001)
    gate_fit.add_argument("--threshold", type=float, default=.03)
    gate_fit.add_argument("--seed", type=int, default=0)
    gate_fit.add_argument("--device", default="cpu")
    gate_fit.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    series_fit = commands.add_parser("refine-series-classifier", help="Ajuster une tête temporelle standardisée")
    series_fit.add_argument("--checkpoint", required=True)
    series_fit.add_argument("--data", required=True, help="JSONL de séries avec class_label")
    series_fit.add_argument("--output", required=True)
    series_fit.add_argument("--ridge", type=float, default=.001)
    series_fit.add_argument("--scale-floor", type=float, default=1e-4)
    series_fit.add_argument("--shape-context", type=int, default=0, help="Nombre de segments de forme normalisée ; 0 désactive")
    series_fit.add_argument("--missing-rate", type=float, default=0., help="Taux de masquage des valeurs à l'ajustement")
    series_fit.add_argument("--mask-probability", type=float, default=.5, help="Probabilité de masquer chaque historique")
    series_fit.add_argument("--augmentation-seed", type=int, default=0)
    series_fit.add_argument("--batch-size", type=int, default=32)
    series_fit.add_argument("--device", default="cpu")
    series_fit.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    calibrating = commands.add_parser("calibrate", help="Calibrer les intervalles sur un jeu séparé")
    calibrating.add_argument("--checkpoint", required=True)
    calibrating.add_argument("--data", required=True)
    calibrating.add_argument("--output", required=True)
    calibrating.add_argument("--batch-size", type=int, default=4)
    calibrating.add_argument("--coverage", type=float, help="Couverture cible ; défaut : quantile max - min")
    calibrating.add_argument("--scope", choices=("marginal", "joint"), default="marginal",
                             help="Couverture par coordonnée ou simultanée sur la trajectoire complète")
    calibrating.add_argument("--score-scale", choices=("none", "width"), default="none",
                             help="Normaliser le score joint par la largeur prédite")
    calibrating.add_argument("--scale-floor", type=float, default=1e-6,
                             help="Largeur minimale pour la normalisation (défaut : 1e-6)")
    calibrating.add_argument("--lower-tail-fraction", type=float,
                             help="Part de l'erreur totale allouée à la queue basse ; active les deux corrections")
    calibrating.add_argument("--device", default="cpu")
    calibrating.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    refining = commands.add_parser("refine-forecast", help="Ajuster seulement la correction des historiques courts")
    refining.add_argument("--checkpoint", required=True)
    refining.add_argument("--data", required=True, help="JSONL avec series et future ; autres modalités facultatives")
    refining.add_argument("--output", required=True, help="Nouveau checkpoint d'inférence")
    refining.add_argument("--report", help="Rapport JSON de provenance et d'ajustement")
    refining.add_argument("--context", type=int, default=8, help="Derniers pas utilisés par la correction")
    refining.add_argument("--relative-features", action=argparse.BooleanOptionalAction, default=None,
                          help="Statistiques sans niveau ni échelle absolus ; sinon conserver le mode du parent")
    refining.add_argument("--missing-gate", action=argparse.BooleanOptionalAction, default=None,
                          help="Activer la correction par variable aussi en présence de valeurs manquantes")
    refining.add_argument("--separate-missing", action=argparse.BooleanOptionalAction, default=None,
                          help="Utiliser une correction distincte pour les historiques longs incomplets")
    refining.add_argument("--ridge", type=float, default=0.001)
    refining.add_argument("--batch-size", type=int, default=32)
    refining.add_argument("--device", default="cpu")
    refining.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads doit être > 0")
    torch.set_num_threads(args.threads)
    if args.command == "prepare-series":
        report = prepare_temporal_splits(args.input, args.output_dir, get_config(args), history=args.history,
                                         train_end=args.train_end, validation_end=args.validation_end,
                                         calibration_end=args.calibration_end,
                                         stride=args.stride, gap=args.gap)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "audit-splits":
        splits = {}
        for entry in args.split:
            name, separator, path = entry.partition("=")
            if not separator or not name or not path or name.strip() != name or name in splits:
                parser.error("Chaque --split doit être NOM=FICHIER avec un nom distinct et non vide")
            splits[name] = path
        report = audit_splits(splits, fields=args.fields, interval_fields=args.interval_fields,
                              output=args.output, max_examples=args.max_examples)
        with prediction_stream(args.output) as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if args.fail_on_overlap and any(pair["shared_unique_records"] or
                                        pair.get("intervals", {}).get("groups_with_overlap", 0)
                                        for pair in report["pairs"]):
            raise SystemExit(2)
    elif args.command == "inspect":
        if args.all:
            report = {name: inspect_config(ChimeraConfig.preset(name))
                      for name in ("tiny", "small", "workstation", "base", "large", "ultra")}
        else:
            report = inspect_config(get_config(args))
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        if args.model_overrides and not args.init_from:
            parser.error("model-overrides nécessite init-from ; utiliser config pour un nouveau modèle")
        train(args, get_config(args))
    elif args.command == "demo":
        torch.manual_seed(42)
        model = (load_checkpoint(args.checkpoint, args.device)[0] if args.checkpoint
                 else Chimera(get_config(args)).to(args.device))
        model.eval()
        batch = synthetic_batch(model.config, batch_size=2, seed=7, device=args.device)
        with torch.inference_mode():
            output = model(**batch)
            prompt = torch.tensor([ByteTokenizer().encode("Signal", eos=False)], device=args.device)
            generated = model.generate(prompt, images=batch["images"][:1], series=batch["series"][:1],
                                       max_new_tokens=16, temperature=0)
        report = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "text_logits_shape": list(output.logits.shape),
            "classification_shape": list(output.classification_logits.shape),
            "forecast_shape": list(output.forecast.shape),
            "predicted_classes": output.classification_logits.argmax(-1).tolist(),
            "target_classes": batch["class_labels"].tolist(),
            "quantiles": list(model.config.quantiles),
            "forecast_first_example": output.forecast[0].tolist(),
            "loss": output.loss.item(),
            "losses": {key: value.item() for key, value in output.losses.items()},
            "expert_usage": output.expert_usage.tolist(),
            "generated_text": ByteTokenizer().decode(generated[0]),
            "checkpoint": args.checkpoint,
            "note": "Données synthétiques ; aucune capacité générale n'est démontrée par cette démo.",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "generate":
        model, _ = load_checkpoint(args.checkpoint, args.device)
        tokens = torch.tensor([ByteTokenizer().encode(args.prompt, eos=False)], device=args.device)
        generated = model.generate(tokens, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        print(ByteTokenizer().decode(generated[0]))
    elif args.command == "predict":
        if args.max_new_tokens < 0:
            parser.error("max-new-tokens doit être >= 0")
        if args.series_probability and not args.series_classifier:
            parser.error("series-probability nécessite series-classifier")
        if args.series_set and not args.series_classifier:
            parser.error("series-set nécessite series-classifier")
        if args.series_gate and not args.series_classifier:
            parser.error("series-gate nécessite series-classifier")
        if args.series_gate_guard and not args.series_gate:
            parser.error("series-gate-guard nécessite series-gate")
        sources = (args.data, args.checkpoint, args.calibration, args.series_classifier, args.series_probability,
                   args.series_set, args.series_gate, args.series_gate_guard)
        if args.output and any(same_file(args.output, path) for path in sources if path):
            parser.error("Le fichier de prédictions doit être distinct des fichiers d'entrée")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        calibration = ForecastCalibration.load(args.calibration) if args.calibration else None
        series_classifier = SeriesClassifier.load(args.series_classifier) if args.series_classifier else None
        series_probability = SeriesProbabilityCalibration.load(args.series_probability) if args.series_probability else None
        if calibration is not None:
            calibration.validate_checkpoint(args.checkpoint)
            calibration.validate(model.config, amp=args.amp)
        if series_classifier is not None:
            series_classifier.validate_checkpoint(args.checkpoint)
            series_classifier.validate(model.config, amp=args.amp)
        if series_probability is not None:
            series_probability.validate_files(args.checkpoint, args.series_classifier)
            series_probability.validate(series_classifier, amp=args.amp)
        series_set = SeriesPredictionSet.load(args.series_set) if args.series_set else None
        if series_set is not None:
            series_set.validate_files(args.checkpoint, args.series_classifier, args.series_probability)
            series_set.validate(series_classifier, series_probability, amp=args.amp)
        series_gate = SeriesGate.load(args.series_gate) if args.series_gate else None
        if series_gate is not None:
            series_gate.validate_files(args.checkpoint, args.series_classifier)
            series_gate.validate(series_classifier, amp=args.amp)
        series_gate_guard = SeriesGateGuard.load(args.series_gate_guard) if args.series_gate_guard else None
        if series_gate_guard is not None:
            series_gate_guard.validate_files(args.checkpoint, args.series_classifier, args.series_gate)
            series_gate_guard.validate(series_gate, series_classifier, amp=args.amp)
        model.eval()
        dataset = JointDataset(args.data, model.config, allow_mixed=True)
        if dataset.references_image(args.output):
            parser.error("Le fichier de prédictions doit être distinct des images d'entrée")
        count = 0
        autocast = nullcontext() if args.amp == "none" else torch.autocast(
            torch.device(args.device).type, dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)
        with prediction_stream(args.output) as stream, torch.inference_mode(), autocast:
            for index in range(len(dataset)):
                batch = {key: value.to(args.device) for key, value in collate_records([dataset[index]]).items()
                         if key not in ("labels", "class_labels", "forecast_targets")}
                if calibration is not None:
                    calibration.validate(model.config, batch)
                output = model(**batch)
                row = {
                    "index": index,
                    "class_id": output.classification_logits[0].argmax().item(),
                    "class_probabilities": output.classification_logits[0].float().softmax(-1).tolist(),
                    "quantiles": list(model.config.quantiles),
                    "forecast": output.forecast[0].tolist(),
                }
                if series_classifier is not None:
                    scores = series_classifier.scores(model, batch)[0]
                    row["series_classifier_class_id"] = scores.argmax().item()
                    row["series_classifier_scores"] = scores.tolist()
                    if series_gate is not None:
                        chosen, used, advantage = series_gate.choose(batch, output.classification_logits, scores.unsqueeze(0))
                        if series_gate_guard is not None:
                            used = series_gate_guard.select(advantage)
                            chosen = torch.where(used, scores.unsqueeze(0).argmax(-1),
                                                 output.classification_logits.argmax(-1))
                            row["series_gate_guard_max_harm"] = series_gate_guard.max_harm
                            row["series_gate_guard_threshold"] = series_gate_guard.threshold
                        row["series_gate_class_id"] = chosen[0].item()
                        row["series_gate_selected"] = "series_classifier" if used[0].item() else "base"
                        row["series_gate_advantage"] = advantage[0].item()
                    if series_probability is not None:
                        row["series_classifier_probabilities"] = series_probability.probabilities(scores.unsqueeze(0))[0].tolist()
                        row["series_classifier_inverse_temperature"] = series_probability.inverse_temperature
                    if series_set is not None:
                        probs = (series_probability.probabilities(scores.unsqueeze(0)) if series_probability
                                 else log_probabilities(scores.unsqueeze(0)).exp())
                        mask = series_set.predict(probs, scores.argmax().reshape(1))[0]
                        row["series_prediction_set"] = mask.nonzero().flatten().tolist()
                        row["series_prediction_set_nominal_coverage"] = series_set.coverage
                        row["series_prediction_set_method"] = series_set.method
                        row["series_prediction_set_scope"] = series_set.scope
                if calibration is not None:
                    row["calibrated_interval"] = calibration.interval(output.forecast)[0].tolist()
                    row["calibrated_nominal_coverage"] = calibration.coverage
                    row["calibrated_coverage_scope"] = calibration.scope
                    row["calibrated_score_scale"] = calibration.score_scale
                    if calibration.lower_tail_fraction is not None:
                        row["calibrated_lower_tail_fraction"] = calibration.lower_tail_fraction
                    if calibration.score_scale == "width":
                        row["calibrated_scale_floor"] = calibration.scale_floor
                if args.max_new_tokens and "input_ids" in batch:
                    prompt = batch["input_ids"]
                    if prompt[0, -1] == ByteTokenizer.eos_id:
                        prompt = prompt[:, :-1]
                    context = {key: value for key, value in batch.items()
                               if key not in ("input_ids", "attention_mask")}
                    generated = model.generate(prompt, max_new_tokens=args.max_new_tokens,
                                               temperature=0, **context)
                    row["generated_text"] = ByteTokenizer().decode(generated[0])
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        if args.output:
            print(f"{count} prédictions écrites dans {args.output}")
    elif args.command == "embed":
        if args.batch_size < 1:
            parser.error("batch-size doit être >= 1")
        if args.output and any(same_file(args.output, path) for path in (args.data, args.checkpoint)):
            parser.error("Le fichier de vecteurs doit être distinct des fichiers d'entrée")
        checkpoint_sha = file_sha256(args.checkpoint)
        model, _ = load_checkpoint(args.checkpoint, args.device)
        model.eval()
        dataset = JointDataset(args.data, model.config, allow_mixed=True)
        if dataset.references_image(args.output):
            parser.error("Le fichier de vecteurs doit être distinct des images d'entrée")
        sampler = OrderedModalityBatchSampler(dataset, args.batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_records,
                            generator=torch.Generator().manual_seed(0))
        autocast = nullcontext() if args.amp == "none" else torch.autocast(
            torch.device(args.device).type, dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)
        count = 0
        with prediction_stream(args.output) as stream, torch.inference_mode(), autocast:
            for batch in loader:
                inputs = {key: value.to(args.device) for key, value in batch.items()
                          if key in ("input_ids", "attention_mask", "images", "series", "series_mask")}
                vectors = {name: value.cpu().tolist() for name, value in model.encode_modalities(**inputs).items()}
                size = len(next(iter(vectors.values())))
                for offset in range(size):
                    row = {"format_version": 1, "index": count,
                           "checkpoint_sha256": checkpoint_sha, "dataset_sha256": dataset.fingerprint,
                           "dataset_images_sha256": dataset.image_fingerprint,
                           "alignment_text_view": model.config.alignment_text_view,
                           "amp": args.amp, "device": str(args.device), "batch_size": args.batch_size,
                           "torch_version": str(torch.__version__),
                           "vectors": {name: value[offset] for name, value in vectors.items()}}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
            if file_sha256(args.checkpoint) != checkpoint_sha:
                raise ValueError("Le checkpoint a changé pendant l'extraction")
            verified = JointDataset(args.data, model.config, allow_mixed=True)
            if (verified.fingerprint, verified.image_fingerprint) != (dataset.fingerprint, dataset.image_fingerprint):
                raise ValueError("Le jeu ou ses images ont changé pendant l'extraction")
        if args.output:
            print(f"{count} vecteurs écrits dans {args.output}")
    elif args.command == "search-embeddings":
        if min(args.top_k, args.query_batch_size, args.candidate_batch_size) < 1:
            parser.error("top-k et les tailles de lot doivent être >= 1")
        if args.output and any(same_file(args.output, path) for path in (args.queries, args.candidates)):
            parser.error("Le fichier de résultats doit être distinct des fichiers de vecteurs")
        count = 0
        with prediction_stream(args.output) as stream:
            for row in search_embeddings(args.queries, args.candidates, args.query_modality, args.candidate_modality,
                                         top_k=args.top_k, query_batch_size=args.query_batch_size,
                                         candidate_batch_size=args.candidate_batch_size, exclude_self=args.exclude_self,
                                         skip_missing=args.skip_missing):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        if args.output:
            print(f"{count} recherches écrites dans {args.output}")
    elif args.command == "prepare-class-relevance":
        if args.batch_size < 1:
            parser.error("batch-size doit être >= 1")
        sources = (args.query_data, args.candidate_data, args.queries, args.candidates)
        if args.output and any(same_file(args.output, path) for path in sources):
            parser.error("Le fichier de pertinence doit être distinct des fichiers d'entrée")
        query_dataset = JointDataset(args.query_data, None, allow_mixed=True)
        candidate_dataset = JointDataset(args.candidate_data, None, allow_mixed=True)
        if any(dataset.references_image(args.output) for dataset in (query_dataset, candidate_dataset)):
            parser.error("Le fichier de pertinence doit être distinct des images d'entrée")
        with prediction_stream(args.output) as stream:
            report = write_class_relevance(query_dataset, candidate_dataset, args.queries, args.candidates,
                                           args.query_modality, args.candidate_modality, stream,
                                           exclude_self=args.exclude_self, batch_size=args.batch_size)
        if args.output:
            print(json.dumps(dict(output=args.output, **report), ensure_ascii=False))
    elif args.command == "evaluate-retrieval":
        if min(args.top_k, args.query_batch_size, args.candidate_batch_size) < 1:
            parser.error("top-k et les tailles de lot doivent être >= 1")
        if args.output and any(same_file(args.output, path) for path in
                               (args.queries, args.candidates, args.relevance)):
            parser.error("Le rapport doit être distinct des fichiers d'entrée")
        report = evaluate_retrieval(args.queries, args.candidates, args.relevance,
                                    args.query_modality, args.candidate_modality, top_k=args.top_k,
                                    query_batch_size=args.query_batch_size,
                                    candidate_batch_size=args.candidate_batch_size, exclude_self=args.exclude_self,
                                    skip_missing=args.skip_missing)
        with prediction_stream(args.output) as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if args.output:
            print(f"Évaluation écrite dans {args.output}")
    elif args.command == "export":
        if args.stable_rms_norm and same_file(args.output, args.checkpoint):
            parser.error("Le checkpoint stable doit être distinct du checkpoint source")
        parent_sha = file_sha256(args.checkpoint) if args.stable_rms_norm else None
        model, payload = load_checkpoint(args.checkpoint, mmap=not args.stable_rms_norm,
                                         overrides={"stable_rms_norm": True} if args.stable_rms_norm else None)
        step = payload["step"]
        trainer = dict(payload.get("trainer", {}), inference_only=True)
        if args.stable_rms_norm:
            if file_sha256(args.checkpoint) != parent_sha:
                raise ValueError("Le checkpoint source a changé pendant l'export")
            trainer.update(stable_rms_norm_parent_sha256=parent_sha)
        del payload
        save_checkpoint(args.output, model, step=step, trainer=trainer)
        print(f"Poids d'inférence écrits dans {args.output}")
    elif args.command == "refine-series-classifier":
        if args.batch_size < 1 or args.augmentation_seed < 0:
            parser.error("batch-size doit être > 0 et augmentation-seed >= 0")
        if any(same_file(args.output, source) for source in (args.data, args.checkpoint)):
            parser.error("La sortie doit être distincte des fichiers d'entrée")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        dataset = JointDataset(args.data, model.config)
        if dataset.references_image(args.output):
            parser.error("La sortie doit être distincte des images d'entrée")
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                            generator=torch.Generator().manual_seed(0))
        augmentation = SeriesAugmentation(missing_rate=args.missing_rate, mask_probability=args.mask_probability)
        classifier = fit_series_classifier(model, loader, ridge=args.ridge, scale_floor=args.scale_floor,
                                           amp=args.amp, shape_context=args.shape_context,
                                           augmentation=augmentation if args.missing_rate else None,
                                           augmentation_seed=args.augmentation_seed,
                                           augmentation_batch_size=args.batch_size if args.missing_rate else None,
                                           checkpoint_sha256=file_sha256(args.checkpoint),
                                           dataset_sha256=dataset.fingerprint)
        classifier.validate_checkpoint(args.checkpoint)
        with prediction_stream(args.output) as stream:
            stream.write(json.dumps(classifier.to_dict(), indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(output=args.output, class_counts=classifier.class_counts,
                              checkpoint_sha256=classifier.checkpoint_sha256)))
    elif args.command == "calibrate-series-gate-guard":
        if args.batch_size < 1:
            parser.error("batch-size doit être > 0")
        sources = (args.data, args.checkpoint, args.series_classifier, args.series_gate,
                   args.classifier_data, args.gate_data, args.gate_validation_data)
        if any(same_file(args.output, path) for path in sources if path):
            parser.error("La sortie doit être distincte des fichiers d'entrée")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        classifier = SeriesClassifier.load(args.series_classifier)
        classifier.validate_checkpoint(args.checkpoint)
        gate = SeriesGate.load(args.series_gate)
        gate.validate_files(args.checkpoint, args.series_classifier)
        dataset = JointDataset(args.data, model.config)
        if dataset.references_image(args.output):
            parser.error("La sortie doit être distincte des images d'entrée")
        references = {}
        if args.classifier_data:
            references["tête"] = (args.classifier_data, classifier.dataset_sha256)
        if args.gate_data:
            references["porte"] = (args.gate_data, gate.dataset_sha256)
        if args.gate_validation_data:
            references["validation porte"] = (args.gate_validation_data, gate.validation_sha256)
        audit_series_calibration(args.data, dataset.fingerprint, references, output=args.output, subject="protection")
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                            generator=torch.Generator().manual_seed(0))
        guard = fit_series_gate_guard(model, classifier, gate, loader, max_harm=args.max_harm, amp=args.amp,
                                      checkpoint_sha256=file_sha256(args.checkpoint),
                                      classifier_sha256=file_sha256(args.series_classifier),
                                      gate_sha256=file_sha256(args.series_gate), dataset_sha256=dataset.fingerprint,
                                      feature_batch_size=args.batch_size)
        guard.validate_files(args.checkpoint, args.series_classifier, args.series_gate)
        with prediction_stream(args.output) as stream:
            stream.write(json.dumps(guard.to_dict(), indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(output=args.output, examples=guard.examples, threshold=guard.threshold,
                              unguarded_harms=guard.unguarded_harms, guarded_harms=guard.guarded_harms)))
    elif args.command == "refine-series-gate":
        if args.batch_size < 1:
            parser.error("batch-size doit être > 0")
        sources = (args.data, args.validation_data, args.checkpoint, args.series_classifier, args.classifier_data)
        if any(same_file(args.output, path) for path in sources if path):
            parser.error("La sortie doit être distincte des fichiers d'entrée")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        classifier = SeriesClassifier.load(args.series_classifier)
        classifier.validate_checkpoint(args.checkpoint)
        dataset, validation = JointDataset(args.data, model.config), JointDataset(args.validation_data, model.config)
        if dataset.references_image(args.output) or validation.references_image(args.output):
            parser.error("La sortie doit être distincte des images d'entrée")
        if args.audit_series or args.classifier_data:
            references = {"validation": (args.validation_data, validation.fingerprint)}
            if args.classifier_data:
                references["tête"] = (args.classifier_data, classifier.dataset_sha256)
            audit_series_calibration(args.data, dataset.fingerprint, references, output=args.output, subject="ajustement")
        def loader(source):
            return DataLoader(source, batch_size=args.batch_size, collate_fn=collate_records,
                              generator=torch.Generator().manual_seed(0))
        gate = fit_series_gate(model, classifier, loader(dataset), loader(validation),
                               checkpoint_sha256=file_sha256(args.checkpoint),
                               classifier_sha256=file_sha256(args.series_classifier),
                               dataset_sha256=dataset.fingerprint, validation_sha256=validation.fingerprint,
                               amp=args.amp, threshold=args.threshold, epochs=args.epochs,
                               batch_size=args.fit_batch_size, learning_rate=args.learning_rate, seed=args.seed,
                               feature_batch_size=args.batch_size)
        gate.validate_files(args.checkpoint, args.series_classifier)
        with prediction_stream(args.output) as stream:
            stream.write(json.dumps(gate.to_dict(), indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(output=args.output, train_examples=gate.train_examples,
                              validation_examples=gate.validation_examples, validation_gain=gate.validation_gain,
                              best_epoch=gate.best_epoch)))
    elif args.command == "calibrate-series-set":
        if args.batch_size < 1:
            parser.error("batch-size doit être > 0")
        if args.probability_data and not args.series_probability:
            parser.error("probability-data nécessite series-probability")
        sources = (args.data, args.checkpoint, args.series_classifier, args.series_probability,
                   args.classifier_data, args.probability_data)
        if any(same_file(args.output, source) for source in sources if source):
            parser.error("La sortie doit être distincte des fichiers d'entrée")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        classifier = SeriesClassifier.load(args.series_classifier)
        classifier.validate_checkpoint(args.checkpoint)
        probability = SeriesProbabilityCalibration.load(args.series_probability) if args.series_probability else None
        if probability is not None:
            probability.validate_files(args.checkpoint, args.series_classifier)
        dataset = JointDataset(args.data, model.config)
        if dataset.references_image(args.output):
            parser.error("La sortie doit être distincte des images d'entrée")
        references = {}
        if args.classifier_data:
            references["tête"] = (args.classifier_data, classifier.dataset_sha256)
        if args.probability_data:
            references["température"] = (args.probability_data, probability.dataset_sha256)
        audit_series_calibration(args.data, dataset.fingerprint, references, output=args.output)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                            generator=torch.Generator().manual_seed(0))
        fitted = fit_series_set(model, classifier, loader, amp=args.amp, coverage=args.coverage,
                                method=args.method, scope=args.scope, checkpoint_sha256=file_sha256(args.checkpoint),
                                classifier_sha256=file_sha256(args.series_classifier), probability=probability,
                                probability_sha256=file_sha256(args.series_probability) if probability else None,
                                dataset_sha256=dataset.fingerprint)
        fitted.validate_files(args.checkpoint, args.series_classifier, args.series_probability)
        with prediction_stream(args.output) as stream:
            stream.write(json.dumps(fitted.to_dict(), indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(output=args.output, examples=fitted.examples, threshold=fitted.threshold,
                              scope=fitted.scope, class_thresholds=fitted.class_thresholds)))
    elif args.command == "calibrate-series-probability":
        if args.batch_size < 1:
            parser.error("batch-size doit être > 0")
        sources = (args.data, args.checkpoint, args.series_classifier, args.classifier_data)
        if any(same_file(args.output, source) for source in sources if source):
            parser.error("La sortie doit être distincte des fichiers d'entrée")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        classifier = SeriesClassifier.load(args.series_classifier)
        classifier.validate_checkpoint(args.checkpoint)
        dataset = JointDataset(args.data, model.config)
        if dataset.references_image(args.output):
            parser.error("La sortie doit être distincte des images d'entrée")
        references = {"tête": (args.classifier_data, classifier.dataset_sha256)} if args.classifier_data else {}
        audit_series_calibration(args.data, dataset.fingerprint, references, output=args.output)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                            generator=torch.Generator().manual_seed(0))
        probability = fit_series_probability(model, classifier, loader, amp=args.amp,
                                             checkpoint_sha256=file_sha256(args.checkpoint),
                                             classifier_sha256=file_sha256(args.series_classifier),
                                             dataset_sha256=dataset.fingerprint)
        probability.validate_files(args.checkpoint, args.series_classifier)
        with prediction_stream(args.output) as stream:
            stream.write(json.dumps(probability.to_dict(), indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(output=args.output, examples=probability.examples,
                              inverse_temperature=probability.inverse_temperature)))
    elif args.command == "refine-forecast":
        if args.batch_size < 1 or args.context < 1 or not math.isfinite(args.ridge) or args.ridge <= 0:
            parser.error("batch-size, context et ridge doivent être strictement positifs ; ridge doit être fini")
        inputs = (args.checkpoint, args.data)
        outputs = [path for path in (args.output, args.report) if path]
        if any(same_file(output, source) for output in outputs for source in inputs) or (
                len(outputs) == 2 and same_file(*outputs)):
            parser.error("Les sorties doivent être distinctes entre elles et des fichiers d'entrée")
        overrides = {"forecast_adapter_context": args.context}
        if args.relative_features is not None:
            overrides["forecast_adapter_relative"] = args.relative_features
        if args.missing_gate is not None:
            overrides["forecast_adapter_missing_gate"] = args.missing_gate
        if args.separate_missing is not None:
            overrides["forecast_adapter_separate_missing"] = args.separate_missing
        if args.separate_missing is True and args.missing_gate is None:
            overrides["forecast_adapter_missing_gate"] = True
        if args.missing_gate is False and args.separate_missing is None:
            overrides["forecast_adapter_separate_missing"] = False
        model, payload = load_checkpoint(args.checkpoint, args.device, overrides=overrides)
        step = payload["step"]
        seed = payload.get("trainer", {}).get("seed", 42)
        del payload
        dataset = JointDataset(args.data, model.config)
        if any(dataset.references_image(output) for output in outputs):
            parser.error("Les sorties doivent être distinctes des images d'entrée")
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                            generator=torch.Generator().manual_seed(0))
        report = fit_forecast_adapter(model, loader, ridge=args.ridge, amp=args.amp)
        report.update(parent_checkpoint=str(Path(args.checkpoint).resolve()),
                      parent_sha256=file_sha256(args.checkpoint), parent_step=step,
                      dataset_sha256=dataset.fingerprint, data=str(Path(args.data).resolve()),
                      output=str(Path(args.output).resolve()), batch_size=args.batch_size,
                      device=str(next(model.parameters()).device), torch_version=str(torch.__version__))
        # Le compteur d'optimisation du parent n'est pas un nombre d'étapes ridge.
        # Ne pas transmettre ses anciens scores de validation au nouveau modèle.
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"
        save_checkpoint(args.output, model, step=step,
                        trainer=dict(seed=seed, inference_only=True, forecast_adapter_fit=report))
        if args.report:
            with prediction_stream(args.report) as stream:
                stream.write(content)
        print(content, end="")
    elif args.command in ("evaluate", "calibrate"):
        if args.batch_size < 1:
            parser.error("batch-size doit être > 0")
        sources = [args.data, args.checkpoint]
        if args.command == "evaluate" and args.calibration:
            sources.append(args.calibration)
        if args.command == "evaluate" and args.series_classifier:
            sources.append(args.series_classifier)
        if args.command == "evaluate" and args.series_probability:
            if not args.series_classifier:
                parser.error("series-probability nécessite series-classifier")
            sources.append(args.series_probability)
        if args.command == "evaluate" and args.series_set:
            if not args.series_classifier:
                parser.error("series-set nécessite series-classifier")
            sources.append(args.series_set)
        if args.command == "evaluate" and args.series_gate:
            if not args.series_classifier:
                parser.error("series-gate nécessite series-classifier")
            sources.append(args.series_gate)
        if args.command == "evaluate" and args.series_gate_guard:
            if not args.series_gate:
                parser.error("series-gate-guard nécessite series-gate")
            sources.append(args.series_gate_guard)
        if args.command == "evaluate":
            for option, artifact, name in ((args.classifier_data, args.series_classifier, "classifier-data"),
                                           (args.probability_data, args.series_probability, "probability-data"),
                                           (args.set_data, args.series_set, "set-data"),
                                           (args.gate_data, args.series_gate, "gate-data"),
                                           (args.gate_validation_data, args.series_gate, "gate-validation-data"),
                                           (args.guard_data, args.series_gate_guard, "guard-data")):
                if option and not artifact:
                    parser.error(f"{name} nécessite l'artefact correspondant")
                if option:
                    sources.append(option)
        if args.output and any(same_file(args.output, path) for path in sources):
            parser.error("Le fichier de sortie doit être distinct des fichiers d'entrée")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        dataset = JointDataset(args.data, model.config, allow_mixed=args.command == "evaluate")
        if dataset.references_image(args.output):
            parser.error("Le fichier de sortie doit être distinct des images d'entrée")
        if args.command == "evaluate":
            sampler = OrderedModalityBatchSampler(dataset, args.batch_size)
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_records,
                                generator=torch.Generator().manual_seed(0))
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                                generator=torch.Generator().manual_seed(0))
        if args.command == "calibrate":
            calibration = fit_calibration(model, loader, coverage=args.coverage, amp=args.amp, scope=args.scope,
                                          score_scale=args.score_scale, scale_floor=args.scale_floor,
                                          lower_tail_fraction=args.lower_tail_fraction,
                                          checkpoint_sha256=file_sha256(args.checkpoint),
                                          dataset_sha256=dataset.fingerprint,
                                          dataset_images_sha256=dataset.image_fingerprint,
                                          dataset_images_recorded=True)
            content = json.dumps(calibration.to_dict(), indent=2, allow_nan=False) + "\n"
            with prediction_stream(args.output) as stream:
                stream.write(content)
            print(json.dumps(dict(calibration=args.output, coverage=calibration.coverage,
                                  scope=calibration.scope, score_scale=calibration.score_scale,
                                  lower_tail_fraction=calibration.lower_tail_fraction,
                                  examples=len(dataset), modalities=calibration.modalities)))
            return
        calibration = ForecastCalibration.load(args.calibration) if args.calibration else None
        series_classifier = SeriesClassifier.load(args.series_classifier) if args.series_classifier else None
        series_probability = SeriesProbabilityCalibration.load(args.series_probability) if args.series_probability else None
        if calibration is not None:
            calibration.validate_checkpoint(args.checkpoint)
        if series_classifier is not None:
            series_classifier.validate_checkpoint(args.checkpoint)
        if series_probability is not None:
            series_probability.validate_files(args.checkpoint, args.series_classifier)
        series_set = SeriesPredictionSet.load(args.series_set) if args.series_set else None
        if series_set is not None:
            series_set.validate_files(args.checkpoint, args.series_classifier, args.series_probability)
        series_gate = SeriesGate.load(args.series_gate) if args.series_gate else None
        if series_gate is not None:
            series_gate.validate_files(args.checkpoint, args.series_classifier)
        series_gate_guard = SeriesGateGuard.load(args.series_gate_guard) if args.series_gate_guard else None
        if series_gate_guard is not None:
            series_gate_guard.validate_files(args.checkpoint, args.series_classifier, args.series_gate)
        references = {}
        if args.classifier_data:
            references["tête"] = (args.classifier_data, series_classifier.dataset_sha256)
        if args.probability_data:
            references["température"] = (args.probability_data, series_probability.dataset_sha256)
        if args.set_data:
            references["ensemble"] = (args.set_data, series_set.dataset_sha256)
        if args.gate_data:
            references["porte"] = (args.gate_data, series_gate.dataset_sha256)
        if args.gate_validation_data:
            references["validation porte"] = (args.gate_validation_data, series_gate.validation_sha256)
        if args.guard_data:
            references["protection"] = (args.guard_data, series_gate_guard.dataset_sha256)
        audit_series_calibration(args.data, dataset.fingerprint, references, output=args.output, subject="test")
        report = evaluate(model, loader, args.amp, calibration=calibration, series_classifier=series_classifier,
                          series_probability=series_probability, series_set=series_set, iid_intervals=args.iid_intervals,
                          series_gate=series_gate, series_gate_guard=series_gate_guard)
        if references:
            report["series_audited_sources"] = list(references)
        if calibration is not None:
            identical = (dataset.fingerprint == calibration.dataset_sha256
                         if calibration.dataset_sha256 is not None else None)
            report["calibration_jsonl_identical"] = identical
            # Des fichiers différents peuvent néanmoins partager des exemples.
            report["calibration_data_reused"] = True if identical else None
            report["calibration_external_images_match"] = (
                dataset.image_fingerprint == calibration.dataset_images_sha256
                if calibration.dataset_images_recorded else None)
        if series_classifier is not None:
            report["series_classifier_data_reused"] = (
                True if dataset.fingerprint == series_classifier.dataset_sha256 else None)
        if series_probability is not None:
            report["series_probability_data_reused"] = (
                True if dataset.fingerprint == series_probability.dataset_sha256 else None)
        if series_set is not None:
            report["series_prediction_set_data_reused"] = (
                True if dataset.fingerprint == series_set.dataset_sha256 else None)
        if series_gate is not None:
            report["series_gate_data_reused"] = (
                True if dataset.fingerprint in (series_gate.dataset_sha256, series_gate.validation_sha256) else None)
        if series_gate_guard is not None:
            report["series_gate_guard_data_reused"] = (
                True if dataset.fingerprint == series_gate_guard.dataset_sha256 else None)
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with prediction_stream(args.output) as stream:
                stream.write(content)
        print(content, end="")


if __name__ == "__main__":
    main()
