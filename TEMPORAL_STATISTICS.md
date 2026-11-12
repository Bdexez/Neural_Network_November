# Statistiques temporelles et transfert de poids

Les historiques très courts ont mis en évidence une limite : l'encodeur travaille sur les valeurs normalisées et leurs indicateurs de présence. Le niveau et l'échelle sont réutilisés pour construire la prévision finale, mais les représentations apprises n'y ont pas directement accès. Ce cycle teste une projection des statistiques de l'historique dans ces représentations.

## Projection optionnelle

`temporal_statistics: true` ajoute six informations par variable : moyenne et dernière observation transformées par `asinh`, logarithme de l'échelle, écart normalisé de la dernière observation à la moyenne, fraction de valeurs observées et âge de la dernière observation. Le logarithme de la longueur valide, rapportée à `forecast_context`, complète ce vecteur.

Les âges comptent seulement les positions temporelles valides. Les statistiques ignorent le padding et les valeurs non finies, et ne dépendent jamais des classes ou des futurs cibles. Une variable entièrement absente garde une moyenne nulle, une échelle unitaire, une fraction observée nulle et un âge normalisé égal à un.

La projection linéaire de `6 × variables + 1` valeurs est ajoutée aux tokens avant la convolution et les blocs temporels. Elle est initialisée à zéro. Le modèle `workstation` gagne **9 728 paramètres**, passant de 549 485 298 à **549 495 026**. Le mode est désactivé par défaut : les architectures et les poids existants gardent leur comportement.

## Affiner une configuration existante

`--model-overrides` accepte un fichier JSON partiel avec `--init-from`. Ses valeurs complètent la configuration du checkpoint. Cette option facilite aussi les modifications de poids de pertes ou de dropout sans réécrire le fichier de poids.

```bash
# Activer la projection sans changer les poids des pertes
.venv/bin/python -m chimera train \
  --init-from runs/workstation-robust/model.pt \
  --model-overrides configs/temporal-statistics-overrides.json \
  --steps 100 --lr 0.00006 --output runs/statistics-example
```

Les poids compatibles sont repris strictement. Lors du passage du mode désactivé au mode activé, seule la nouvelle projection est créée à zéro. Un autre poids absent ou une forme incompatible reste une erreur. Les embeddings de texte et la tête de sortie restent partagés, les buffers RoPE sont matérialisés et les poids existants restent chargés avec `mmap`.

Un checkpoint déjà doté de la projection conserve ses valeurs apprises. Sa désactivation peut être demandée explicitement dans les changements de configuration pour un nouvel entraînement. Les changements effectifs sont conservés dans la provenance du checkpoint. `--resume` restaure la configuration et l'optimiseur enregistrés ; il ne permet pas de fournir `--model-overrides`.

La nouvelle projection utilise le groupe d'optimisation temporel, avec un taux dix fois supérieur au taux principal, comme la connexion directe de prévision. Le transfert initialise un optimiseur neuf ; la reprise exacte conserve ensuite cet état et les générateurs aléatoires.

## Comparaison contrôlée

Les essais de développement sur le petit modèle améliorent nettement la classification avec ces statistiques, sans gain clair de prévision à huit pas. Un second essai donne davantage de poids à la prévision. Il motive la comparaison suivante sur le modèle `workstation`, en conservant des exemples de test distincts.

Les deux branches partent de `runs/workstation-robust/model.pt`. Elles exécutent chacune 180 étapes de 8 exemples, seed 2718, avec le même optimiseur neuf, un taux principal de 0.00006, dix étapes d'échauffement, des modalités équilibrées et la même augmentation. Le poids de prévision vaut **3 dans les deux branches**. Seule la projection statistique les distingue au départ ; initialisée à zéro, elle ne change pas leurs prédictions initiales.

```bash
# Branche avec statistiques
.venv/bin/python -m chimera --threads 8 train \
  --init-from runs/workstation-robust/model.pt \
  --model-overrides configs/temporal-statistics-forecast-overrides.json \
  --steps 180 --seed 2718 --batch-size 8 --lr 0.00006 --warmup 10 \
  --modality-policy balanced --series-crop-probability 0.5 \
  --series-min-history 8 --series-missing-rate 0.2 --series-mask-probability 0.5 \
  --save-every 180 --eval-every 180 --output runs/workstation-statistics

# Témoin : mêmes arguments, autre fichier de changements et autre sortie
# --model-overrides configs/forecast-refinement-overrides.json
# --output runs/workstation-statistics-control
```

Les checkpoints finaux sont comparés, sans sélection d'étape sur le test. La perte totale du checkpoint de départ, dont le poids de prévision vaut 1, ne doit pas être comparée directement à celle des branches affinées. Les comparaisons portent sur les mesures de chaque tâche, la MAE et l'accuracy.

Les outils de benchmark acceptent maintenant des architectures différentes lorsque leurs entrées et leurs cibles restent identiques : vocabulaire, longueurs, dimensions d'image, variables temporelles, classes, horizon et quantiles. Ils conservent les configurations comparées dans leurs rapports.

## Mesures indépendantes et limites

Le test principal utilise 256 exemples, seeds 13000000 à 13000015, et des masques issus des seeds 14000000 à 14000015. Ils sont distincts du développement et de l'entraînement. Les deux branches voient les mêmes exemples, longueurs et valeurs masquées. Le budget est identique en étapes et exemples ; la branche statistique possède 9 728 paramètres supplémentaires. Une seule seed d'entraînement est comparée.

Avec les séries seules :

| Historique | MAE témoin | MAE statistiques | Accuracy témoin | Accuracy statistiques |
| --- | ---: | ---: | ---: | ---: |
| Complet, 32 pas | 0.2278 | 0.2296 | 75.39 % | 76.95 % |
| 16 derniers pas | 0.4476 | 0.4256 | 45.70 % | 62.89 % |
| 8 derniers pas | 0.5581 | 0.5505 | 43.36 % | 58.20 % |
| 10 % masqués | 0.2460 | 0.2475 | 70.31 % | 75.00 % |
| 30 % masqués | 0.3158 | 0.3170 | 64.84 % | 70.31 % |
| 16 pas, 30 % masqués | 0.4968 | 0.4790 | 45.31 % | 59.77 % |
| 8 pas, 30 % masqués | 0.5728 | 0.5729 | 42.19 % | 47.27 % |

La classification gagne **17.19 points à seize pas** et **14.84 points à huit pas** face au témoin. La MAE baisse de 4.91 % à seize pas, mais de seulement 1.36 % à huit pas ; plusieurs autres conditions régressent légèrement. La persistance atteint **0.5157** à huit pas complets, meilleure que les **0.5505** du modèle statistique avec les séries seules. Ce mode ne résout donc pas encore la prévision très courte.

Avec les trois modalités :

| Historique de la série | MAE témoin | MAE statistiques |
| --- | ---: | ---: |
| Complet, 32 pas | 0.2192 | 0.2270 |
| 16 derniers pas | 0.3729 | 0.3792 |
| 8 derniers pas | 0.5143 | 0.5133 |
| 10 % masqués | 0.2428 | 0.2498 |
| 30 % masqués | 0.3060 | 0.3132 |
| 16 pas, 30 % masqués | 0.4479 | 0.4414 |
| 8 pas, 30 % masqués | 0.5339 | 0.5316 |

La MAE sur les données complètes avec les trois modalités augmente de **3.57 %**. Le défaut reste sans projection statistique. Les deux checkpoints sont conservés : `runs/workstation-statistics/model.pt` et `runs/workstation-statistics-control/model.pt`. Les poids restent locaux, exclus de Git.

Un contrôle complémentaire sur 64 autres exemples, seeds à partir de 15000000, couvre les sept combinaisons de modalités. La perte de texte avec texte et série passe de 0.03374 pour le témoin à 0.02087 avec statistiques ; avec les trois modalités, de 0.001316 à 0.001112. L'accuracy avec texte ou image reste à 100 % sur ce jeu synthétique facile, dont le texte contient explicitement la classe. Ces résultats ne démontrent pas de capacité générale de langage ou de vision, ni de robustesse sur des séries réelles ou sous changement de niveau et d'échelle hors distribution.

Les résultats complets et toutes les régressions sont dans [statistics-robustness-benchmark.json](artifacts/statistics-robustness-benchmark.json), [statistics-clean-benchmark.json](artifacts/statistics-clean-benchmark.json) et [statistics-paired-clean-benchmark.json](artifacts/statistics-paired-clean-benchmark.json). Le [rapport d'entraînement](artifacts/statistics-training.json) confirme une exposition identique des deux branches : 832 exemples contenant une série, 21 375 positions valides et 57 717 valeurs finies. La nouvelle projection possède 9 728 coefficients non nuls après apprentissage.

```bash
.venv/bin/python tools/benchmark_robustness.py \
  --reference-checkpoint runs/workstation-robust/model.pt \
  --control-checkpoint runs/workstation-statistics-control/model.pt \
  --checkpoint runs/workstation-statistics/model.pt --candidate-name statistics \
  --seed 13000000 --mask-seed 14000000 --output runs/new-statistics-benchmark.json
```

Les mesures utilisent les quantiles bruts. Toute nouvelle version des poids doit être [calibrée séparément](CALIBRATION.md) avant de réutiliser des intervalles conformalisés.

## Vérifications

Les **75 tests passent**. Ils vérifient les prédictions initiales identiques, le gradient de la projection, le padding et les variables entièrement absentes, le transfert strict et la sauvegarde des changements, la reprise exacte avec dropout et accumulation, le cache multimodal et le backward BF16 avec recalcul d'activations. Un test DDP à deux processus CPU exécute aussi le transfert et l'augmentation.

Les anciens checkpoints ont été réévalués sur les mêmes 64 exemples et les sept combinaisons du rapport précédent : la différence maximale de qualité mesurée est **zéro**. Les sorties sont dans [statistics-tests.txt](artifacts/statistics-tests.txt), [statistics-tests.json](artifacts/statistics-tests.json) et [statistics-legacy-compatibility.json](artifacts/statistics-legacy-compatibility.json). Le chemin GPU n'a pas été exécuté ici.

Le [rapport de validation](artifacts/statistics-acceptance.json) regroupe les contrôles locaux et le test distribué. Les [essais de développement](artifacts/statistics-development.json) sont séparés des mesures finales.
