# Apprentissage et modalités absentes

Ce cycle ajoute un fine-tuning depuis des poids existants, une exposition équilibrée aux modalités et un alignement optionnel tenant compte des classes. Les checkpoints précédents restent chargeables ; aucun paramètre n'est ajouté au réseau.

## Comparaison contrôlée des politiques

Les deux politiques partent des mêmes initialisations `tiny` et reçoivent les mêmes lots synthétiques avant sélection des modalités. Chaque exécution utilise 280 étapes, des lots de 16 exemples, un taux de 0.001, un échauffement de 10 étapes et les seeds 42, 43 et 44. Le dropout de référence vaut 0.3. Le checkpoint final est évalué sur les mêmes 256 exemples indépendants, à partir de la seed 3000000.

Le budget en étapes et exemples est identique ; le nombre de tokens de chaque modalité et le coût de calcul diffèrent volontairement. La politique équilibrée couvre chacune des sept combinaisons une fois par cycle, avec un ordre mélangé déterministe. L'ordre est reconstruit à partir de l'index global des micro-lots, de la seed et du numéro de cycle, sans consommer le RNG d'entraînement.

| Mesure, moyenne sur trois seeds | Dropout | Équilibrage |
| --- | ---: | ---: |
| Accuracy, séries seules | 57.16 % | 58.72 % |
| MAE, séries seules | 0.2661 | 0.3089 |
| MAE, trois modalités | 0.2631 | 0.3372 |
| MAE, texte seul | 0.6054 | 0.4942 |
| Perte de texte, trois modalités | 0.1424 | 0.2458 |

**L'équilibrage ne constitue pas un gain général dans cet essai.** L'écart de classification est faible devant la variabilité entre seeds : écart-type de 9.26 points pour le dropout et 7.22 pour l'équilibrage. La prévision avec historique et la perte de texte se dégradent ; la prévision sans historique s'améliore. Le défaut reste donc le dropout, et les mesures brutes sont conservées dans [modality-study.json](artifacts/modality-study.json).

```bash
.venv/bin/python tools/compare_modality_training.py \
  --seeds 42 43 44 --steps 280 --output-dir runs/new-modality-study
```

Le dossier de sortie doit être vide. Le protocole conserve les commandes, les compteurs d'exposition, les résultats de chaque seed et leur agrégation.

## Fine-tuning du modèle existant

Le checkpoint local de 549 485 298 paramètres, entraîné pendant 240 étapes, reçoit 140 étapes supplémentaires avec une exposition équilibrée, un taux principal de 0.0001, un échauffement de 10 étapes et des lots de 8. Le planning et AdamW repartent de zéro. La seed d'entraînement reste 42 : ces données synthétiques réutilisent les exemples du premier entraînement, avec une nouvelle répartition des modalités. Les jeux de mesure sont distincts.

```bash
.venv/bin/python -m chimera --threads 8 train \
  --init-from runs/workstation/model.pt --steps 140 --batch-size 8 \
  --lr 0.0001 --warmup 10 --modality-policy balanced \
  --eval-every 140 --save-every 140 --output runs/workstation-refined
```

Sur les 64 exemples du benchmark initial :

| Mesure | Poids initiaux | Après fine-tuning |
| --- | ---: | ---: |
| Accuracy, séries seules | 46.88 % | 90.62 % |
| MAE, séries seules | 0.2967 | 0.2270 |
| MAE, texte + séries | 0.2082 | 0.1987 |
| MAE, image + séries | 0.2048 | 0.2109 |
| MAE, trois modalités | 0.2146 | 0.2201 |

La classification avec texte ou image reste à 100 % sur ce jeu synthétique facile. Le texte contient explicitement la classe ; cette accuracy ne mesure donc pas une compréhension générale. Le gain sur les séries seules s'accompagne d'une petite régression de la prévision avec certaines combinaisons. Les deux checkpoints sont conservés. Le gain concerne l'ensemble du fine-tuning ; cet essai ne permet pas de l'attribuer uniquement à la politique équilibrée.

Le rapport complet est dans [refined-benchmark.json](artifacts/refined-benchmark.json). Les poids restent locaux sous `runs/workstation-refined/`, exclus de Git.

Un second ensemble de **256 exemples**, seeds 4000000 à 4000031, confirme le gain sans utiliser les exemples du benchmark initial :

| Mesure indépendante | Poids initiaux | Après fine-tuning |
| --- | ---: | ---: |
| Accuracy, séries seules | 60.16 % | 88.67 % |
| MAE, séries seules | 0.2838 | 0.2282 |
| MAE, trois modalités | 0.2268 | 0.2321 |

L'accuracy gagne 28.52 points et la MAE baisse de 19.6 % avec les séries seules. La MAE avec les trois modalités augmente de 2.35 %. Les intervalles restent imparfaitement calibrés : sur les séries seules, la couverture passe de 86.69 % à 89.75 % pour une couverture nominale de 80 %. Le fine-tuning améliore la médiane, sans résoudre la calibration des quantiles. Le [rapport indépendant](artifacts/refined-independent-benchmark.json) conserve toutes les combinaisons.

```bash
.venv/bin/python tools/benchmark.py \
  --checkpoint runs/workstation-refined/model.pt \
  --reference-checkpoint runs/workstation/model.pt \
  --batches 32 --seed-start 4000000 --threads 8 \
  --output runs/refined-independent.json
```

## Alignement tenant compte des classes

Avec `alignment_mode: "class"`, l'objectif d'alignement considère toutes les correspondances de classe connue comme positives, en plus de la paire d'origine. Cela évite de demander à deux descriptions de la même classe de se repousser uniquement parce qu'elles occupent des lignes différentes du lot. Les labels `-100` ne créent pas de groupe artificiel, et les labels ne sont jamais injectés dans les représentations ou les prédictions.

La perte moyenne les log-probabilités des positifs dans chaque direction. Elle adapte l'idée de [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) à nos paires de modalités. L'option reste désactivée par défaut : les tests vérifient ses gradients et la compatibilité des poids, mais aucun gain de qualité après entraînement n'est revendiqué pour cette option. La température est configurable et les scores sont calculés en FP32 sous AMP.

## Vérification

Les **52 tests passent**. Les nouveaux cas vérifient les sept combinaisons, le maintien des cibles supervisées, le RNG, la partition du calendrier entre rangs, la reprise exacte de l'entraînement équilibré, le fine-tuning avec un optimiseur neuf et les gradients de l'alignement multi-positif. Un test local DDP à deux processus exécute sept étapes avec deux micro-lots accumulés et une validation finale. La CI ajoute un démarrage depuis un checkpoint dans ce scénario distribué. Le [rapport de validation](artifacts/learning-acceptance.json) regroupe les vérifications locales et les compteurs d'exposition.
