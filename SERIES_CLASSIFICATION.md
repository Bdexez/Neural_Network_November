# Classification temporelle standardisée

Les changements de niveau et d'unités peuvent modifier la classification du modèle, même lorsque la forme du signal reste identique. Une tête facultative ajuste un classifieur linéaire sur la représentation de l'encodeur temporel après standardisation de chaque variable. Elle ne modifie pas les poids du checkpoint et n'agit que sur des exemples dont la seule modalité est `series`.

Pour chaque exemple et variable, la moyenne et l'écart type sont calculés sur les observations finies et les positions valides. L'écart type est borné inférieurement par `scale_floor`. Les valeurs absentes restent absentes ; le padding n'entre ni dans les statistiques ni dans la moyenne finale des représentations temporelles. L'encodeur du checkpoint, en mode évaluation et sans gradient, traite cet historique standardisé. Une régression ridge multiclasse ajuste ensuite les scores sur les étiquettes one-hot : elle régularise les poids, mais pas le biais. Les scores sont des **scores de classification, pas des probabilités calibrées**.

La standardisation supprime les informations de niveau et d'échelle : elle peut être utile si ces quantités représentent seulement un changement d'unités, et nuisible si elles portent une information pertinente pour la classe. Elle n'assure pas une invariance numérique exacte lorsque le plancher est actif, que des valeurs sont arrondies à l'entrée ou que la précision est insuffisante. Une variable entièrement absente reste représentée comme absente. Le nombre d'exemples par classe, le plancher, la précision AMP, les coefficients et les empreintes sont enregistrés dans le JSON de la tête.

## Utilisation

Préparer un JSONL de séries contenant `class_label`, distinct des jeux de sélection et de test. Les lignes non étiquetées sont ignorées ; chaque classe de la configuration doit posséder au moins un exemple étiqueté. Les autres modalités sont refusées pour éviter de laisser croire que la tête les exploite.

```bash
.venv/bin/python -m chimera refine-series-classifier \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --data data/series-head-train.jsonl --ridge 0.001 --scale-floor 1e-4 \
  --output runs/series-classifier.json

.venv/bin/python -m chimera evaluate \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --data data/series-head-test.jsonl --series-classifier runs/series-classifier.json \
  --output runs/series-head-evaluation.json

.venv/bin/python -m chimera predict \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --data data/new-series.jsonl --series-classifier runs/series-classifier.json \
  --output runs/series-head-predictions.jsonl
```

La prédiction conserve `class_id`, `class_probabilities` et `forecast` du modèle d'origine, et ajoute `series_classifier_class_id` et `series_classifier_scores`. L'évaluation conserve ses métriques habituelles et ajoute `series_classifier_accuracy`, l'effectif et une confusion creuse sous forme de triplets `actual`, `predicted`, `count`. `series_classifier_data_reused` vaut `true` si les octets JSONL sont identiques à ceux de l'ajustement, sinon `null` : des fichiers différents peuvent partager des exemples. La tête est liée au SHA-256 exact du checkpoint, à sa configuration et à la précision AMP utilisée à l'ajustement. L'extraction nécessite un passage supplémentaire dans l'encodeur temporel.

## Forme temporelle et masquage facultatifs

`--shape-context 16` ajoute à la représentation moyenne seize segments de valeurs standardisées par variable, leur fraction observée et des moments de différences et de produits aux décalages 1, 2 et 4. Les segments utilisent les positions valides ; dans chaque paire, les deux valeurs doivent être observées. Un segment ou un décalage sans observation correspondante vaut zéro. Ces caractéristiques conservent une partie de la forme temporelle, mais la régression devient plus grande : avec trois variables et `d_model=64`, elle passe de 64 à 187 entrées, et de 390 à 1 128 coefficients avec les biais.

`--missing-rate` et `--mask-probability` permettent de masquer des valeurs uniquement pendant l'ajustement de la tête. Le tirage utilise un générateur dédié ; les cibles et autres données ne sont pas modifiées. La seed, les paramètres de masquage, le device et, pour la commande CLI, la taille des lots sont enregistrés. La reproduction dépend aussi de l'ordre des exemples, du découpage en lots et de l'environnement PyTorch. Les anciens fichiers de tête version 1 restent lisibles ; les caractéristiques de forme ou l'augmentation utilisent le format version 2.

```bash
.venv/bin/python -m chimera refine-series-classifier \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --data data/series-head-train.jsonl \
  --shape-context 16 --missing-rate 0.3 --mask-probability 0.5 --augmentation-seed 49000000 \
  --output runs/series-shape-classifier.json
```

La taille des statistiques ridge croît quadratiquement avec le nombre d'entrées. Le calcul des caractéristiques de forme parcourt chaque historique ; son coût dépend de la longueur des séries et du matériel.

## Calibration des probabilités

Une température facultative transforme les scores de la tête en probabilités par `softmax(score × température_inverse)`. Elle est ajustée sur un jeu étiqueté réservé en minimisant la log-vraisemblance négative (NLL), selon le principe de [temperature scaling de Guo et al.](https://proceedings.mlr.press/v70/guo17a.html). La recherche est déterministe et bornée à une température inverse entre `1e-4` et `1e4`. Une valeur strictement positive conserve l'ordre des scores, sauf égalités dues à la précision numérique ; l'identifiant de classe publié reste calculé directement sur les scores d'origine.

Le fichier de calibration est distinct de la tête et lié aux SHA-256 exacts du checkpoint et de la tête, ainsi qu'à leur précision AMP. Lorsque l'empreinte d'ajustement est disponible, la commande refuse un fichier de calibration dont les octets JSONL sont identiques au jeu d'ajustement de la tête ; des fichiers différents peuvent toutefois partager des exemples. Utiliser des jeux distincts pour ajuster la tête, calibrer la température et tester les probabilités. Les lignes sans étiquette sont ignorées ; au moins une étiquette observée est nécessaire.

L'ajustement ne conserve pas les historiques, mais stocke les scores des exemples étiquetés pour chercher la température : sa mémoire croît avec le produit du nombre d'exemples et du nombre de classes.

```bash
.venv/bin/python -m chimera calibrate-series-probability \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --series-classifier runs/series-classifier.json --data data/series-probability-calibration.jsonl \
  --output runs/series-probability.json

.venv/bin/python -m chimera evaluate \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --data data/series-head-test.jsonl \
  --series-classifier runs/series-classifier.json --series-probability runs/series-probability.json \
  --output runs/series-probability-evaluation.json
```

Avec les deux fichiers, `predict` ajoute `series_classifier_probabilities` et la température inverse tout en conservant les scores et la classe précédents. `evaluate` compare les probabilités `softmax` avant et après calibration : NLL, score de Brier multiclasse (somme des carrés sur les classes) et erreur de calibration des confiances maximales (ECE) sur quinze intervalles de largeur égale. Le rapport contient les effectifs, accuracies et confiances moyennes par intervalle. `series_probability_data_reused` vaut `true` pour le même JSONL de calibration et `null` sinon.

Ces indicateurs sont empiriques. L'ECE dépend du découpage et de l'effectif ; ni une faible ECE ni une baisse de NLL ne garantit la calibration par classe ou sur une distribution différente. Le réglage de température améliore au mieux la NLL du jeu utilisé pour l'ajuster, pas nécessairement celle du test.

### Essai synthétique indépendant

Un essai utilise le checkpoint `tiny` du contrôle joint-affine : 3 072 exemples pour la tête avec caractéristiques de forme, 1 024 autres pour la température et 2 048 pour le test. Les historiques alternent 8, 16 et 32 pas. Un masquage de 30 % des valeurs touche environ la moitié des exemples dans chaque jeu, avec des seeds distinctes. Les scores sont identiques pour les deux transformations appariées testées ici.

| Test | Probabilités | Accuracy | NLL | Brier | ECE, 15 intervalles |
| --- | --- | ---: | ---: | ---: | ---: |
| Identité | Softmax des scores bruts | 56,59 % | 1,5528 | 0,7449 | 0,3318 |
| Identité | Température ajustée | 56,59 % | 1,1066 | 0,5745 | 0,0616 |
| Décalage +20 | Softmax des scores bruts | 56,59 % | 1,5528 | 0,7449 | 0,3318 |
| Décalage +20 | Température ajustée | 56,59 % | 1,1066 | 0,5745 | 0,0616 |

La température inverse ajustée vaut environ 5,86. Cet essai utilise un seul checkpoint et des données synthétiques : il ne valide pas les probabilités sur des données réelles ni en cas de changement de distribution. Les [résultats complets](artifacts/series-probability-study.json) comprennent les intervalles de fiabilité, seeds et empreintes.

```bash
.venv/bin/python tools/benchmark_series_probability.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output runs/series-probability-study.json
```

## Comparaison synthétique indépendante

Le checkpoint `tiny` du contrôle joint-affine est fixé après 180 étapes. La tête est ajustée par ridge sur 3 072 exemples de séries seulement, en alternant des historiques de 8, 16 et 32 pas. Les 512 exemples de test par condition utilisent d'autres seeds ; les transformations et masques d'une même ligne sont appariés. Les observations manquantes sont masquées indépendamment des valeurs ; l'ajustement ne comportait pas ces masques.

| Historique | Transformation | Accuracy du modèle | Accuracy de la tête |
| --- | --- | ---: | ---: |
| 8 pas | Identité | 69,53 % | 51,37 % |
| 8 pas | Décalage +20 | 17,38 % | 51,37 % |
| 16 pas | Identité | 73,83 % | 55,66 % |
| 16 pas | Décalage +20 | 17,38 % | 55,66 % |
| 32 pas | Identité | 93,55 % | 61,72 % |
| 32 pas | Décalage +20 | 17,38 % | 61,72 % |
| 8 pas, 30 % masqués | Identité | 69,53 % | 43,55 % |
| 8 pas, 30 % masqués | Décalage +20 | 17,38 % | 43,55 % |
| 16 pas, 30 % masqués | Identité | 73,05 % | 48,83 % |
| 16 pas, 30 % masqués | Décalage +20 | 17,38 % | 48,83 % |

Dans cet essai, la tête améliore les conditions décalées mais perd nettement en accuracy sur les données d'origine. Ses prédictions ne changent pas pour les cinq transformations positives testées sur les mêmes exemples ; cela ne constitue pas une garantie pour d'autres valeurs, d'autres précisions ou un plancher actif. Les données sont synthétiques, un seul checkpoint est étudié et les capacités multimodales ne sont pas évaluées par cette tête. Les [résultats détaillés](artifacts/series-classifier-study.json) incluent toutes les transformations, les confusions, les paramètres et l'empreinte du checkpoint.

```bash
.venv/bin/python tools/benchmark_series_classifier.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output runs/series-classifier-study.json
```

## Comparaison des caractéristiques de forme

Un second essai utilise de nouvelles seeds, le même checkpoint, 3 072 exemples d'ajustement et 512 exemples de test par condition. Les deux têtes reçoivent exactement les mêmes lots et les mêmes masques d'ajustement : chaque historique a une probabilité de 50 % de subir un masquage indépendant de 30 % de ses valeurs. Les mesures de test avec et sans masquage sont appariées entre transformations.

| Historique | Modèle, identité | Tête moyenne, identité | Tête avec forme, identité | Modèle, +20 | Tête avec forme, +20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 pas | 68,95 % | 47,66 % | 59,96 % | 16,02 % | 59,96 % |
| 16 pas | 73,24 % | 49,61 % | 50,39 % | 16,02 % | 50,39 % |
| 32 pas | 91,99 % | 56,25 % | 59,18 % | 16,02 % | 59,18 % |
| 8 pas, 30 % masqués | 68,95 % | 45,90 % | 51,37 % | 16,02 % | 51,37 % |
| 16 pas, 30 % masqués | 72,07 % | 53,12 % | 47,07 % | 16,02 % | 47,07 % |

La forme améliore la tête moyenne sur plusieurs conditions, mais régresse de 6,05 points à seize pas avec masquage et reste moins précise que le modèle de base sur les séries non transformées. Sur ces exemples, aucune prédiction de la tête avec forme ne change parmi les cinq transformations positives testées ; ce constat ne garantit pas l'invariance numérique en général. L'essai utilise un seul checkpoint et des données synthétiques. Les [résultats complets](artifacts/series-shape-classifier-study.json) contiennent les seeds, les paramètres, les scores comparatifs et l'empreinte du checkpoint.

```bash
.venv/bin/python tools/benchmark_series_classifier.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --shape-context 16 \
  --train-missing-rate 0.3 --train-mask-probability 0.5 \
  --train-seed 48000000 --train-mask-seed 49000000 \
  --test-seed 50000000 --mask-seed 51000000 \
  --output runs/series-shape-classifier-study.json
```
