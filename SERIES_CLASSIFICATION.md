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

L'option facultative `--classifier-data` fournit le JSONL d'ajustement de la tête : la commande vérifie son empreinte enregistrée, puis refuse tout historique `series` identique entre ce fichier et celui de calibration, même si les étiquettes ou les futurs diffèrent. La commande d'ensemble accepte aussi `--probability-data` pour comparer le jeu de température ; si les deux références sont fournies, toutes les paires sont comparées. Chaque référence doit correspondre exactement à l'empreinte de son artefact. L'audit conserve en mémoire des empreintes de tous les historiques distincts : son coût augmente avec la taille des jeux. Sans ces options, seule l'identité du fichier entier est détectée.

L'évaluation accepte les mêmes références `--classifier-data` et `--probability-data`, ainsi que `--set-data` pour le jeu de calibration de l'ensemble. Chaque option exige l'artefact correspondant ; les empreintes sont vérifiées avant de comparer toutes les paires de jeux fournis avec le jeu de test. L'évaluation échoue en cas de série partagée, sans remplacer le rapport existant ; en cas de succès, `series_audited_sources` liste les références effectivement contrôlées.

Ces comparaisons exactes ne détectent pas toutes les fuites : notamment, deux fenêtres différentes peuvent partager des pas temporels, et des transformations ou arrondis peuvent masquer une répétition. Auditer séparément les intervalles temporels avec [`audit-splits`](README.md#données-et-api) lorsque les métadonnées sont disponibles ; inclure aussi le jeu de test dans cet audit. Aucun de ces contrôles ne prouve l'indépendance statistique.

L'ajustement ne conserve pas les historiques, mais stocke les scores des exemples étiquetés pour chercher la température : sa mémoire croît avec le produit du nombre d'exemples et du nombre de classes.

```bash
.venv/bin/python -m chimera calibrate-series-probability \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --series-classifier runs/series-classifier.json --data data/series-probability-calibration.jsonl \
  --classifier-data data/series-head-train.jsonl --output runs/series-probability.json

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

## Ensembles conformes de classes

Une tête temporelle figée peut produire un ensemble de classes plutôt qu'une seule réponse. Prévoir un jeu supplémentaire, réservé à cette étape et distinct de ceux utilisés pour ajuster la tête et, le cas échéant, la température :

```bash
.venv/bin/python -m chimera.cli calibrate-series-set \
  --checkpoint model.pt --series-classifier head.json --series-probability temperature.json \
  --data ensemble-calibration.jsonl --coverage 0.9 --method aps --output ensemble.json \
  --classifier-data head-train.jsonl --probability-data temperature-calibration.jsonl
.venv/bin/python -m chimera.cli predict \
  --checkpoint model.pt --series-classifier head.json --series-probability temperature.json \
  --series-set ensemble.json --data test.jsonl --output predictions.jsonl
.venv/bin/python -m chimera.cli evaluate \
  --checkpoint model.pt --series-classifier head.json --series-probability temperature.json \
  --series-set ensemble.json --data test.jsonl --output report.json \
  --classifier-data head-train.jsonl --probability-data temperature-calibration.jsonl \
  --set-data ensemble-calibration.jsonl
```

L'option `--series-probability` est facultative, mais doit être identique à l'ajustement et à l'utilisation : sans elle, la méthode utilise le softmax des scores bruts. Les empreintes lient l'ensemble à la tête, au checkpoint, à la température facultative et à la précision AMP. Le JSONL de calibration ne peut pas être identique à celui de la tête ou de la température ; des fichiers différents peuvent toutefois partager des exemples, ce que l'empreinte ne détecte pas. L'évaluation signale une réutilisation exacte du fichier de calibration, pas l'absence de recouvrement.

`lac` classe les candidats par score `1 - probabilité`. `aps` utilise la somme des probabilités décroissantes jusqu'à chaque candidat, avec départage stable par indice : c'est une variante déterministe, sans randomisation, inspirée des ensembles adaptatifs de [Romano, Sesia et Candès](https://arxiv.org/abs/2006.02544). Le seuil est le rang `ceil((n+1) × couverture)` des scores de calibration ; si ce rang dépasse le nombre d'exemples étiquetés, la commande échoue. Les candidats au seuil sont inclus et la classe de score maximal est toujours ajoutée. L'évaluation donne la couverture globale, la taille moyenne, la fréquence et l'accuracy des ensembles à une classe, ainsi que les couvertures par classe.

Avec un prédicteur fixé avant la calibration et des exemples de calibration et de test échangeables, la construction conforme fournit une couverture **marginale** d'au moins le niveau demandé, sur la répétition des tirages. Elle ne garantit pas la couverture de chaque classe ou de chaque exemple. Un décalage de distribution, une dépendance temporelle ou une sélection de données peuvent invalider l'hypothèse d'échangeabilité.

### Essai synthétique

Sur le checkpoint `tiny` utilisé ci-dessus, quatre plages de seeds servent respectivement à ajuster la tête (3 072 exemples), la température (1 024), le seuil conforme (1 024) et à tester (2 048). Chaque lot contient des historiques de même longueur, alternant 8, 16 et 32 pas ; les exemples ne sont donc pas tous échangeables au sens requis par la garantie. Un masquage indépendant de 30 % est appliqué à la moitié des historiques. Les deux méthodes utilisent les mêmes données, une cible de 90 % et une température commune :

| Méthode | Couverture mesurée | Taille moyenne | Ensembles à une classe | Accuracy de ces ensembles |
| --- | ---: | ---: | ---: | ---: |
| LAC | 88,13 % | 2,32 | 22,12 % | 74,61 % |
| Cumulatif déterministe | 93,02 % | 3,08 | 12,45 % | 76,86 % |

Dans cet essai, la couverture par classe varie de 77,87 % à 100 % pour LAC et de 86,55 % à 100 % pour le score cumulatif. Le décalage apparié de +20 donne les mêmes résultats ; cela ne garantit pas une robustesse générale. Cette mesure unique, synthétique et avec un seul checkpoint ne prouve pas la couverture sur des données réelles. Les [résultats complets](artifacts/series-set-study.json) incluent les seeds, seuils et comptes par classe.

```bash
.venv/bin/python tools/benchmark_series_sets.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output runs/series-set-study.json
```

### Calibration par classe

Pour chercher une couverture conditionnelle à la vraie classe, ajouter `--scope class` à `calibrate-series-set`. La commande calcule un seuil distinct à partir des seuls exemples de calibration de chaque classe ; les seuils et effectifs sont stockés dans l'artefact. Chaque classe doit disposer d'assez d'exemples pour le rang conforme demandé : à 90 %, il en faut au moins neuf par classe, sinon la commande échoue sans remplacer la sortie existante. Le mode par défaut, `--scope marginal`, reste compatible avec les artefacts précédents.

Pour un prédicteur, une méthode et un niveau fixés avant calibration, et des données échangeables dans chaque classe, cette construction offre une couverture marginale conditionnelle à la classe, sur la répétition des tirages. Elle ne garantit pas la couverture pour chaque individu ni pour un artefact de calibration particulier. Cette distinction et le coût de l'ajustement par classe sont discutés par [Ding et al.](https://arxiv.org/abs/2306.09335). Une classe rare peut rendre le calibrage difficile ou imposer des ensembles plus grands.

Sur les mêmes données synthétiques appariées que l'essai précédent, avec 150 à 189 exemples de calibration par classe, les résultats sont les suivants :

| Méthode | Portée | Couverture globale | Taille moyenne | Couvertures par classe |
| --- | --- | ---: | ---: | --- |
| LAC | Marginale | 88,13 % | 2,32 | 100,0 %, 77,9 %, 79,3 %, 87,4 %, 91,1 %, 93,0 % |
| LAC | Par classe | 89,06 % | 2,46 | 95,2 %, 89,9 %, 83,2 %, 86,4 %, 90,3 %, 89,0 % |
| Cumulatif | Marginale | 93,02 % | 3,08 | 100,0 %, 86,6 %, 88,0 %, 94,4 %, 94,9 %, 94,6 % |
| Cumulatif | Par classe | 92,24 % | 2,97 | 100,0 %, 88,2 %, 87,1 %, 91,7 %, 91,1 %, 95,2 % |

Les résultats par classe ne sont pas tous supérieurs à 90 % dans ce tirage : les limites du protocole synthétique et de l'échangeabilité indiquées ci-dessus restent applicables. Les [résultats détaillés](artifacts/series-class-set-study.json) conservent les seuils et les comptes.

```bash
.venv/bin/python tools/benchmark_series_sets.py --scope class \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output runs/series-class-set-study.json
```

## Sélection supervisée entre classifieurs

Une porte facultative choisit, pour chaque historique, la classe du modèle d'origine ou celle de la tête standardisée. Elle utilise les scores des deux classifieurs, la moyenne et l'écart type observés de chaque variable, ainsi que la longueur valide. Un petit réseau apprend à prédire la différence `(tête correcte) − (modèle correct)` ; il choisit la tête lorsque cette différence estimée dépasse le seuil demandé. Cette estimation n'est pas une probabilité. La porte s'inspire du principe des [mélanges d'experts adaptatifs](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), avec deux experts figés et une sélection discrète.

Prévoir quatre jeux distincts : ajustement de la tête, ajustement de la porte, sélection de la porte, puis test final. Les jeux d'ajustement et de validation de la porte doivent représenter les conditions d'utilisation visées ; il est possible d'y inclure des transformations dont l'étiquette reste valide. La commande choisit l'époque maximisant le gain d'accuracy sur le jeu de validation. Si aucune époque ne dépasse le modèle de base sur cette validation, elle enregistre une porte qui choisit toujours le modèle de base. La sélection sur plusieurs époques peut elle-même surajuster la validation : mesurer ensuite le résultat sur un test indépendant.

```bash
.venv/bin/python -m chimera refine-series-gate \
  --checkpoint model.pt --series-classifier head.json --data gate-train.jsonl \
  --validation-data gate-validation.jsonl --classifier-data head-train.jsonl \
  --audit-series --output gate.json
.venv/bin/python -m chimera predict --checkpoint model.pt --series-classifier head.json \
  --series-gate gate.json --data test.jsonl --output predictions.jsonl
.venv/bin/python -m chimera evaluate --checkpoint model.pt --series-classifier head.json \
  --series-gate gate.json --data test.jsonl --classifier-data head-train.jsonl \
  --gate-data gate-train.jsonl --gate-validation-data gate-validation.jsonl --output report.json
```

L'artefact est lié aux SHA-256 du checkpoint et de la tête, ainsi qu'à la précision AMP. Il conserve les empreintes des jeux, les hyperparamètres, la seed, la taille des lots d'extraction, le device et la version de PyTorch ; la reproduction dépend aussi de l'environnement numérique. La commande refuse les jeux dont les octets sont identiques ; `--audit-series` compare aussi les historiques exacts entre ajustement et validation. `--classifier-data` ajoute le jeu de la tête à cet audit et vérifie son empreinte. Ces contrôles ne détectent pas tous les recouvrements temporels ni les transformations d'un même exemple.

Les sorties d'origine restent présentes : `class_id` et `class_probabilities` décrivent toujours le modèle de base. `series_gate_class_id`, `series_gate_selected` et `series_gate_advantage` décrivent la décision supplémentaire ; l'évaluation donne son accuracy, son gain par rapport au modèle d'origine et la fréquence de sélection de la tête. Les éventuelles probabilités et ensembles conformes de la tête restent associés à **la tête**, et non à la décision de la porte. La porte ajoute un passage dans l'encodeur temporel pour la tête et une petite inférence MLP ; son ajustement conserve en mémoire les caractéristiques des exemples étiquetés. Un gain sur la validation ne garantit aucun gain sur une autre distribution.

### Comparaison synthétique de la porte

L'essai utilise un seul checkpoint `tiny` et quatre plages de seeds distinctes : 3 072 exemples pour la tête, 6 144 pour la porte, 1 536 pour sélectionner son époque et 768 exemples de test par condition. Les historiques comptent 8, 16 ou 32 pas ; un masquage indépendant de 30 % touche la moitié des historiques. L'ajustement de la porte mélange des séries inchangées et des décalages et gains aléatoires ; les transformations de test sont appliquées aux mêmes exemples et masques. La validation sélectionne l'époque 26, avec un gain de 13,80 points sur son mélange propre.

| Pas | Transformation de test | Modèle de base | Tête | Porte | Tête sélectionnée |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | Identité | 66,28 % | 52,08 % | 63,54 % | 26,69 % |
| 8 | Décalage +1 | 29,04 % | 52,08 % | 47,53 % | 62,50 % |
| 8 | Gain ×2 | 64,19 % | 52,08 % | 60,16 % | 29,17 % |
| 16 | Identité | 70,44 % | 49,87 % | 68,62 % | 21,61 % |
| 16 | Décalage +1 | 20,70 % | 49,87 % | 45,96 % | 51,69 % |
| 16 | Gain ×2 | 64,97 % | 49,87 % | 58,85 % | 33,72 % |
| 32 | Identité | 88,02 % | 60,16 % | 83,59 % | 20,18 % |
| 32 | Décalage +1 | 16,28 % | 60,16 % | 56,90 % | 80,21 % |
| 32 | Gain ×0,1 | 31,38 % | 60,16 % | 49,74 % | 61,07 % |
| 32 | Gain ×2 | 58,20 % | 60,16 % | 64,06 % | 50,39 % |

La porte améliore fortement certains décalages, mais régresse sur les données inchangées et reste inférieure à la tête pour le gain ×0,1 à 32 pas. Le mélange d'entraînement, le checkpoint et ces séries synthétiques conditionnent les résultats ; aucune amélioration universelle n'est établie. Les [mesures complètes](artifacts/series-gate-study.json) incluent les autres transformations, les seeds et les empreintes.

```bash
.venv/bin/python tools/benchmark_series_gate.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output runs/series-gate-study.json
```
