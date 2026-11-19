# Calibration des intervalles de prévision

Le réseau fournit des quantiles ordonnés, mais leur couverture observée peut différer du niveau demandé. Une calibration séparée produit maintenant des intervalles adaptés aux erreurs du checkpoint choisi. Les poids, les prévisions médianes et les quantiles bruts restent inchangés.

## Utilisation

Choisir le checkpoint définitif, puis préparer un JSONL de calibration contenant des `future` observés. Ce fichier doit être distinct des données d'apprentissage, de sélection du modèle et du test final. Le format est celui des [données personnalisées](README.md).

```bash
# Ajuster sur des exemples réservés à la calibration
.venv/bin/python -m chimera --threads 8 calibrate \
  --checkpoint runs/workstation-refined/model.pt \
  --data data/calibration.jsonl --coverage 0.8 \
  --output runs/forecast-calibration.json

# Mesurer sur un autre fichier
.venv/bin/python -m chimera --threads 8 evaluate \
  --checkpoint runs/workstation-refined/model.pt --data data/test.jsonl \
  --calibration runs/forecast-calibration.json --output runs/calibrated-evaluation.json

# Ajouter les intervalles aux prédictions
.venv/bin/python -m chimera --threads 8 predict \
  --checkpoint runs/workstation-refined/model.pt --data data/new-cases.jsonl \
  --calibration runs/forecast-calibration.json --output runs/calibrated-predictions.jsonl
```

La couverture demandée vaut par défaut le quantile maximal moins le quantile minimal, soit 0.8 pour `(0.1, 0.5, 0.9)`. Les prédictions ajoutent `calibrated_interval`, de forme `[horizon, variables, 2]`, `calibrated_nominal_coverage` et `calibrated_coverage_scope`. Le champ `forecast` garde les quantiles bruts.

Une calibration correspond à **un fichier de checkpoint, une combinaison de modalités et une précision AMP**. Exporter les poids avant de calibrer : les fichiers complet et d'inférence ont des empreintes différentes. Le chargement vérifie le SHA-256 du fichier entier. Si une image ou le texte est ajouté ou retiré, ajuster une calibration pour cette combinaison. Si `--amp bf16` est utilisé à l'ajustement, le conserver pour `evaluate` et `predict`. Ces incompatibilités produisent une erreur explicite.

En mode marginal, les cibles non finies sont ignorées séparément par horizon et variable. Un effectif insuffisant pour le rang demandé produit une erreur, sans interpolation ni troncature du rang. L'ajustement conserve seulement les scores scalaires en mémoire, soit `O(exemples × horizons × variables)` ; il ne conserve pas les images ou les entrées du réseau. Le JSON est remplacé atomiquement après un ajustement réussi.

## Méthode et portée

Nous adaptons la [régression quantile conformalisée de Romano et al.](https://arxiv.org/html/1905.03222v1). Pour chaque horizon et variable, le score est `max(borne_basse - cible, cible - borne_haute)`. Avec `n` cibles observées et une couverture `c`, la correction est la valeur au rang `ceil((n + 1) × c)`, en comptant depuis 1. Elle est soustraite à la borne basse et ajoutée à la borne haute. Elle peut être négative et resserrer l'intervalle.

Notre implémentation élargit cet ensemble si nécessaire pour inclure la médiane et garder des bornes ordonnées. Cette extension ne réduit pas sa couverture. Les bornes calculées en FP64 puis ramenées en FP32 sont arrondies vers l'extérieur lorsque nécessaire pour ne pas exclure une cible située sur la limite. Les quantiles internes du réseau ne sont pas recalibrés individuellement.

Par défaut, `--scope marginal` conserve une portée marginale par cible, sous échangeabilité entre exemples de calibration et futurs exemples. Elle ne garantit pas la couverture simultanée d'une trajectoire entière, ni une couverture conditionnelle de 80 % pour chaque entrée. Des fenêtres temporelles successives ou un changement de régime ne satisfont pas automatiquement ces hypothèses.

`--scope joint` propose une autre cible : couvrir simultanément tous les horizons et variables d'une trajectoire. Pour chaque exemple complet, le score est le maximum des scores CQR sur toutes ses coordonnées ; une correction commune est choisie au même rang conformel, puis appliquée à toutes les bornes. L'utilisation d'un score maximal pour une région couvrant une trajectoire est une approche de [prédiction conformelle multivariée](https://ojs.aaai.org/index.php/AAAI/article/download/30089/31918). Les coordonnées de grandes unités peuvent dominer ce score et rendre les autres intervalles larges.

Seules les trajectoires dont toutes les cibles sont finies contribuent à cet ajustement ; l'effectif commun est enregistré dans `counts`. La garantie de couverture simultanée suppose notamment un modèle fixé avant calibration et l'échangeabilité des trajectoires complètes de calibration avec les trajectoires futures concernées. L'exclusion de cibles manquantes ne garantit pas ces hypothèses si leur présence dépend des valeurs ; aucune garantie n'est revendiquée pour une population différente ou une série temporelle non échangeable. Le fichier joint sans normalisation utilise le format version 2 ; les fichiers marginaux version 1 restent lisibles.

```bash
.venv/bin/python -m chimera calibrate --checkpoint runs/workstation-refined/model.pt \
  --data data/calibration.jsonl --coverage 0.8 --scope joint --output runs/joint-calibration.json
```

Pour des variables d'unités très différentes, `--scope joint --score-scale width` divise chaque score CQR par `max(borne_haute - borne_basse, plancher)` avant de prendre le maximum. La correction commune est ensuite multipliée par cette même largeur à chaque coordonnée. La normalisation dépend uniquement des prédictions ; sous les mêmes hypothèses d'échangeabilité, le raisonnement conformel s'applique au score ainsi construit. Les [résidus multivariés standardisés](https://arxiv.org/abs/2507.20941) illustrent plus généralement l'intérêt des scores adaptés à l'échelle. Cette option ne garantit ni une meilleure efficacité sur chaque jeu de données ni une couverture conditionnelle.

Le plancher `--scale-floor` est strictement positif et vaut `1e-6` par défaut. Choisir une valeur adaptée aux unités si les largeurs prédites peuvent être très petites. Tant que le plancher n'est pas actif, le résultat de calibration suit les changements positifs d'unité appliqués aux prévisions et aux cibles ; lorsqu'il est actif, cette propriété peut ne plus tenir. Les fichiers correspondants utilisent le format version 3 et enregistrent le plancher. Les sorties indiquent `calibrated_score_scale` et, pour cette option, `calibrated_scale_floor`.

```bash
.venv/bin/python -m chimera calibrate --checkpoint runs/workstation-refined/model.pt \
  --data data/calibration.jsonl --coverage 0.8 --scope joint --score-scale width \
  --scale-floor 1e-6 --output runs/joint-width-calibration.json
```

L'évaluation ajoute `forecast_complete_examples`, `forecast_joint_interval_coverage` et `calibrated_joint_interval_coverage` : une trajectoire compte comme couverte si toutes ses coordonnées sont dans leurs intervalles. Les exemples incomplets sont exclus de ce seul diagnostic ; les métriques marginales continuent à utiliser toutes les cibles observées. Une absence de trajectoire complète donne une couverture jointe `null`. Pour `scope: "joint"`, la couverture nominale désigne la trajectoire entière ; le score marginal d'intervalle calibré vaut `null`, car cette cible ne définit pas un niveau nominal par coordonnée.

L'évaluation affiche couverture et largeur, globalement et par horizon, ainsi que la largeur par variable (`forecast_interval_width_by_feature` et `calibrated_interval_width_by_feature`). Elle ajoute aussi le [score d'intervalle de Gneiting et Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), orienté vers les petites valeurs : `largeur + 2/(1-c) × erreurs hors intervalle`. C'est le score d'un intervalle central de niveau `c`, y compris lorsque les quantiles de départ sont asymétriques. Il pénalise à la fois les intervalles trop larges et les observations manquées. Comparer les scores à couverture nominale identique.

## Comparaison avec des échelles différentes

Un second essai utilise le checkpoint `tiny` du contrôle joint-affine après 180 étapes et multiplie les trois variables de séries et leurs cibles par 0,1, 1 et 10. Il utilise 512 trajectoires de calibration (seeds 6100000 à 6100031) et 512 trajectoires de test distinctes (7100000 à 7100031), pour une cible de 80 %. La largeur moyenne agrège des variables exprimées ici à des échelles différentes ; les largeurs par variable montrent mieux les compromis.

| Modalités | Méthode jointe | Couverture de trajectoire | Largeur moyenne | Largeurs par variable |
| --- | --- | ---: | ---: | --- |
| Séries seules | Sans normalisation | 83,98 % | 7,632 | 3,713 ; 4,681 ; 14,502 |
| Séries seules | Par largeur | 79,49 % | 7,442 | 0,201 ; 1,989 ; 20,136 |
| Trois modalités | Sans normalisation | 83,20 % | 7,103 | 3,187 ; 4,146 ; 13,976 |
| Trois modalités | Par largeur | 80,86 % | 7,007 | 0,189 ; 1,857 ; 18,974 |

Dans cet essai, la normalisation réduit fortement la largeur des deux premières variables, mais augmente celle de la troisième. Les couvertures observées varient autour de la cible ; elles ne démontrent pas à elles seules une garantie pour d'autres données. Un seul checkpoint et des données synthétiques ont été utilisés. Les mesures complètes et l'empreinte sont dans [scaled-joint-calibration-study.json](artifacts/scaled-joint-calibration-study.json). Reproduction :

```bash
.venv/bin/python tools/benchmark_joint_calibration.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --calibration-seed 6100000 --test-seed 7100000 --feature-scales 0.1 1 10 \
  --output runs/scaled-joint-calibration-study.json
```

## Comparaison simultanée sur données synthétiques

Un essai fixe le checkpoint `tiny` du mode top-1 historique, seed d'entraînement 42 et 80 étapes. Il utilise 512 trajectoires complètes pour la calibration, seeds de lots 4100000 à 4100031, puis 512 autres pour le test, seeds 5100000 à 5100031. Chaque trajectoire comporte 12 horizons et trois variables ; la cible de couverture est 80 %. Les deux méthodes voient les mêmes prévisions et données de calibration.

| Modalités | Méthode | Couverture par cible | Couverture de trajectoire | Largeur moyenne |
| --- | --- | ---: | ---: | ---: |
| Séries seules | Brute | 83,34 % | 16,21 % | 1,072 |
| Séries seules | Marginale | 80,57 % | 26,76 % | 0,919 |
| Séries seules | Jointe | 97,97 % | 79,10 % | 2,082 |
| Trois modalités | Brute | 83,71 % | 22,66 % | 1,069 |
| Trois modalités | Marginale | 80,90 % | 31,45 % | 1,023 |
| Trois modalités | Jointe | 98,86 % | 81,45 % | 2,011 |

Sur ces exemples, la calibration marginale donne une couverture par cible proche de 80 %, mais couvre beaucoup moins souvent les 36 coordonnées simultanément. La calibration jointe atteint une couverture empirique de trajectoire proche de la cible, au prix d'intervalles environ deux fois plus larges. Une mesure finie peut être inférieure ou supérieure au niveau demandé. Cet essai utilise un seul checkpoint et un générateur synthétique ; il ne valide pas une couverture sur séries réelles, dépendantes ou soumises à un changement de régime.

Les mesures, corrections et empreinte du checkpoint sont dans [joint-calibration-study.json](artifacts/joint-calibration-study.json). Reproduction avec le checkpoint local :

```bash
.venv/bin/python tools/benchmark_joint_calibration.py \
  --checkpoint runs/top1-routing-experts/normalized-42/latest.pt \
  --output runs/nouvelle-calibration-jointe.json
```

## Mesures indépendantes

Le modèle affiné de 549 485 298 paramètres est fixé avant la calibration. Chaque combinaison utilise 512 exemples de calibration, seeds 5000000 à 5000031, puis **512 autres exemples de test**, seeds 6000000 à 6000031. Les lots contiennent 16 exemples. Les sept combinaisons partagent les mêmes exemples avant retrait des modalités. La cible est de 80 % ; les mesures sont obtenues sur CPU.

| Modalités | Couverture brute | Couverture calibrée | Largeur brute | Largeur calibrée |
| --- | ---: | ---: | ---: | ---: |
| Texte | 83.73 % | 81.48 % | 1.3700 | 1.3241 |
| Image | 83.54 % | 81.32 % | 1.3607 | 1.3027 |
| Séries | 89.77 % | 82.24 % | 0.9336 | 0.7699 |
| Texte + image | 81.68 % | 81.26 % | 1.3482 | 1.3434 |
| Texte + séries | 91.15 % | 83.20 % | 0.9242 | 0.7439 |
| Image + séries | 90.05 % | 82.43 % | 0.9488 | 0.8045 |
| Trois modalités | 88.89 % | 82.99 % | 0.9513 | 0.8615 |

Avec les séries seules, la largeur moyenne baisse de **17.54 %** et le score d'intervalle passe de **1.1214 à 1.0413**. Le score s'améliore pour les sept combinaisons. Les couvertures restent au-dessus de la cible dans cet essai : la calibration ne rend pas chaque mesure empirique exactement égale à 80 %. Ces résultats synthétiques ne démontrent pas la calibration sur des données réelles.

Les [résultats bruts](artifacts/calibration-benchmark.json) contiennent les corrections, les effectifs, les mesures par horizon et l'empreinte du checkpoint. Pour reproduire l'essai avec des poids locaux :

```bash
.venv/bin/python tools/benchmark_calibration.py \
  --checkpoint runs/workstation-refined/model.pt \
  --output-dir runs/new-calibration-study
```

Le dossier doit être vide. Le script rejette les plages de seeds qui se chevauchent et vérifie que le fichier de poids reste inchangé.

## Vérifications

Les **63 tests passent** : rang exact, effectifs manquants par cible, correction négative, bornes ordonnées, absence de mutation des poids, reprise de l'état du modèle, invariance au découpage des lots, calcul du score et parcours CLI complet. Les tests rejettent aussi un mauvais checkpoint, des modalités ou une précision différentes, ainsi que les sorties qui écraseraient leurs entrées. Le parcours BF16 a été vérifié sur CPU ; aucun résultat GPU n'est revendiqué.

Les sorties sont conservées dans [calibration-tests.txt](artifacts/calibration-tests.txt), [calibration-tests.json](artifacts/calibration-tests.json) et [calibration-acceptance.json](artifacts/calibration-acceptance.json).
