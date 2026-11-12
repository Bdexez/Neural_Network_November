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

La couverture demandée vaut par défaut le quantile maximal moins le quantile minimal, soit 0.8 pour `(0.1, 0.5, 0.9)`. Les prédictions ajoutent `calibrated_interval`, de forme `[horizon, variables, 2]`, et `calibrated_nominal_coverage`. Le champ `forecast` garde les quantiles bruts.

Une calibration correspond à **un fichier de checkpoint, une combinaison de modalités et une précision AMP**. Exporter les poids avant de calibrer : les fichiers complet et d'inférence ont des empreintes différentes. Le chargement vérifie le SHA-256 du fichier entier. Si une image ou le texte est ajouté ou retiré, ajuster une calibration pour cette combinaison. Si `--amp bf16` est utilisé à l'ajustement, le conserver pour `evaluate` et `predict`. Ces incompatibilités produisent une erreur explicite.

Les cibles non finies sont ignorées séparément par horizon et variable. Un effectif insuffisant pour le rang demandé produit une erreur, sans interpolation ni troncature du rang. L'ajustement conserve seulement les scores scalaires en mémoire, soit `O(exemples × horizons × variables)` ; il ne conserve pas les images ou les entrées du réseau. Le JSON est remplacé atomiquement après un ajustement réussi.

## Méthode et portée

Nous adaptons la [régression quantile conformalisée de Romano et al.](https://arxiv.org/html/1905.03222v1). Pour chaque horizon et variable, le score est `max(borne_basse - cible, cible - borne_haute)`. Avec `n` cibles observées et une couverture `c`, la correction est la valeur au rang `ceil((n + 1) × c)`, en comptant depuis 1. Elle est soustraite à la borne basse et ajoutée à la borne haute. Elle peut être négative et resserrer l'intervalle.

Notre implémentation élargit cet ensemble si nécessaire pour inclure la médiane et garder des bornes ordonnées. Cette extension ne réduit pas sa couverture. Les quantiles internes du réseau ne sont pas recalibrés individuellement.

La portée statistique est marginale par cible, sous échangeabilité entre exemples de calibration et futurs exemples. Elle ne garantit pas la couverture simultanée d'une trajectoire entière, ni une couverture conditionnelle de 80 % pour chaque entrée. Des fenêtres temporelles successives ou un changement de régime ne satisfont pas automatiquement ces hypothèses.

L'évaluation affiche couverture et largeur, globalement et par horizon. Elle ajoute aussi le [score d'intervalle de Gneiting et Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), orienté vers les petites valeurs : `largeur + 2/(1-c) × erreurs hors intervalle`. C'est le score d'un intervalle central de niveau `c`, y compris lorsque les quantiles de départ sont asymétriques. Il pénalise à la fois les intervalles trop larges et les observations manquées. Comparer les scores à couverture nominale identique.

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
