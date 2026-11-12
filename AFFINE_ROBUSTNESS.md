# Prévision après un changement de niveau ou d'unités

La correction courte du cycle précédent a été ajustée sur des séries synthétiques proches de zéro, d'amplitude voisine de un. Une nouvelle mesure de développement révèle une sensibilité aux changements de niveau : avec seize pas, ajouter 20 à l'historique et au futur fait passer la MAE de 0.13 à 0.82. Le niveau absolu était directement présent dans les caractéristiques de la correction.

## Données augmentées et caractéristiques relatives

`SeriesAffineAugmentation` applique, par exemple et par variable, la même transformation `gain × valeur + décalage` à l'historique et aux futurs cibles. Le gain positif est tiré uniformément dans l'espace logarithmique. Le décalage est tiré uniformément dans un intervalle symétrique. Les valeurs manquantes restent manquantes, les masques restent identiques et les autres modalités sont conservées. Les tirages sont déterministes par index et utilisent un générateur séparé du RNG du modèle.

L'augmentation est désactivée par défaut. Le générateur de JSONL l'expose avec `--affine-probability`, `--min-scale`, `--max-scale`, `--max-shift` et `--affine-seed`. Les futurs sont transformés uniquement comme cibles d'apprentissage ; ils ne sont jamais utilisés pour préparer les caractéristiques du réseau.

`forecast_adapter_relative: true` remplace les quatre statistiques de la correction par : écart entre moyenne et dernière observation divisé par l'échelle, fraction de valeurs observées, âge normalisé de la dernière observation et logarithme de la durée relative. Une variable entièrement absente a une fraction observée nulle et un âge égal à un. Les derniers pas normalisés, leurs indicateurs de présence et les médianes initiales normalisées restent dans les entrées de la couche.

Le nombre de paramètres reste **396**, pour un contexte de huit pas et douze horizons. Les poids de base ne sont pas modifiés par `refine-forecast`. La porte reste nulle à partir de 32 positions temporelles valides et la translation commune préserve l'ordre des quantiles.

Ce mode ne garantit pas une équivariance affine exacte du modèle : les médianes initiales dépendent encore du réseau de base, qui utilise des statistiques absolues, et la normalisation comporte un plancher de variance. Les gains sont mesurés par des transformations explicites des entrées et des cibles.

Le défaut reste `false` pour conserver les checkpoints existants. Changer explicitement ce mode réinitialise seulement les deux tenseurs de correction, même si leur forme est identique : des coefficients appris sur d'autres caractéristiques ne sont pas réinterprétés. Sans option explicite, `refine-forecast` conserve le mode du parent. `--relative-features` et `--no-relative-features` permettent de le choisir.

## Comparaison contrôlée

Les deux nouveaux checkpoints partent de `runs/workstation-statistics/model.pt`. Ils utilisent le même JSONL de 4 096 exemples, le même contexte de huit pas et la même pénalisation ridge 0.001. Seul le choix des quatre statistiques diffère. Le témoin conserve les caractéristiques absolues ; la variante utilise les caractéristiques relatives. La correction précédente, ajustée sans transformations affines, constitue une troisième référence.

Les exemples sont générés par lots de 64 avec les seeds 80000 à 80063. L'augmentation de longueur et de valeurs manquantes commence à la seed 1080000 ; les transformations affines utilisent 2080000. La probabilité de transformation vaut 0.5, les gains vont de 0.1 à 10 et les décalages de -20 à 20. Les longueurs sont tirées entre 8 et 32 ; la moitié des exemples est sélectionnée pour un masquage de 20 % des valeurs. Les trois tirages sont séparés.

Sur les 4 096 exemples, **3 931** ont un historique de moins de 32 pas et contribuent à l'ajustement. Les **760 tenseurs de base sont exactement identiques** au parent pour les deux variantes. Chacune possède 549 495 422 paramètres. Le fichier d'ajustement, son empreinte et les réglages sont conservés dans la provenance.

```bash
.venv/bin/python tools/make_forecast_data.py \
  --checkpoint runs/workstation-statistics/model.pt \
  --output runs/affine-refinement/train.jsonl --seed 80000 \
  --augmentation-seed 1080000 --affine-seed 2080000 --affine-probability 0.5

.venv/bin/python -m chimera --threads 8 refine-forecast \
  --checkpoint runs/workstation-statistics/model.pt \
  --data runs/affine-refinement/train.jsonl --batch-size 64 --relative-features \
  --output runs/workstation-affine-relative/model.pt \
  --report runs/workstation-affine-relative/fit.json

# Témoin : mêmes données et réglages, avec --no-relative-features
# et une sortie sous runs/workstation-affine-control/
```

Le développement utilise d'autres exemples, seeds de test 21000000 à 21000003. Il compare aussi des statistiques centrées conservant le logarithme de l'échelle, ainsi qu'un objectif de régression divisé par l'échelle de l'historique. Le mode relatif avec l'objectif dans les unités des cibles est retenu avant le test final. Les essais de développement ne sont pas des mesures finales.

Le test final utilise **128 exemples par condition**, seeds 23000000 à 23000007. Chaque modèle voit les mêmes signaux, transformations et valeurs masquées. Les masques de l'essai avec 30 % de valeurs absentes utilisent les seeds 23100000 à 23100007. Les cinq transformations fixes sont l'identité, un décalage de +20, des gains de 0.1 et 10, et un gain de 4 accompagné d'un décalage de -50.

Les erreurs sont présentées après remise dans les unités d'origine : la MAE dans les unités transformées est divisée par le gain positif. Cela permet de comparer un changement d'unités sans confondre son gain avec celui du modèle. Les rapports conservent aussi les MAE brutes, la persistance, les mesures par horizon et les scores des autres tâches.

```bash
.venv/bin/python tools/benchmark_affine_forecast.py \
  --reference-checkpoint runs/workstation-forecast-refined/model.pt \
  --control-checkpoint runs/workstation-affine-control/model.pt \
  --checkpoint runs/workstation-affine-relative/model.pt \
  --output runs/affine-benchmark.json

# Essai complémentaire avec masquage : mêmes paramètres et seeds
# --cases history8_missing30 history16_missing30
# --output runs/affine-missing-benchmark.json
```

## Résultats indépendants et compromis

Avec les séries seules, MAE après remise dans les unités d'origine :

| Historique et transformation | Correction précédente | Augmentation, statistiques absolues | Augmentation, statistiques relatives |
| --- | ---: | ---: | ---: |
| 8 pas, identité | 0.2661 | 0.2911 | **0.2627** |
| 8 pas, +20 | 0.5825 | 0.2842 | **0.2533** |
| 8 pas, ×0.1 | 0.4611 | 0.3309 | **0.2748** |
| 8 pas, ×10 | 0.3203 | 0.2599 | **0.2525** |
| 8 pas, ×4 puis -50 | 0.5710 | 0.2802 | **0.2559** |
| 16 pas, identité | **0.1495** | 0.1736 | 0.1754 |
| 16 pas, +20 | 0.8378 | **0.1812** | 0.1849 |
| 16 pas, ×0.1 | 0.3657 | 0.2517 | **0.2373** |
| 16 pas, ×10 | 0.4538 | 0.1625 | **0.1618** |
| 16 pas, ×4 puis -50 | 0.6172 | **0.1795** | 0.1841 |

La persistance vaut environ **0.5077** dans les unités d'origine pour toutes ces conditions. À seize pas avec +20, l'augmentation réduit la MAE de **78.4 %** pour le témoin et de **77.9 %** pour la variante relative. Une grande partie du gain vient donc de l'augmentation. À huit pas sans valeurs manquantes, les statistiques relatives améliorent le témoin dans les cinq transformations et avec les deux combinaisons de modalités testées.

Le résultat à seize pas sans transformation régresse : **+16.1 %** pour le témoin et **+17.3 %** pour la variante relative face à la correction précédente. Avec les trois modalités, cette régression est plus forte : **0.1609 → 0.2324 / 0.2480**. Avec un décalage de +20, les trois modalités passent en revanche de **0.9915 à 0.3345 / 0.3727**. La variante relative ne domine pas le témoin à seize pas.

Avec 30 % de valeurs masquées, les séries seules donnent :

| Historique et transformation | Correction précédente | Augmentation, statistiques absolues | Augmentation, statistiques relatives |
| --- | ---: | ---: | ---: |
| 8 pas, identité | **0.3572** | 0.4100 | 0.4112 |
| 8 pas, +20 | 0.6346 | **0.4105** | 0.4123 |
| 16 pas, identité | **0.2679** | 0.3365 | 0.3394 |
| 16 pas, +20 | 0.8216 | **0.3487** | 0.3500 |

Les cinq transformations et les résultats avec les trois modalités sont conservés dans le rapport de masquage. L'augmentation améliore la résistance aux changements de niveau, avec un coût sur plusieurs conditions proches du jeu d'origine. Le mode relatif reste optionnel et les trois checkpoints sont conservés pour ces usages différents. Le test final n'a pas servi à réajuster les coefficients ou les réglages.

Tous les indicateurs à 32 pas restent exactement identiques entre les trois checkpoints, pour les cinq transformations et les deux combinaisons de modalités. L'accuracy et la perte de texte restent identiques entre modèles dans toutes les conditions. La classification à partir des séries reste cependant sensible au niveau : avec seize pas, son accuracy passe de 62.5 % sans transformation à 13.3 % après +20. La correction de prévision ne résout pas cette limite du réseau de base.

Les poids restent dans l'espace de travail sous `runs/`, exclus de Git. Ces transformations ne changent ni les fréquences ni la dynamique du générateur synthétique. Elles ne constituent pas une validation sur des séries réelles ou sur de nouveaux processus temporels. Une seule réalisation du jeu d'ajustement et 128 exemples sous-jacents sont utilisés ; les conditions sont appariées et ne représentent pas des cohortes indépendantes.

Les intervalles sont bruts et aucune couverture nominale n'est revendiquée pour ces nouveaux checkpoints. Les fichiers de calibration du cycle précédent sont liés à un autre SHA-256. Ajuster une nouvelle calibration sur des exemples séparés de la distribution d'utilisation avant de mesurer ses intervalles sur un autre jeu.

## Vérifications et preuves

Les **88 tests passent**. Les nouveaux cas vérifient la transformation couplée des historiques et des futurs, la préservation des valeurs absentes et des autres modalités, la reproductibilité sans modification du RNG global, les bornes invalides, les dépassements numériques, les statistiques relatives sous translation, l'âge des observations, le transfert entre modes, le cache, la commande d'ajustement et le benchmark avec protection des entrées.

Les résultats complets sont dans [affine-benchmark.json](artifacts/affine-benchmark.json) et [affine-missing-benchmark.json](artifacts/affine-missing-benchmark.json). Le [rapport d'acceptation](artifacts/affine-acceptance.json) liste aussi toutes les régressions de la variante relative face à la correction précédente. La [provenance des deux ajustements](artifacts/affine-training.json), les [essais de développement](artifacts/affine-development.json) et les [sorties des tests](artifacts/affine-tests.txt) permettent de vérifier le protocole. Les entraînements et mesures de ce cycle ont été exécutés sur CPU.
