# Correction des prévisions avec peu d'historique

Les comparaisons précédentes montraient que la prévision à huit pas restait moins précise que la persistance. La connexion temporelle directe redimensionne l'historique vers une fenêtre de référence ; les derniers pas réels apportent une information complémentaire lorsqu'il reste peu d'observations. Ce cycle ajoute une correction affine ajustable séparément, après la tête de prévision.

## Fonctionnement

`forecast_adapter_context: 8` active une couche partagée entre variables. Pour chaque variable, elle reçoit les huit derniers pas valides, centrés sur la dernière valeur finie et normalisés par l'échelle de l'historique, leurs indicateurs de présence, quatre statistiques de niveau et de durée, et les douze médianes initiales normalisées. Les valeurs absentes et le padding sont exclus des statistiques. Les futurs cibles ne font jamais partie de ces entrées.

La couche transforme ces 32 caractéristiques en douze corrections. Elle possède **396 paramètres**, biais compris. Sa sortie est multipliée par l'échelle de la variable et par `max(0, 1 - longueur_valide / forecast_context)`. Avec la référence de 32 pas, elle intervient seulement sur les historiques plus courts. Des valeurs manquantes dans un historique de 32 positions valides ne déclenchent pas cette correction.

La même translation est ajoutée à chaque quantile d'un horizon : leur ordre est conservé. La largeur des intervalles reste identique, aux arrondis près ; leur couverture peut changer et doit être mesurée. Les logits de texte et de classification ne dépendent pas de cette couche. Son contexte est aussi conservé dans le cache de génération.

Le défaut `forecast_adapter_context: 0` désactive le module. Son activation depuis un ancien checkpoint initialise les deux nouveaux tenseurs à zéro. Le chargement reste strict pour les autres poids. Changer explicitement sa taille réinitialise cette correction ; conserver sa taille recharge les coefficients appris.

## Ajustement depuis un JSONL

Chaque ligne contient `series` et `future`. Les historiques peuvent avoir des longueurs différentes et contenir `null` pour les valeurs absentes. D'autres modalités sont acceptées si elles sont présentes dans tout le fichier, comme pour les autres commandes JSONL.

```bash
.venv/bin/python -m chimera --threads 8 refine-forecast \
  --checkpoint runs/workstation-statistics/model.pt \
  --data runs/forecast-refinement/train.jsonl \
  --context 8 --ridge 0.001 --batch-size 64 \
  --output runs/workstation-forecast-refined/model.pt \
  --report runs/workstation-forecast-refined/fit.json

.venv/bin/python -m chimera predict \
  --checkpoint runs/workstation-forecast-refined/model.pt \
  --data runs/forecast-refinement/train.jsonl --output runs/forecast-predictions.jsonl
```

`refine-forecast` accumule les matrices de régression en FP64, sans gradients ni optimiseur du réseau de base. La mémoire des statistiques ne dépend pas du nombre d'exemples. L'objectif est l'erreur quadratique de la médiane dans les unités des cibles, avec pénalisation L2 des coefficients, biais non pénalisé. Les cibles manquantes sont ignorées séparément pour chaque horizon ; chacun doit disposer de cibles avec un historique de moins de 32 pas.

L'ajustement remplace les coefficients d'une correction existante : il utilise la prévision qui la précède. Il restaure le mode du modèle et conserve les anciens coefficients si l'ajustement échoue. Les sorties de la commande doivent être distinctes des entrées. Le checkpoint et le rapport sont chacun écrits de manière atomique.

Le checkpoint produit contient les poids d'inférence, le SHA-256 du parent, celui du fichier d'ajustement, les effectifs et les réglages utilisés. Le compteur d'optimisation du parent est conservé comme provenance ; il ne compte pas des étapes de régression. Les anciens scores de validation du parent ne sont pas attribués au modèle ajusté. Pour un nouvel entraînement conjoint, utiliser ce checkpoint avec `train --init-from`.

## Protocole de mesure

Le modèle parent est `runs/workstation-statistics/model.pt`, après ses 180 étapes de fine-tuning. Le contexte de huit pas et la pénalisation 0.001 ont été choisis pendant le développement, avant le test final. L'ajustement final utilise 4 096 nouveaux exemples synthétiques avec les séries seules, en 64 lots de 64, seeds 50000 à 50063. L'augmentation utilise un flux distinct, seed 1050000, et tire indépendamment les longueurs entre 8 et 32 et le masquage de 20 % des valeurs dans environ la moitié des exemples.

```bash
.venv/bin/python tools/make_forecast_data.py \
  --checkpoint runs/workstation-statistics/model.pt \
  --output runs/forecast-refinement/train.jsonl

.venv/bin/python tools/benchmark_robustness.py \
  --reference-checkpoint runs/workstation-statistics/model.pt \
  --checkpoint runs/workstation-forecast-refined/model.pt --candidate-name refined \
  --seed 18000000 --mask-seed 18100000 \
  --output runs/forecast-refinement-robustness.json

.venv/bin/python tools/benchmark.py \
  --reference-checkpoint runs/workstation-statistics/model.pt \
  --checkpoint runs/workstation-forecast-refined/model.pt \
  --batches 4 --batch-size 16 --seed-start 18200000 \
  --output runs/forecast-refinement-clean.json
```

Sur les 4 096 exemples d'ajustement, 3 936 ont un historique court et contribuent aux statistiques. Les 160 historiques complets ont une porte nulle. Les 760 tenseurs de base restent exactement identiques à ceux du parent. Le modèle passe de 549 495 026 à **549 495 422 paramètres**. Les poids restent locaux sous `runs/` et sont exclus de Git.

Le test principal comporte 256 exemples par condition, seeds 18000000 à 18000015, et des masques issus des seeds 18100000 à 18100015. Il compare les mêmes entrées avant et après ajustement, avec les séries seules puis les trois modalités. Le développement et l'ajustement utilisent d'autres seeds. Un contrôle complémentaire couvre les sept combinaisons de modalités sur 64 exemples complets, seeds à partir de 18200000.

## Résultats indépendants

Une MAE plus basse indique une meilleure prévision de la médiane.

| Historique | Séries : parent | Séries : correction | Trois modalités : parent | Trois modalités : correction | Persistance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complet, 32 pas | 0.2298 | 0.2298 | 0.2200 | 0.2200 | 0.5210 |
| 16 derniers pas | 0.4177 | **0.1372** | 0.3720 | **0.1462** | 0.5210 |
| 8 derniers pas | 0.5437 | **0.2380** | 0.5082 | **0.2520** | 0.5210 |
| 32 pas, 10 % masqués | 0.2489 | 0.2489 | 0.2456 | 0.2456 | 0.5284 |
| 32 pas, 30 % masqués | 0.3181 | 0.3181 | 0.3170 | 0.3170 | 0.5537 |
| 16 pas, 30 % masqués | 0.4723 | **0.2729** | 0.4364 | **0.2600** | 0.5537 |
| 8 pas, 30 % masqués | 0.5680 | **0.3644** | 0.5298 | **0.3526** | 0.5537 |

Avec les séries seules, la MAE baisse de **67.1 % à seize pas** et de **56.2 % à huit pas** face au parent. À huit pas, elle est aussi **54.3 % sous la persistance**. Avec 30 % de valeurs masquées, le gain face au parent reste de **35.8 % à huit pas**. L'ajustement utilise les séries seules ; les mesures avec les trois modalités vérifient son comportement en présence du texte et de l'image.

L'accuracy reste identique dans toutes les conditions, ainsi que la perte de texte avec les trois modalités. Tous les indicateurs restent exactement identiques sur les conditions à 32 pas, même avec masquage, et sur le contrôle des sept combinaisons de modalités avec historique complet.

La correction améliore la médiane sans élargir les quantiles bruts. À huit pas avec les séries seules, leur couverture n'est que de **49.1 %** pour une couverture nominale de 80 % ; avec 30 % de valeurs manquantes, elle vaut **33.0 %**. La précision de la médiane ne suffit donc pas à valider les intervalles.

Ces mesures portent sur les sinusoïdes avec tendance du générateur synthétique. Elles ne prouvent pas un gain sur des séries réelles, d'autres dynamiques ou un changement de distribution. Une seule réalisation du jeu d'ajustement est étudiée. Une calibration existante liée au SHA-256 du parent doit être ajustée à nouveau pour ce checkpoint, sur des données séparées adaptées aux historiques utilisés.

Les [mesures de robustesse](artifacts/forecast-refinement-robustness.json), le [contrôle des sept modalités](artifacts/forecast-refinement-clean.json), la [provenance et l'intégrité des poids](artifacts/forecast-refinement-fit.json) et le [prototype de développement](artifacts/forecast-refinement-development.json) sont conservés dans Git.

## Calibration séparée des intervalles

La calibration conforme existante est réajustée pour ce nouveau checkpoint. Chaque condition utilise 512 exemples de calibration, seeds 19000000 à 19000031, puis 512 exemples de test, seeds 20000000 à 20000031. Pour les historiques masqués, les flux de masquage commencent respectivement à 19100000 et 20100000. Ces exemples sont distincts de l'ajustement affine, du développement et du test principal.

| Historique, entrées | Couverture brute | Couverture calibrée | Largeur brute | Largeur calibrée |
| --- | ---: | ---: | ---: | ---: |
| 8 pas, séries | 50.05 % | **81.16 %** | 0.3599 | 0.9224 |
| 8 pas, trois modalités | 46.06 % | **80.69 %** | 0.3253 | 0.9354 |
| 8 pas, 30 % masqués, séries | 34.10 % | **80.48 %** | 0.3400 | 1.2697 |
| 8 pas, 30 % masqués, trois modalités | 34.08 % | **80.48 %** | 0.3088 | 1.2412 |

La cible vaut 80 %. La calibration élargit les intervalles et diminue le score d'intervalle dans les quatre cas. Elle ne change pas la médiane ni sa MAE. Les couvertures sont marginales par cible, sans garantie simultanée pour toute une trajectoire. Chaque fichier correspond au checkpoint, aux modalités, à la précision et à la distribution d'historiques utilisée pour son ajustement. Les deux conditions ci-dessus ont des calibrations distinctes ; elles ne valident pas une réutilisation sur d'autres distributions.

```bash
.venv/bin/python tools/benchmark_calibration.py \
  --checkpoint runs/workstation-forecast-refined/model.pt \
  --output-dir runs/forecast-calibration-history8 \
  --case history8 --modalities series text+image+series \
  --calibration-seed 19000000 --test-seed 20000000 \
  --calibration-mask-seed 19100000 --test-mask-seed 20100000 --threads 4

# Autre condition : mêmes arguments, avec --case history8_missing30
# et --output-dir runs/forecast-calibration-missing30

# Prédire avec un fichier JSONL de séries de huit pas de la même distribution
.venv/bin/python -m chimera predict \
  --checkpoint runs/workstation-forecast-refined/model.pt --data mon-test-8-pas.jsonl \
  --calibration runs/forecast-calibration-history8/series.json --output predictions.jsonl
```

Les sorties du benchmark de calibration doivent aller dans un dossier vide. Les rapports détaillés sont dans [calibration à huit pas](artifacts/forecast-refinement-calibration-history8.json) et [calibration à huit pas avec masquage](artifacts/forecast-refinement-calibration-history8-missing30.json).

## Vérifications

Les **83 tests passent** sur CPU. Ils couvrent l'identité à l'initialisation, le padding et les trous dans les masques, les variables entièrement absentes, le cache multimodal, le transfert strict et le réajustement reproductible, l'amélioration sur des exemples indépendants, le calcul BF16, les cibles manquantes, la restauration du mode et du RNG, les erreurs sans modification des coefficients et la commande JSONL complète. Le benchmark de calibration vérifie aussi la séparation des seeds de valeurs et de masquage entre calibration et test.

Le [rapport de validation](artifacts/forecast-refinement-acceptance.json) regroupe les résultats, les contrôles d'intégrité et les limites. Les [sorties des tests](artifacts/forecast-refinement-tests.txt) sont conservées avec leur [résumé](artifacts/forecast-refinement-tests.json). L'ajustement et les mesures de ce cycle ont été exécutés sur CPU ; le chemin GPU n'a pas été testé.
